
package uk.ac.bham.cs.kdebt.cloudsim.elasticity;

import burlap.behavior.learningrate.SoftTimeInverseDecayLR;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.valuefunction.ConstantValueFunction;
import burlap.behavior.valuefunction.QFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.singleagent.SADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import uk.ac.bham.cs.kdebt.burlap.environment.CloudEnvironment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import uk.ac.bham.cs.kdebt.burlap.learning.QLearningDebtOptimus;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.ex.IAutoscalingPolicy;
import org.cloudbus.cloudsim.ex.MonitoringBorkerEX;
import uk.ac.bham.cs.kdebt.burlap.domain.CloudDomainGenerator;
import uk.ac.bham.cs.kdebt.burlap.domain.ElasticityAction;
import uk.ac.bham.cs.kdebt.burlap.domain.StateVariableKey;
import uk.ac.bham.cs.kdebt.burlap.learning.QLearningDebt;
import uk.ac.bham.cs.kdebt.burlap.state.CloudState;
import static uk.ac.bham.cs.kdebt.cloudsim.KDebt.SUMMARY_PERIOD_LEN;
import uk.ac.bham.cs.kdebt.cloudsim.broker.DebtTracker;
import uk.ac.bham.cs.kdebt.cloudsim.broker.LearningBroker;
import uk.ac.bham.cs.kdebt.cloudsim.util.ComparatorByMips;
import uk.ac.bham.cs.kdebt.cloudsim.util.ScalabilityEvent;
import uk.ac.bham.cs.kdebt.cloudsim.util.ScalabilityLog;
import uk.ac.bham.cs.kdebt.cloudsim.util.Util;
import uk.ac.bham.cs.kdebt.cloudsim.vm.MachineTypeCharacteristics;
import uk.ac.bham.cs.kdebt.cloudsim.vm.ScalableMonitoredVMex;
import uk.ac.bham.cs.kdebt.memory.IDebtAwareLearningAgent;


public class DebtAwareLearningElasticity implements IAutoscalingPolicy
{

    private double coolDownPeriod;//       = 5.0; //seconds     
    private double lastActionTime = -1;
    //private double highQueuedRequests; 
    //private double lowQueuedRequests;
    private SADomain domain;
    private CloudEnvironment environment;
    //private QLearningDebt qLearningDebt;
    private QLearningDebtOptimus qLearningDebt;
    private List<Episode> episodes;
    private DebtTracker debtTracker;
    private double lazyVMThreshold;

    public DebtAwareLearningElasticity(/*double highQueuedRequests, double lowQueuedRequests,*/double coolDownPeriod, double lazyVMThreshold)
    {
        super();
        //this.highQueuedRequests = highQueuedRequests;
        //this.lowQueuedRequests  = lowQueuedRequests;
        this.coolDownPeriod = coolDownPeriod;
        this.lazyVMThreshold = lazyVMThreshold;
        CloudDomainGenerator domainGenerator = new CloudDomainGenerator();
        domain = domainGenerator.generateDomain();
        environment = new CloudEnvironment(coolDownPeriod, lazyVMThreshold);
        double gamma = 0.99;
        double qInitDoubleValue = 0.0;
        QFunction qInit = new ConstantValueFunction(qInitDoubleValue);
        double learningRate = 0.1;
        double epsilon = 0.1; // EpsilonGreedy
        //qLearningDebt = new QLearningDebt(domain, gamma, new SimpleHashableStateFactory(), qInitDoubleValue, learningRate);
        qLearningDebt = new QLearningDebtOptimus(domain, gamma, new SimpleHashableStateFactory(), qInit, learningRate, epsilon);
        double initialLearningRate = Math.floor(Double.parseDouble(Util.getPropertyAsString("initialLearningRate").trim())); //1.0;
        double decayConstantShift = Math.floor(Double.parseDouble(Util.getPropertyAsString("decayConstantShift").trim())); //0.05;
        double minimumLearningRate = Math.floor(Double.parseDouble(Util.getPropertyAsString("minimumLearningRate").trim())); //0.1;
        qLearningDebt.setLearningRateFunction(new SoftTimeInverseDecayLR(initialLearningRate, decayConstantShift, minimumLearningRate));
        episodes = new ArrayList();
    }

    protected Action getNextAction(MonitoringBorkerEX broker)
    {
        Action action = qLearningDebt.decideAction(environment);
        /*
        boolean hasAction = false;
        IDebtAwareLearningAgent learningAgent = (IDebtAwareLearningAgent) broker;
        while (!hasAction)
        {
            if (ElasticityAction.valueOf(action.actionName()) == ElasticityAction.RELEASE)
            {
                int runningVms = learningAgent.getRunningVms();
                if (runningVms > 1)
                {
                    break;
                }
            }
            else
            {
                break;
            }
            action = qLearningDebt.decideAction(environment);
        }*/
        return action;
    }

    @Override
    public void scale(MonitoringBorkerEX broker)
    {
        LearningBroker learningBroker = (LearningBroker) broker;

        double currentTime = CloudSim.clock();
        boolean analyseAction = getLastActionTime() < 0 || getLastActionTime() + getCoolDownPeriod() < currentTime;
        Action action;
        if (getLastActionTime() < 0)
        {
            environment.setDebtAwareLearningAgent((IDebtAwareLearningAgent) broker);
        }
        if (analyseAction)
        {
            if (getLastActionTime() < 0)
            {
                action = getNextAction(broker);
            }
            else
            {
                Episode episode = qLearningDebt.runLearningEpisode(environment, debtTracker);
                episodes.add(episode);

                action = getNextAction(broker);
            }
            CloudState cloudState = (CloudState) environment.currentObservation();
            Integer queuedJobsInTermsOfVms = (Integer) cloudState.get(StateVariableKey.VMSWITHQUEUEDJOBS);
            Integer vmsCloseToNextBillingCycle = (Integer) cloudState.get(StateVariableKey.VMSWITHOUTQUEUEDJOBSCLOSETONEXTBILLINGCYCLE);
            String vmData = debtTracker.getAllVmStatusAsString();
            debtTracker.logLearning(String.format("DETAIL {%d,%d}.%s \t time=%5.2f \t %s", queuedJobsInTermsOfVms, vmsCloseToNextBillingCycle, action.actionName(), currentTime, vmData));

            if (action.actionName().equals(ElasticityAction.LAUNCH.toString()))
            {
                MachineTypeCharacteristics vmType = MachineTypeCharacteristics.getMachineTypeCharacteristicsByType(Util.getPropertyAsString("machinetype"));
                ScalableMonitoredVMex newVm = Util.createVM("VM_" + currentTime, broker.getId(), vmType, SUMMARY_PERIOD_LEN);
                broker.createVmsAfter(Arrays.asList(newVm), 0);
                debtTracker.addAdaptationDecision(currentTime, ScaleOperation.SCALE_OUT,
                        ((LearningBroker) broker).getRunningVmList(),//  vmAllList, 
                        Arrays.asList(newVm));
                ((LearningBroker) broker).migrateQueuedCloudlets(((LearningBroker) broker).getVmsWithQueuedRequests(), //vmLaunchers, 
                        Arrays.asList(newVm));
                ScalabilityEvent event = new ScalabilityEvent(newVm, currentTime, ScalabilityEvent.ScalingType.UP);
                ScalabilityLog.registerEvent(event);

            }
            else if (action.actionName().equals(ElasticityAction.RELEASE.toString()))
            {
                List<ScalableMonitoredVMex> vmTerminators = ((LearningBroker) broker).getLazyVmsCloseToNextBillingCycleList(coolDownPeriod, lazyVMThreshold);
                if (vmTerminators.isEmpty())
                {
                    vmTerminators.addAll(((LearningBroker) broker).getRunningVmList());
                    Collections.sort(vmTerminators, new ComparatorByMips());
                    vmTerminators.remove(vmTerminators.size() - 1);
                }
                List<ScalableMonitoredVMex> vmTerminatorsClone = new ArrayList<>();
                vmTerminatorsClone.addAll(vmTerminators);
                //debtTracker.addAdaptationDecision(currentTime, ScaleOperation.SCALE_IN, vmAllList, tmpTerminators);
                /*List<ScalableMonitoredVMex> tmpTerminators = new ArrayList<>();
                tmpTerminators.add(vmCandidateToTerminate);
                if (!vmTerminators.isEmpty())
                {
                    vmTerminators.stream().forEach((item)
                            -> 
                            {
                                if (!tmpTerminators.contains(item))
                                {
                                    tmpTerminators.add(item);
                                }
                    });
                }*/
                debtTracker.addAdaptationDecision(currentTime, ScaleOperation.SCALE_IN,
                        ((LearningBroker) broker).getRunningVmList(), //vmAllList, 
                        vmTerminatorsClone); //tmpTerminators
                //TODO why only one virtual machine
                //lastActionTime = currentTime;
                double delta = 0.000001;
                ScalableMonitoredVMex vmCandidateToTerminate = vmTerminators.get(0);
                vmTerminators.remove(vmCandidateToTerminate);
                if (vmTerminators.size() > 0)
                {
                    ((LearningBroker) broker).migrateNonFinishedCloudlets(vmCandidateToTerminate, vmTerminators);
                }
                else
                {
                    //TODO where to put the pending cloudlets
                    List<ScalableMonitoredVMex> availableVmList = new ArrayList<>();
                    availableVmList.addAll(((LearningBroker) broker).getRunningVmList()); //vmAllList
                    availableVmList.remove(vmCandidateToTerminate);
                    ((LearningBroker) broker).migrateNonFinishedCloudlets(vmCandidateToTerminate, availableVmList);
                }
                broker.destroyVMsAfter(Arrays.asList(vmCandidateToTerminate), delta);
                ScalabilityEvent event = new ScalabilityEvent(vmCandidateToTerminate, currentTime, ScalabilityEvent.ScalingType.DOWN);
                ScalabilityLog.registerEvent(event);
            }
            else //if (action.actionName().equals(ElasticityAction.MAINTAIN.toString()))
            {
                debtTracker.addAdaptationDecision(currentTime, ScaleOperation.NO_SCALE, ((LearningBroker) broker).getRunningVmList(), new ArrayList<>());
            }
            lastActionTime = currentTime;
        }
    }

    /**
     * @return the lastActionTime
     */
    public double getLastActionTime()
    {
        return lastActionTime;
    }

    /**
     * @return the coolDownPeriod
     */
    public double getCoolDownPeriod()
    {
        return coolDownPeriod;
    }

    public void setDebtTracker(DebtTracker debtTracker)
    {
        this.debtTracker = debtTracker;
    }
}
