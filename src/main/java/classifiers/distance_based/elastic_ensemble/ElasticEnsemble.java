package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.elastic_ensemble.selection.BestPerTypeSelector;
import classifiers.distance_based.elastic_ensemble.selection.Selector;
import classifiers.distance_based.knn.Knn;
import classifiers.template_classifier.TemplateClassifier;
import distances.derivative_time_domain.ddtw.CachedDdtw;
import distances.time_domain.dtw.Dtw;
import distances.derivative_time_domain.wddtw.CachedWddtw;
import distances.time_domain.erp.Erp;
import distances.time_domain.lcss.Lcss;
import distances.time_domain.msm.Msm;
import distances.time_domain.twe.Twe;
import distances.time_domain.wdtw.Wdtw;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import timeseriesweka.classifiers.ensembles.voting.MajorityVoteByConfidence;
import timeseriesweka.classifiers.ensembles.voting.ModuleVotingScheme;
import timeseriesweka.classifiers.ensembles.weightings.ModuleWeightingScheme;
import timeseriesweka.classifiers.ensembles.weightings.TrainAcc;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.Function;

public class ElasticEnsemble extends TemplateClassifier {

    public static List<Function<Instances, ParameterSpace>> getClassicParameterSpaceGetters() {
        return new ArrayList<>(Arrays.asList(
            instances -> Dtw.euclideanParameterSpace(),
            instances -> Dtw.fullWindowParameterSpace(),
            Dtw::discreteParameterSpace,
            instances -> CachedDdtw.fullWindowParameterSpace(),
            CachedDdtw::discreteParameterSpace,
            instances -> Wdtw.discreteParameterSpace(),
            instances -> CachedWddtw.discreteParameterSpace(),
            Lcss::discreteParameterSpace,
            Erp::discreteParameterSpace,
            instances -> Msm.discreteParameterSpace(),
            instances -> Twe.discreteParameterSpace()
            ));
    }

    public static List<Function<Instances, ParameterSpace>> getDefaultParameterSpaceGetters() {
        return new ArrayList<>(Arrays.asList(
            Dtw::allDiscreteParameterSpace,
            CachedDdtw::allDiscreteParameterSpace,
            instances -> Wdtw.discreteParameterSpace(),
            instances -> CachedWddtw.discreteParameterSpace(),
            Lcss::discreteParameterSpace,
            Erp::discreteParameterSpace,
            instances -> Msm.discreteParameterSpace(),
            instances -> Twe.discreteParameterSpace()
                                            ));
    }

    public ElasticEnsemble() {
        this(getDefaultParameterSpaceGetters());
    }

    public ElasticEnsemble(Function<Instances, ParameterSpace>... parameterSpaceGetters) {
        this(Arrays.asList(parameterSpaceGetters));
    }

    private final List<Function<Instances, ParameterSpace>> parameterSpaceGetters = new ArrayList<>();
    private final List<ParameterSpace> parameterSpaces = new ArrayList<>();

    public ElasticEnsemble(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        setParameterSpaceGetters(parameterSpaceGetters);
    }

    public void setParameterSpaceGetters(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        this.parameterSpaceGetters.clear();
        this.parameterSpaceGetters.addAll(parameterSpaceGetters);
    }

    public static List<ParameterSpace> getParameterSpaces(Instances instances, List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        List<ParameterSpace> parameterSpaces = new ArrayList<>();
        for(Function<Instances, ParameterSpace> getter : parameterSpaceGetters) {
            ParameterSpace parameterSpace = getter.apply(instances);
            parameterSpaces.add(parameterSpace);
        }
        return parameterSpaces;
    }

    private boolean removeDuplicateParameterValues = true;
    private EnsembleModule[] modules = null;
    private ModuleWeightingScheme weightingScheme = new TrainAcc();
    private ModuleVotingScheme votingScheme = new MajorityVoteByConfidence();
    private long phaseTime = 0;
    private final List<Knn> knns = new ArrayList<>();
    private final List<Iterator<String[]>> parameterSetIterators = new ArrayList<>();
    private Selector<Candidate> selector = new BestPerTypeSelector<>(candidate -> candidate.getKnn()
                                                                                           .getDistanceMeasure()
                                                                                           .toString(), (candidate, other) -> {
        int comparison = Integer.compare(candidate.getKnn().getNeighbourhoodSize(), other.getKnn().getNeighbourhoodSize());
        if(comparison <= 0) {
            comparison = Comparator.comparingDouble(ClassifierResults::getAcc).compare(candidate.getTrainResults(), other.getTrainResults());
        }
        return comparison;
    });
    private final List<Knn> finishedKnns = new ArrayList<>();

    public int getNeighbourhoodSize() {
        return neighbourhoodSize;
    }

    public void setNeighbourhoodSize(final int neighbourhoodSize) {
        if(this.neighbourhoodSize < neighbourhoodSize) {
            knns.addAll(finishedKnns);
            finishedKnns.clear();
        }
        this.neighbourhoodSize = neighbourhoodSize;
    }

    private int numParameterSets = -1;
    private int parameterSetCount = 0;
    private int neighbourhoodSize = -1;

    private void setupNeighbourhoodSize(Instances trainInstances) {
        if(neighbourhoodSizePercentage >= 0) {
            setNeighbourhoodSize((int) (neighbourhoodSizePercentage * trainInstances.size()));
        }
    }

    private void setupNumParameterSets() {
        if(numParameterSetsPercentage >= 0) {
            int size = 0;
            for(ParameterSpace parameterSpace : parameterSpaces) {
                size += parameterSpace.size();
            }
            numParameterSets = (int) (numParameterSetsPercentage * size);
        }
    }

    public double getNumParameterSetsPercentage() {
        return numParameterSetsPercentage;
    }

    public void setNumParameterSetsPercentage(final double numParameterSetsPercentage) {
        this.numParameterSetsPercentage = numParameterSetsPercentage;
    }

    private double numParameterSetsPercentage = -1;

    public double getNeighbourhoodSizePercentage() {
        return neighbourhoodSizePercentage;
    }

    public void setNeighbourhoodSizePercentage(final double neighbourhoodSizePercentage) {
        this.neighbourhoodSizePercentage = neighbourhoodSizePercentage;
    }

    private double neighbourhoodSizePercentage = -1;
    private final List<Candidate> constituents = new ArrayList<>();

    private boolean limitedNumParameterSets() {
        return numParameterSets >= 0;
    }

    private boolean withinNumParameterSets() {
        return parameterSetCount < numParameterSets;
    }

    private boolean remainingParameterSets() {
        return !parameterSetIterators.isEmpty() && (!limitedNumParameterSets() || withinNumParameterSets());
    }

    @Override
    public void buildClassifier(final Instances trainInstances) throws
                                                      Exception {
        long startTime = System.nanoTime();
        Random random = getTrainRandom();
        if(trainSetChanged(trainInstances)) {
            knns.clear();
            selector.setRandom(random);
            parameterSetIterators.clear();
            parameterSpaces.clear();
            parameterSetCount = 0;
            finishedKnns.clear();
            parameterSpaces.addAll(getParameterSpaces(trainInstances, parameterSpaceGetters));
            if(removeDuplicateParameterValues) {
                for(ParameterSpace parameterSpace : parameterSpaces) {
                    parameterSpace.removeDuplicateValues();
                }
            }
            for(ParameterSpace parameterSpace : parameterSpaces) {
                Iterator<String[]> iterator = new ParameterSetIterator(parameterSpace, new RandomIndexIterator(random, parameterSpace.size()));
                if(iterator.hasNext()) parameterSetIterators.add(iterator);
            }
            setupNeighbourhoodSize(trainInstances);
            setupNumParameterSets();
            incrementTrainTimeNanos(System.nanoTime() - startTime);
        }
        boolean remainingParameters = remainingParameterSets();
        boolean remainingKnns = !knns.isEmpty();
//        int count = 0;
        if(getNeighbourhoodSize() != 0) {
            while((remainingParameters || remainingKnns) && remainingTrainContractNanos() > phaseTime) {
//                System.out.println(count++);
                long startPhaseTime = System.nanoTime();
                Knn knn;
                boolean choice = true;
                if(remainingParameters && remainingKnns) {
                    choice = random.nextBoolean();
                } else if(remainingKnns) {
                    choice = false;
                }
                int knnIndex;
                if(choice) {
                    int index = random.nextInt(parameterSetIterators.size());
                    Iterator<String[]> iterator = parameterSetIterators.get(index);
                    String[] parameters = iterator.next();
                    if(!iterator.hasNext()) {
                        parameterSetIterators.remove(index);
                    } // todo random guess if no params or constituents
                    knn = new Knn();
                    knn.setOptions(parameters);
                    knn.setNeighbourhoodSize(1);
                    knn.setEarlyAbandon(true);
                    knns.add(knn);
                    knnIndex = knns.size() - 1;
                    System.out.println(parameterSetCount);
                    parameterSetCount++;
                } else {
                    knnIndex = random.nextInt(knns.size());
                    knn = knns.get(knnIndex);
                    int sampleSize = knn.getNeighbourhoodSize() + 1;
                    knn.setNeighbourhoodSize(sampleSize);
                }
                if((knn.getNeighbourhoodSize() + 1 > getNeighbourhoodSize() && getNeighbourhoodSize() >= 0) || knn.getNeighbourhoodSize() + 1 > trainInstances.size()) {
                    finishedKnns.add(knns.remove(knnIndex));
                }
                knn.setTrainContractNanos(remainingTrainContractNanos());
                knn.buildClassifier(trainInstances);
                Candidate candidate = new Candidate(knn.copy(), knn.getTrainResults());
                selector.add(candidate);
                phaseTime = Long.max(System.nanoTime() - startPhaseTime, phaseTime);
                remainingParameters = remainingParameterSets();
                remainingKnns = !knns.isEmpty();
                incrementTrainTimeNanos(System.nanoTime() - startTime);
            }
        }
        constituents.clear();
        constituents.addAll(selector.getSelected());
        modules = new EnsembleModule[constituents.size()];
        for(int i = 0; i < constituents.size(); i++) {
            Knn knn = constituents.get(i).getKnn();
            modules[i] = new EnsembleModule(knn.toString(), knn, knn.getParameters());
            modules[i].trainResults = knn.getTrainResults();
        }
        weightingScheme.defineWeightings(modules, trainInstances.numClasses());
        votingScheme.trainVotingScheme(modules, trainInstances.numClasses());
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for(int i = 0; i < trainInstances.size(); i++) {
            long predictionTime = System.nanoTime();
            double[] distribution = votingScheme.distributionForTrainInstance(modules, i);
            predictionTime = System.nanoTime() - predictionTime;
            for (EnsembleModule module : modules) {
                predictionTime += module.trainResults.getPredictionTimeInNanos(i);
            }
            trainResults.addPrediction(trainInstances.get(i).classValue(), distribution, Utilities.argMax(distribution, getTrainRandom()), predictionTime, null);
        }
        incrementTrainTimeNanos(System.nanoTime() - startTime);
        setClassifierResultsMetaInfo(trainResults);
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                     Exception {
        return votingScheme.distributionForInstance(modules, testInstance);
    }

    private class Candidate {
        private final Knn knn;
        private final ClassifierResults trainResults;

        private Candidate(final Knn knn, final ClassifierResults trainResults) {
            this.knn = knn;
            this.trainResults = trainResults;
        }

        public Knn getKnn() {
            return knn;
        }

        public ClassifierResults getTrainResults() {
            return trainResults;
        }

    }
}
