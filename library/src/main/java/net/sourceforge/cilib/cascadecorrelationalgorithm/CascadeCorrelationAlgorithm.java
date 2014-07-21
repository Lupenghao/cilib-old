/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.cascadecorrelationalgorithm;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import net.sourceforge.cilib.algorithm.AbstractAlgorithm;
import net.sourceforge.cilib.algorithm.initialisation.ClonedPopulationInitialisationStrategy;
import net.sourceforge.cilib.algorithm.population.SinglePopulationBasedAlgorithm;
import net.sourceforge.cilib.controlparameter.ControlParameter;
import net.sourceforge.cilib.controlparameter.ConstantControlParameter;
import net.sourceforge.cilib.ec.Individual;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.entity.Property;
import net.sourceforge.cilib.entity.comparator.AscendingFitnessComparator;
import net.sourceforge.cilib.entity.initialisation.CovarianceInitialisationStrategy;
import net.sourceforge.cilib.math.random.generator.Rand;
import net.sourceforge.cilib.nn.architecture.builder.CascadeArchitectureBuilder;
import net.sourceforge.cilib.nn.architecture.builder.LayerConfiguration;
import net.sourceforge.cilib.nn.architecture.visitors.CascadeVisitor;
import net.sourceforge.cilib.nn.components.Neuron;
import net.sourceforge.cilib.nn.NeuralNetwork;
import net.sourceforge.cilib.problem.nn.CascadeHiddenNeuronCorrelationProblem;
import net.sourceforge.cilib.problem.nn.CascadeOutputLayerTrainingProblem;
import net.sourceforge.cilib.problem.nn.NNTrainingProblem;
import net.sourceforge.cilib.problem.Problem;
import net.sourceforge.cilib.problem.solution.Fitness;
import net.sourceforge.cilib.problem.solution.InferiorFitness;
import net.sourceforge.cilib.problem.solution.OptimisationSolution;
import net.sourceforge.cilib.type.types.container.Vector;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.type.types.Real;

public class CascadeCorrelationAlgorithm extends AbstractAlgorithm {

    private AbstractAlgorithm phase1Algorithm;
    private AbstractAlgorithm phase2Algorithm;
    private CascadeHiddenNeuronCorrelationProblem phase1Problem;
    private CascadeOutputLayerTrainingProblem phase2Problem;
    private Fitness trackedFitness;
    private Neuron neuronPrototype;
    private Vector covarianceVector;
	private ArrayList<Entity> oldPopulation = null;
    private ControlParameter survivalRatio;

    public CascadeCorrelationAlgorithm() {
        neuronPrototype = new Neuron();
        trackedFitness = InferiorFitness.instance();
        phase1Problem = new CascadeHiddenNeuronCorrelationProblem();
        phase2Problem = new CascadeOutputLayerTrainingProblem();
        survivalRatio = ConstantControlParameter.of(0.5);
        oldPopulation = new ArrayList<>();
    }

    public CascadeCorrelationAlgorithm(CascadeCorrelationAlgorithm rhs) {
        neuronPrototype = rhs.neuronPrototype.getClone();
        trackedFitness = rhs.trackedFitness.getClone();
        phase1Algorithm = rhs.phase1Algorithm.getClone();
        phase2Algorithm = rhs.phase2Algorithm.getClone();
        phase1Problem = rhs.phase1Problem.getClone();
        phase2Problem = rhs.phase2Problem.getClone();
        survivalRatio = rhs.survivalRatio.getClone();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public CascadeCorrelationAlgorithm getClone() {
        return new CascadeCorrelationAlgorithm(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void algorithmInitialisation() {
        NNTrainingProblem problem = (NNTrainingProblem) optimisationProblem;
        problem.initialise();
        
        NeuralNetwork network = problem.getNeuralNetwork();
        
        phase1Problem.setNeuron(neuronPrototype);
        phase1Problem.setTrainingSet(problem.getTrainingSet());
        phase1Problem.setValidationSet(problem.getValidationSet());
        phase1Problem.setGeneralisationSet(problem.getGeneralisationSet());
        phase1Problem.setNeuralNetwork(network);
        
        phase2Problem.setTrainingSet(problem.getTrainingSet());
        phase2Problem.setValidationSet(problem.getValidationSet());
        phase2Problem.setGeneralisationSet(problem.getGeneralisationSet());
        phase2Problem.setNeuralNetwork(network);
        phase2Problem.initialise();

        covarianceVector = Vector.newBuilder().copyOf(phase2Problem.getDomain().getBuiltRepresentation()).build();
        for (int curElement = 0; curElement < covarianceVector.size(); curElement++) {
            covarianceVector.setReal(curElement, Double.NaN);
        }
    }

    /**
     * Performs a phase 1 followed by a phase 2. Phase 1 is omitted during
     * the first iteration.
     */
    @Override
    protected void algorithmIteration() {
        if (getIterations() > 0) {
            phase1();
        }

        phase2();
    }

    /**
     * Perform the correlation phase.
     * A clone is made of the phase 1 algorithm to ensure a clean start.
     * Once the algorithm stops, the solution is used to add a neuron in
     * a new layer just before the output layer. If the algorithm produces
     * multiple solutions, one new neuron is added for each solution.
     * Regardless of the number of solutions, only one layer is added.
     */
    @VisibleForTesting
    protected void phase1() {
        NNTrainingProblem problem = (NNTrainingProblem) optimisationProblem;
        NeuralNetwork network = problem.getNeuralNetwork();
        Vector trackedWeights = network.getWeights();
        
        AbstractAlgorithm alg1 = (AbstractAlgorithm) phase1Algorithm.getClone();

        phase1Problem.initialise();

        alg1.setOptimisationProblem(phase1Problem);
        alg1.performInitialisation();
        alg1.runAlgorithm();

        //List<OptimisationSolution> solutions = Lists.<OptimisationSolution>newLinkedList(alg1.getSolutions());
        List<OptimisationSolution> solutions = Arrays.asList(alg1.getBestSolution());

        List<LayerConfiguration> layers = network.getArchitecture().getArchitectureBuilder().getLayerConfigurations();

        Vector covariance = phase1Problem.calculateCovarianceVector((Vector) solutions.get(0).getPosition());

        //expand weight vector
        int consolidatedLayerSize = 0;
        int insertionIndex = 0;
        for (int curLayer = 0; curLayer < layers.size()-1; ++curLayer) {
            insertionIndex += consolidatedLayerSize*layers.get(curLayer).getSize();
            consolidatedLayerSize += layers.get(curLayer).getSize();
            if (layers.get(curLayer).isBias())
                consolidatedLayerSize++;
        }

        for (OptimisationSolution curSolution : solutions) {
            for (Numeric curElement : ((Vector) curSolution.getPosition())) {
                trackedWeights.insert(insertionIndex++, curElement);
            }
        }

        insertionIndex += consolidatedLayerSize;
        for (int curOutput = 0; curOutput < layers.get(layers.size()-1).getSize(); ++curOutput) {
            for (int curSolution = 0; curSolution < solutions.size(); ++curSolution) {
                trackedWeights.insert(insertionIndex, Real.valueOf(Double.NaN));
            }

            covarianceVector.insert((curOutput+1)*(consolidatedLayerSize+1)-1, covariance.get(curOutput));
            for (Entity curEntity : oldPopulation) {
                ((Vector) curEntity.getPosition()).insert((curOutput+1)*(consolidatedLayerSize+1)-1, Real.valueOf(Double.NaN));
            }

            insertionIndex += solutions.size() + consolidatedLayerSize;
        }

        //initialise oldpopulation
        for (Entity curEntity : oldPopulation) {
            (new CovarianceInitialisationStrategy(covarianceVector,false)).initialise(Property.CANDIDATE_SOLUTION, curEntity);
        }
        
        //expand neural network
        LayerConfiguration targetLayerConfiguration = new LayerConfiguration(solutions.size(), neuronPrototype.getActivationFunction(), false);
        network.getArchitecture().getArchitectureBuilder().addLayer(layers.size()-1, targetLayerConfiguration);
        network.initialise();
        network.setWeights(trackedWeights);
    }

    /**
     * Performs the output-training phase.
     * A clone is made of the phase 2 algorithm to ensure a clean start.
     * Once the algorithm is finished the best solution is used.
     */
    @VisibleForTesting
    protected void phase2() {
        NNTrainingProblem problem = (NNTrainingProblem) optimisationProblem;
        NeuralNetwork network = problem.getNeuralNetwork();
        Vector trackedWeights = network.getWeights();
        
        phase2Problem.initialise();

        SinglePopulationBasedAlgorithm alg2 = (SinglePopulationBasedAlgorithm) phase2Algorithm.getClone();
        alg2.setOptimisationProblem(phase2Problem);
        ((CovarianceInitialisationStrategy) ((Individual) ((ClonedPopulationInitialisationStrategy) 
            alg2.getInitialisationStrategy())
            .getEntityType()).getInitialisationStrategy()).setMask(covarianceVector);
        alg2.performInitialisation();

        if (oldPopulation.size() > 0) {
		    ArrayList<Entity> newPopulation = Lists.newArrayList(alg2.getTopology());
		    for (int curCount = 0; curCount < oldPopulation.size(); curCount++) {
                int index = (int) Math.floor(Rand.nextDouble()*newPopulation.size());
                newPopulation.remove(index);
            }
            for (Entity curEntity : oldPopulation) {
                newPopulation.add(curEntity);
            }
            alg2.setTopology(fj.data.Java.<Entity>ArrayList_List().f(newPopulation));
        }
        
        alg2.runAlgorithm();

        OptimisationSolution solution = alg2.getBestSolution();
        trackedFitness = solution.getFitness();
        Vector newWeights = (Vector) solution.getPosition();

        for (int curElement = 0; curElement < newWeights.size(); ++curElement) {
            trackedWeights.set(trackedWeights.size()-1-curElement,
                             newWeights.get(newWeights.size()-1-curElement));
        }

        oldPopulation.clear();
        java.util.List<Entity> tmp = Lists.newArrayList(alg2.getTopology());
        Collections.sort(tmp, new AscendingFitnessComparator<Entity>());
        for (int curCount = 0; curCount < tmp.size()*survivalRatio.getParameter(); curCount++) {
            oldPopulation.add(tmp.get(tmp.size() - 1 - curCount).getClone());
        }
        
        network.setWeights(trackedWeights);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public OptimisationSolution getBestSolution() {
        NNTrainingProblem problem = (NNTrainingProblem) optimisationProblem;
        NeuralNetwork network = problem.getNeuralNetwork();
        Vector trackedWeights = network.getWeights();
        return new OptimisationSolution(trackedWeights, trackedFitness);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Iterable<OptimisationSolution> getSolutions() {
        ArrayList<OptimisationSolution> solutions = new ArrayList<OptimisationSolution>();
        solutions.add(getBestSolution());
        return solutions;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setOptimisationProblem(Problem problem) {
        Preconditions.checkArgument(problem instanceof NNTrainingProblem,
                                    "CascadeCorrelationAlgorithm can only be used with NNTrainingProblem.");
        Preconditions.checkArgument(((NNTrainingProblem) problem).getNeuralNetwork().getArchitecture().getArchitectureBuilder()
                                    instanceof CascadeArchitectureBuilder,
                                    "Cascade architecture is needed.");
        Preconditions.checkArgument(((NNTrainingProblem) problem).getNeuralNetwork().getOperationVisitor()
                                    instanceof CascadeVisitor,
                                    "CascadeVisitor is needed.");
        optimisationProblem = problem;
    }

    /**
     * Set the algorithm that should be used during the correlation phase.
     * @param algorithm The optimisation algorithm to be used.
     */
    public void setPhase1Algorithm(AbstractAlgorithm algorithm) {
        this.phase1Algorithm = algorithm;
    }

    /**
     * Set the algorithm that should be used during the output-training phase.
     * @param algorithm The optimisation algorithm to be used.
     */
    public void setPhase2Algorithm(AbstractAlgorithm algorithm) {
        this.phase2Algorithm = algorithm;
    }

    /**
     * Gets the number of evaluations performed during the correlation
     * phase.
     * @return The number of evaluations.
     */
    public int getPhase1EvaluationCount() {
        return phase1Problem.getFitnessEvaluations();
    }

    /**
     * Gets the number of evaluations performed during the output-training
     * phase.
     * @return The number of evaluations.
     */
    public int getPhase2EvaluationCount() {
        return phase2Problem.getFitnessEvaluations();
    }

    /**
     * Gets the number of weight evaluations performed. This only includes
     * weight evaluations performed while evaluating new candidate hidden
     * neurons.
     * @return The number of weight evaluations.
     */
    public int getPhase1WeightEvaluationCount() {
        return phase1Problem.getWeightEvaluationCount();
    }

    /**
     * Gets the number of weight evaluations performed. This only includes
     * weight evaluations performed in the output layer.
     * @return The number of weight evaluations.
     */
    public int getPhase2WeightEvaluationCount() {
        return phase2Problem.getWeightEvaluationCount();
    }

    public void setSurvivalRatio(ControlParameter survivalRatio) {
        this.survivalRatio = survivalRatio;
    }
}
