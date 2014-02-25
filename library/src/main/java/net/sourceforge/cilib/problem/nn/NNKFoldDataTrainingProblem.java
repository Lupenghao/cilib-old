/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.problem.nn;

import com.google.common.collect.Lists;
import com.google.common.annotations.VisibleForTesting;
import java.util.ArrayList;
import net.sourceforge.cilib.algorithm.AbstractAlgorithm;
import net.sourceforge.cilib.io.DataTable;
import net.sourceforge.cilib.io.DataTableBuilder;
import net.sourceforge.cilib.io.DelimitedTextFileReader;
import net.sourceforge.cilib.io.StandardPatternDataTable;
import net.sourceforge.cilib.io.exception.CIlibIOException;
import net.sourceforge.cilib.io.pattern.StandardPattern;
import net.sourceforge.cilib.io.transform.DataOperator;
import net.sourceforge.cilib.io.transform.ShuffleOperator;
import net.sourceforge.cilib.io.transform.TypeConversionOperator;
import net.sourceforge.cilib.nn.architecture.visitors.OutputErrorVisitor;
import net.sourceforge.cilib.nn.domain.*;
import net.sourceforge.cilib.problem.AbstractProblem;
import net.sourceforge.cilib.problem.solution.Fitness;
import net.sourceforge.cilib.type.DomainRegistry;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 * Class represents a {@link NNTrainingProblem} where the goal is to optimize
 * the set of weights of a neural network to best fit a given static dataset (either
 * regression, classification etc.).
 */
public class NNKFoldDataTrainingProblem extends NNTrainingProblem {

    private DataOperator initialShuffler;
    private DataTableBuilder dataTableBuilder;
    private SolutionConversionStrategy solutionConversionStrategy;
    private int previousShuffleIteration;
    private boolean initialised;

    private int numOfWindows;
    private int windowOffset;

    /**
     * Default constructor.
     */
    public NNKFoldDataTrainingProblem() {
        super();
        dataTableBuilder = new DataTableBuilder(new DelimitedTextFileReader());
        solutionConversionStrategy = new WeightSolutionConversionStrategy();
        previousShuffleIteration = -1;
        initialised = false;
        initialShuffler = new ShuffleOperator();
    }

    /**
     * Initialises the problem by reading in the data and constructing the training
     * and generalisation sets. Also initialises (constructs) the neural network.
     */
    @Override
    public void initialise() {
        if (initialised) {
            return;
        }
        try {
            dataTableBuilder.addDataOperator(new TypeConversionOperator());
            dataTableBuilder.addDataOperator(patternConversionOperator);
            dataTableBuilder.buildDataTable();
            DataTable dataTable = dataTableBuilder.getDataTable();

            dataTable = initialShuffler.operate(dataTable);

            int windowSize = dataTable.size()/numOfWindows;
            int windowsWithExtra = dataTable.size() % numOfWindows;
            int curPattern = 0;
            ArrayList<DataTable> windows = new ArrayList<DataTable>();
            for (int curWindow = 0; curWindow < numOfWindows; curWindow++) {
                DataTable curTable = new StandardPatternDataTable();
                for (int patternCount = 0; patternCount < windowSize; patternCount++) {
                    curTable.addRow((StandardPattern) dataTable.getRow(curPattern));
                    curPattern++;
                }
                if (curWindow < windowsWithExtra) {
                    curTable.addRow((StandardPattern) dataTable.getRow(curPattern));
                    curPattern++;
                }
                windows.add(curTable);
            }

            int trainingSize = (int) Math.round(numOfWindows * trainingSetPercentage);
            int validationSize = (int) Math.round(numOfWindows * validationSetPercentage);
            int generalisationSize = numOfWindows - trainingSize - validationSize;

            trainingSet = new StandardPatternDataTable();
            validationSet = new StandardPatternDataTable();
            generalisationSet = new StandardPatternDataTable();

            for (int curWindow = 0; curWindow < trainingSize; curWindow++) {
                trainingSet.addRows(Lists.newArrayList(windows.get((curWindow+windowOffset)%numOfWindows)));
            }

            for (int curWindow = trainingSize; curWindow < validationSize + trainingSize; curWindow++) {
                validationSet.addRows(Lists.newArrayList(windows.get((curWindow+windowOffset)%numOfWindows)));
            }

            for (int curWindow = validationSize + trainingSize; curWindow < generalisationSize + validationSize + trainingSize; curWindow++) {
                generalisationSet.addRows(Lists.newArrayList(windows.get((curWindow+windowOffset)%numOfWindows)));
            }

            neuralNetwork.initialise();
            
        } catch (CIlibIOException exception) {
            exception.printStackTrace();
        }
        initialised = true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public AbstractProblem getClone() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * Calculates the fitness of the given solution by setting the neural network
     * weights to the solution and evaluating the training set in order to calculate
     * the MSE (which is minimized).
     *
     * @param solution the weights representing a solution.
     * @return a new MinimisationFitness wrapping the MSE training error.
     */
    @Override
    protected Fitness calculateFitness(Type solution) {
        if (trainingSet == null) {
            this.initialise();
        }

        int currentIteration = AbstractAlgorithm.get().getIterations();
        if (currentIteration != previousShuffleIteration) {
            try {
                shuffler.operate(trainingSet);
            } catch (CIlibIOException exception) {
                exception.printStackTrace();
            }
        }

        neuralNetwork.getArchitecture().accept(solutionConversionStrategy.interpretSolution(solution));

        double errorTraining = 0.0;
        OutputErrorVisitor visitor = new OutputErrorVisitor();
        Vector error = null;
        for (StandardPattern pattern : trainingSet) {
            Vector output = neuralNetwork.evaluatePattern(pattern);
            visitor.setInput(pattern);
            neuralNetwork.getArchitecture().accept(visitor);
            error = visitor.getOutput();
            for (Numeric real : error) {
                errorTraining += real.doubleValue() * real.doubleValue();
            }
        }
        errorTraining /= trainingSet.getNumRows() * error.size();

        return objective.evaluate(errorTraining);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DomainRegistry getDomain() {
        if (!initialised) {
            this.initialise();
        }
        return neuralNetwork.getArchitecture().getDomain();
    }

    public void setInitialShuffler(DataOperator initialShuffler) {
        this.initialShuffler = initialShuffler;
    }

    /**
     * Gets the datatable builder.
     *
     * @return the datatable builder.
     */
    public DataTableBuilder getDataTableBuilder() {
        return dataTableBuilder;
    }

    /**
     * Sets the datatable builder.
     *
     * @param dataTableBuilder the new datatable builder.
     */
    public void setDataTableBuilder(DataTableBuilder dataTableBuilder) {
        this.dataTableBuilder = dataTableBuilder;
    }

    /**
     * Gets the source URL of the the datatable builder.
     *
     * @return the source URL of the the datatable builder.
     */
    public String getSourceURL() {
        return dataTableBuilder.getSourceURL();
    }

    /**
     * Sets the source URL of the the datatable builder.
     *
     * @param sourceURL the new source URL of the the datatable builder.
     */
    public void setSourceURL(String sourceURL) {
        dataTableBuilder.setSourceURL(sourceURL);
    }

    public SolutionConversionStrategy getSolutionConversionStrategy() {
        return solutionConversionStrategy;
    }

    public void setSolutionConversionStrategy(SolutionConversionStrategy solutionConversionStrategy) {
        this.solutionConversionStrategy = solutionConversionStrategy;
    }

    public void setNumOfWindows(int numOfWindows) {
        this.numOfWindows = numOfWindows;
    }

    public void setWindowOffset(int windowOffset) {
        this.windowOffset = windowOffset;
    }
}
