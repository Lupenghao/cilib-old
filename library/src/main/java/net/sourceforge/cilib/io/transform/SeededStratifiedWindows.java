/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.io.transform;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import net.sourceforge.cilib.io.DataTable;
import net.sourceforge.cilib.io.StandardPatternDataTable;
import net.sourceforge.cilib.io.exception.CIlibIOException;
import net.sourceforge.cilib.io.pattern.StandardPattern;
import net.sourceforge.cilib.math.random.generator.MersenneTwister;
import net.sourceforge.cilib.math.random.generator.Rand;
import net.sourceforge.cilib.type.types.Real;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 * A data operator that efficiently shuffles a datatable.
 */
public class SeededStratifiedWindows implements DataOperator {

    private MersenneTwister random;
    private int numOfWindows;
    //private SeededShuffleOperator shuffler;

    public SeededStratifiedWindows() {
        random = new MersenneTwister(Rand.nextLong());
        //shuffler = new SeededShuffleOperator();
        //shuffler.setRandom(random);
        numOfWindows = 1;
    }

    public SeededStratifiedWindows(SeededStratifiedWindows copy) {
        random = copy.random;
        numOfWindows = copy.numOfWindows;
    }

    public SeededStratifiedWindows getClone() {
        return new SeededStratifiedWindows(this);
    }

    /**
     * Modern version of Fisher-Yates shuffle algorithm based on the Richard Durstenfeld
     * implementation as published in:
     * Durstenfeld, Richard (July 1964). "Algorithm 235: Random permutation". Communications of the ACM 7 (7): 420. doi:10.1145/364520.364540.
     * The shuffle in-place (i.e. it doesn't not use additional memory).
     * @param dataTable the table to shuffle.
     * @return the same table as given with patterns in a uniform random order.
     * @throws CIlibIOException an IO Exception that might occur.
     */
    @Override
    public StandardPatternDataTable operate(DataTable badDataTable) throws CIlibIOException {
        StandardPatternDataTable table = (StandardPatternDataTable) badDataTable;

        int baseWindowSize = table.size()/numOfWindows;
        int nrOfLargeWindows = table.size() % numOfWindows;

        //divide patterns into classes
        Map<Type, StandardPatternDataTable> classes = new HashMap<Type, StandardPatternDataTable>();
        for (StandardPattern curPattern : table) {
            //Construct bitstring of classes
            if (curPattern.getTarget() instanceof Real) {
                Real target = (Real) curPattern.getTarget();
                if (target.doubleValue() < 0.5)
                    target.setValue(0.1);
                else
                    target.setValue(0.9);
            }
            else {
                Vector target = (Vector) curPattern.getTarget();
                for (int curElement = 0; curElement < target.size(); curElement++) {
                    if (target.doubleValueOf(curElement) < 0.5)
                        target.setReal(curElement, 0.1);
                    else
                        target.setReal(curElement, 0.9);
                }
            }

            //add current pattern to it's class' table
            StandardPatternDataTable selectedClass = classes.get(curPattern.getTarget());
            if (selectedClass == null) {
                selectedClass = new StandardPatternDataTable();
                classes.put(curPattern.getTarget(), selectedClass);
            }
            selectedClass.addRow(curPattern);
        }

        //TODO: remove this sanity check
        if (classes.size() > 50)
            throw new IllegalStateException("Sanity check: data set is producing too many classes");

        //create empty windows
        ArrayList<StandardPatternDataTable> windows = new ArrayList<StandardPatternDataTable>();
        for (int curWindow = 0; curWindow < numOfWindows; curWindow++) {
            windows.add(new StandardPatternDataTable());
        }

        //divide each class amongst windows
        int windowCounter = 0;
        for (StandardPatternDataTable curClass : classes.values()) {
            //int proportion = (int) Math.floor(((double)curClass.size() / (double)table.size()) * curClass.size());

            while (curClass.size() > 0) {
                int selected = random.nextInt(curClass.size());
                windows.get(windowCounter % windows.size()).addRow(curClass.getRow(selected));
                curClass.removeRow(selected);
                windowCounter++;
            }
        }

        //TODO: remove sanity check
        for (int curWindow = 0; curWindow < numOfWindows; curWindow++) {
            int expectedSize = baseWindowSize + (curWindow < nrOfLargeWindows?1:0);
            if (windows.get(curWindow).size() != expectedSize)
                throw new IllegalStateException("Sanity check: window does not contain the correct amount of patterns");
        }

        //concatenate all the windows into one data set
        StandardPatternDataTable newTable = new StandardPatternDataTable();
        for (StandardPatternDataTable curWindow : windows) {
            for (StandardPattern curPattern : curWindow) {
                newTable.addRow(curPattern);
            }
        }

        /*for (Type curKey : classes.keySet()) {
            if (classes.get(curKey).size() == 0) {
                classes.remove(curKey);
            }
        }*/

        /*while (classes.size() > 0) {
            for (StandardPatternDataTable curWindow : windows) {
                int eSize = (windows.indexOf(curWindow) < nrOfLargeWindows) ? 1 : 0;
                int needed = Math.min(baseWindowSize + eSize - curWindow.size(), classes.size());

                int selection[] = new int[needed];
                for (int curCount = 0; curCount < needed; curCount++) {
                    int selected = random.nextInt(needed - curCount);

                    for (int i = 0; i < curCount; i++){
                        if (selected == selection[i]){
                            selected++;
                            i = 0;
                        }
                    }
                }

                for (int curCount = 0; curCount < needed; curCount++) {
                    StandardPatternDataTable curClass = classes.values().toArray(new StandardPatternDataTable[0])[selection[curCount]];
                    int s = random.nextInt(curClass.size());
                    curWindow.addRow(curClass.getRow(s));
                    curClass.removeRow(s);
                }

                for (Type curKey : classes.keySet()) {
                    if (classes.get(curKey).size() == 0) {
                        classes.remove(curKey);
                    }
                }
            }
        }*/

        /*for (int curWindow = 0; curWindow < numOfWindows; ++curWindow) {
            if (table.getRow(0).getTarget() instanceof Real) {
                //count class
                int classCount = 0;
                for (StandardPattern curPattern: table) {
                    if (((Real) curPattern.getTarget()).doubleValue() > 0.5)
                        classCount++;
                }

                int selected = random.nextInt(classCount)+1;
                int count = 0;
                for (int n = size - 1 -startingPatterns; n >= 0; n--) {
                    if (((Real) table.getRow(n).getTarget()).doubleValue() > 0.5)
                        count++;

                    if (count == selected) {
                        StandardPattern tmp = table.getRow(n);
                        table.setRow(n, table.getRow(size-1-startingPatterns));
                        table.setRow(size-1-startingPatterns, tmp);

                        startingPatterns++;

                        break;
                    }
                }

                selected = Rand.nextInt(size-classCount) +1;
                count = 0;
                for (int n = size - 1 -startingPatterns; n >= 0; n--) {
                    if (((Real) table.getRow(n).getTarget()).doubleValue() < 0.5)
                        count++;

                    if (count == selected) {
                        StandardPattern tmp = table.getRow(n);
                        table.setRow(n, table.getRow(size-1-startingPatterns));
                        table.setRow(size-1-startingPatterns, tmp);

                        startingPatterns++;

                        break;
                    }
                }
            }
        }
        
        int size = dataTable.size();
        for (int n = size - 1; n > 1; n--) {
            int k = random.nextInt(n + 1);
            Object tmp = dataTable.getRow(k);
            dataTable.setRow(k, dataTable.getRow(n));
            dataTable.setRow(n, tmp);
        }*/

        //TODO: fix this
        return newTable;
    }

    public void setSeed(long seed) {
        random = new MersenneTwister(seed);
        //shuffler.setRandom(random);
    }

    public void setNumOfWindows(int numOfWindows) {
        this.numOfWindows = numOfWindows;
    }
}
