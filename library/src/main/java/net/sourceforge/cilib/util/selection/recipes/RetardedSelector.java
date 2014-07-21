/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.util.selection.recipes;

import java.util.List;
import net.sourceforge.cilib.util.selection.PartialSelection;
import net.sourceforge.cilib.util.selection.Samples;
import net.sourceforge.cilib.util.selection.Selection;

/**
 * A recipe for Roulette wheel selection.
 * <p>
 * Roulette wheel selection is performed by:
 * <ol>
 *   <li>Weighing the elements of a selection.</li>
 *   <li>Performing a proportional ordering of the weighed elements.</li>
 *   <li>Returning the best result.</li>
 * </ol>
 * @param <E> The selection type.
 */
public class RetardedSelector<E extends Comparable> implements Selector<E> {

    private static final long serialVersionUID = 4194450350205390514L;
    private int popSize;

    /**
     * Create a new instance.
     */
    public RetardedSelector() {
        popSize = 30;
    }

    /**
     * Create a copy of the provided instance.
     * @param copy The instance to copy.
     */
    public RetardedSelector(RetardedSelector<E> copy) {
        this.popSize = copy.popSize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public PartialSelection<E> on(Iterable<E> iterable) {
        List<E> oldPop = Selection.copyOf(iterable).select(Samples.first(popSize));
        List<E> newPop;
        //if (iterable.size() > popSize)
        newPop = Selection.copyOf(iterable).exclude(oldPop).select(Samples.all());
        if (newPop.size() > popSize-1)
            newPop = (new RandomSelector()).on(newPop).select(Samples.first(popSize-1));

        List<E> best = (new ElitistSelector()).on(oldPop).select(Samples.first());
        List<E> oldPop2 = Selection.copyOf(oldPop).exclude(best).select(Samples.all());

        int popDifference = oldPop2.size() - newPop.size();
        newPop.addAll((new RandomSelector()).on(oldPop2).select(Samples.first(popDifference)));
        newPop.addAll(best);

        return Selection.copyOf(newPop);
    }

    public void setPopSize(int popSize) {
        this.popSize = popSize;
    }
}
