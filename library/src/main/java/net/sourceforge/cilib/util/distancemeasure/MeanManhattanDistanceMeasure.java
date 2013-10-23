/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.util.distancemeasure;

import java.util.Collection;
import net.sourceforge.cilib.type.types.Numeric;

/**
 * Manhattan Distance is a special case of the {@link MinkowskiMetric} with
 * 'alpha' = 1.
 */
public class MeanManhattanDistanceMeasure extends ManhattanDistanceMeasure {

    /**
     * Create an instance of the {@linkplain ManhattanDistanceMeasure}.
     */
    @Override
    public double distance(Collection<? extends Numeric> x, Collection<? extends Numeric> y) {
        double manhattan = super.distance(x, y);
        return manhattan/x.size();
    }
    
}
