/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.problem.boundaryconstraint;

import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.entity.Property;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.type.types.Types;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 * Once the entity has over shot the search space boundaries, re-initialise
 * the Entity once again to be within the search space of the problem at a
 * random position.
 *
 * @see Types#isInsideBounds(net.sourceforge.cilib.type.types.Type)
 */
public class PreviousPosition implements BoundaryConstraint {

    private double bound = 1.0e290;

    /**
     * {@inheritDoc}
     */
    @Override
    public PreviousPosition getClone() {
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void enforce(Entity entity) {
        
        double total = 0;
        for (Numeric curElement : (Vector) entity.getPosition()) {
            total += Math.abs(curElement.doubleValue() * 2);
        }

        if (total > bound || Double.isNaN(total) || total == Double.POSITIVE_INFINITY) {
            entity.setPosition(entity.get(Property.PREVIOUS_SOLUTION));
        }
    }
    
    public void setBound(double bound) {
        this.bound = bound;
    }
}
