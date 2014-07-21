/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.entity.initialisation;

import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.entity.Property;
import net.sourceforge.cilib.math.random.generator.Rand;
import net.sourceforge.cilib.type.types.Numeric;
import net.sourceforge.cilib.type.types.Type;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 * Initialises a vector property according to a mask. If a mask element has
 * NaN as its value, the corresponding element in the property vector is
 * randomised. Otherwise, the element is set to the value given by the mask.
 *
 * @param <E> The entity type.
 */
public class CovarianceInitialisationStrategy<E extends Entity> implements
        InitialisationStrategy<E> {
    private Vector mask;
    private boolean doAll = true;

    public CovarianceInitialisationStrategy() {
        this.mask = null;
    }

    public CovarianceInitialisationStrategy(Vector mask) {
        this.mask = mask;
    }

    public CovarianceInitialisationStrategy(Vector mask, boolean doAll) {
        this.mask = mask;
        this.doAll = doAll;
    }

    public CovarianceInitialisationStrategy(CovarianceInitialisationStrategy copy) {
        this.mask = copy.mask.getClone();
        this.doAll = copy.doAll;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public CovarianceInitialisationStrategy getClone() {
        return new CovarianceInitialisationStrategy<E>(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialise(Property key, E entity) {
        Type type = entity.get(key);

        if (type instanceof Vector) {
            Vector vector = (Vector) type;
            //System.err.println(vector.size());

            for (int curElement = 0; curElement < vector.size(); ++curElement) {
                if (doAll || Double.isNaN(vector.doubleValueOf(curElement))) {
                    if (Double.isNaN(mask.doubleValueOf(curElement))){
                        vector.get(curElement).randomise();
                    }
                    else {
                        vector.setReal(curElement, Rand.nextDouble()*mask.doubleValueOf(curElement));
                    }
                }
            }
        }
        else {
            throw new UnsupportedOperationException("Cannot perform initialisation on non Vector type.");
        }
    }

    public void setMask(Vector mask) {
        this.mask = mask;
    }
}
