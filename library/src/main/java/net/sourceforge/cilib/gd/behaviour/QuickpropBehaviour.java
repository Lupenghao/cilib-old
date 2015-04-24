/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.gd.behaviour;

import net.sourceforge.cilib.algorithm.AbstractAlgorithm;
import net.sourceforge.cilib.entity.behaviour.AbstractBehaviour;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.entity.Property;
import net.sourceforge.cilib.problem.DifferentiableProblem;
import net.sourceforge.cilib.problem.objective.Maximise;
import net.sourceforge.cilib.type.types.Real;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 * Behaviour representing normal particle behaviour. The behaviour is:
 * 1-update velocity
 * 2-update position
 * 3-enforce boundary constraints
 * 4-calculate fitness
 */
public class QuickpropBehaviour extends AbstractBehaviour {

    private double stepSize;
    private double maxStepSize;
    private double weightDecay;
    private double sign;

    /**
     * Default constructor assigns standard position and velocity provider.
     */
    public QuickpropBehaviour() {
        maxStepSize = 1.75;
        weightDecay = 0.0001;
    }

    /**
     * Copy Constructor.
     *
     * @param copy The {@link StandardParticleBehaviour} object to copy.
     */
    public QuickpropBehaviour(QuickpropBehaviour copy) {
        super(copy);

        this.maxStepSize = copy.maxStepSize;
        this.weightDecay = copy.weightDecay;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public QuickpropBehaviour getClone() {
        return new QuickpropBehaviour(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Entity performIteration(Entity entity) {

        Vector oldPosition = (Vector) entity.get(Property.PREVIOUS_SOLUTION);
        Vector curPosition = (Vector) entity.getPosition();
        Vector oldVelocity = (Vector) entity.get(Property.VELOCITY);
        Vector oldGradient = (Vector) entity.get(Property.PREVIOUS_GRADIENT);
        Vector newGradient = ((DifferentiableProblem) AbstractAlgorithm.get().getOptimisationProblem()).getGradient(curPosition);

        //update velocity
        Vector newVelocity;
        if (oldGradient == null) {
            newVelocity = newGradient.multiply(sign);
        }
        else {
            newVelocity = newGradient.divide(
                              oldGradient.subtract(
                                  newGradient
                              )
                          ).multiply(
                              oldVelocity
                          );

            for (int curElement = 0; curElement < newVelocity.size(); curElement++) {
                if (Math.abs(newVelocity.doubleValueOf(curElement)) > maxStepSize * Math.abs(oldVelocity.doubleValueOf(curElement))
                        || newGradient.doubleValueOf(curElement) * newVelocity.doubleValueOf(curElement) * sign < 0) {
                    newVelocity.setReal(curElement, Math.abs(maxStepSize*oldVelocity.doubleValueOf(curElement))
                                        *Math.signum(newGradient.doubleValueOf(curElement))*sign);
                }
            }

            newVelocity = newVelocity.subtract(oldPosition.multiply(weightDecay));
        }
        entity.put(Property.VELOCITY, newVelocity);

        //update position
        Vector newPosition = curPosition.plus(newVelocity.multiply(stepSize));
        entity.put(Property.PREVIOUS_SOLUTION, entity.getPosition());
        entity.put(Property.CANDIDATE_SOLUTION, newPosition);

        boundaryConstraint.enforce(entity);
        
        entity.updateFitness(fitnessCalculator.getFitness(entity));
        
        return entity;
    }

    public void setStepSize(double stepSize) {
        this.stepSize = stepSize;
    }

    public void setMaxStepSize(double maxStepSize) {
        this.maxStepSize = maxStepSize;
    }
    
    public void setWeightDecay(double weightDecay) {
        this.weightDecay = weightDecay;
    }

    public void setSign(double sign) {
        this.sign = sign;
    }
}
