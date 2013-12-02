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
public class GradientDescentBehaviour extends AbstractBehaviour {

    private double stepSize;
    private double momentum;

    /**
     * Default constructor assigns standard position and velocity provider.
     */
    public GradientDescentBehaviour() {
        stepSize = 0.9;
        momentum = 0.5;
    }

    /**
     * Copy Constructor.
     *
     * @param copy The {@link StandardParticleBehaviour} object to copy.
     */
    public GradientDescentBehaviour(GradientDescentBehaviour copy) {
        super(copy);

        this.stepSize = copy.stepSize;
        this.momentum = copy.momentum;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public GradientDescentBehaviour getClone() {
        return new GradientDescentBehaviour(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Entity performIteration(Entity entity) {

        double sign = (AbstractAlgorithm.get().getOptimisationProblem() instanceof Maximise) ? 1 : -1;

        Vector curPosition = (Vector) entity.getPosition();
        Vector oldVelocity = (Vector) entity.get(Property.VELOCITY);
        Vector newGradient = ((DifferentiableProblem) AbstractAlgorithm.get().getOptimisationProblem()).getGradient(curPosition);

        //update velocity
        Vector newVelocity = newGradient.multiply(sign*stepSize);
        entity.put(Property.VELOCITY, newVelocity);

        //update position
        Vector newPosition = curPosition.plus(newVelocity);
        if (oldVelocity != null) {
            newPosition = newPosition.plus(oldVelocity.multiply(momentum));
        }
        entity.put(Property.PREVIOUS_SOLUTION, entity.getPosition());
        entity.put(Property.CANDIDATE_SOLUTION, newPosition);

        boundaryConstraint.enforce(entity);
        
        entity.updateFitness(fitnessCalculator.getFitness(entity));
        
        return entity;
    }

    public void setStepSize(double stepSize) {
        this.stepSize = stepSize;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }
}
