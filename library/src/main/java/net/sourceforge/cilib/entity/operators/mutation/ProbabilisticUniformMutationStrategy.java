/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.entity.operators.mutation;

import java.util.List;
import java.util.ListIterator;
import net.sourceforge.cilib.controlparameter.ControlParameter;
import net.sourceforge.cilib.controlparameter.ConstantControlParameter;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.math.random.generator.Rand;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 */
public class ProbabilisticUniformMutationStrategy extends MutationStrategy {

    private static final long serialVersionUID = -3951730432882403768L;
    private ControlParameter temparature;

    public ProbabilisticUniformMutationStrategy() {
        super();
        temparature = ConstantControlParameter.of(0.5);        
    }

    public ProbabilisticUniformMutationStrategy(ProbabilisticUniformMutationStrategy copy) {
        super(copy);
        temparature = copy.temparature.getClone();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ProbabilisticUniformMutationStrategy getClone() {
        return new ProbabilisticUniformMutationStrategy(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <E extends Entity> List<E> mutate(List<E> entity) {
        for (ListIterator<? extends Entity> individual = entity.listIterator(); individual.hasNext();) {
            Entity current = individual.next();
            Vector chromosome = (Vector) current.getPosition();

            for (int i = 0; i < chromosome.size(); i++) {
                if (this.getMutationProbability().getParameter() >= this.getRandomDistribution().getRandomNumber()) {
                    double value = this.getOperatorStrategy().evaluate(chromosome.doubleValueOf(i), this.getRandomDistribution().getRandomNumber(-temparature.getParameter(), temparature.getParameter()));
                    chromosome.setReal(i, value);
                }
            }
        }
        return entity;
    }

    public void setTemparature(ControlParameter temparature) {
        this.temparature = temparature;
    }
}
