/**
 * Computational Intelligence Library (CIlib)
 * Copyright (C) 2003 - 2010
 * Computational Intelligence Research Group (CIRG@UP)
 * Department of Computer Science
 * University of Pretoria
 * South Africa
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 */
package net.sourceforge.cilib.pso.crossover;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import net.sourceforge.cilib.entity.Entity;
import net.sourceforge.cilib.entity.EntityType;
import net.sourceforge.cilib.entity.Particle;
import net.sourceforge.cilib.entity.operators.crossover.CrossoverStrategy;
import net.sourceforge.cilib.entity.operators.crossover.DiscreteCrossoverStrategy;
import net.sourceforge.cilib.entity.operators.crossover.OnePointCrossoverStrategy;
import net.sourceforge.cilib.pso.crossover.pbestupdate.CurrentPositionOffspringPBestProvider;
import net.sourceforge.cilib.pso.crossover.pbestupdate.OffspringPBestProvider;
import net.sourceforge.cilib.util.selection.recipes.ElitistSelector;

public class DiscreteVelocityCrossoverStrategy implements CrossoverStrategy {

    private DiscreteCrossoverStrategy crossoverStrategy;
    private OffspringPBestProvider pbestProvider;

    public DiscreteVelocityCrossoverStrategy() {
        this.crossoverStrategy = new OnePointCrossoverStrategy();
        this.pbestProvider = new CurrentPositionOffspringPBestProvider();
    }

    public DiscreteVelocityCrossoverStrategy(DiscreteVelocityCrossoverStrategy copy) {
        this.crossoverStrategy = copy.crossoverStrategy.getClone();
        this.pbestProvider = copy.pbestProvider;
    }

    public CrossoverStrategy getClone() {
        return new DiscreteVelocityCrossoverStrategy(this);
    }

    public <E extends Entity> List<E> crossover(List<E> parentCollection) {
        List<Particle> parents = (List<Particle>) parentCollection;
        List<Particle> offspring = crossoverStrategy.crossover(parents);
        List<Particle> offspringVelocity = new ArrayList<Particle>();
        Particle nBest = new ElitistSelector<Particle>().on(parents).select();

        for (Particle p : parents) {
            Particle v = p.getClone();
            v.setCandidateSolution(v.getVelocity());
            offspringVelocity.add(v);
        }

        offspringVelocity = crossoverStrategy.crossover(offspringVelocity, crossoverStrategy.getCrossoverPoints());

        Iterator<Particle> vIter = offspringVelocity.iterator();
        for (Particle p : offspring) {
            p.getProperties().put(EntityType.Particle.BEST_POSITION, pbestProvider.f(parents, p));

            Particle pbCalc = p.getClone();
            pbCalc.setNeighbourhoodBest(nBest);
            pbCalc.setCandidateSolution(p.getBestPosition());
            pbCalc.calculateFitness();

            p.getProperties().put(EntityType.Particle.BEST_FITNESS, pbCalc.getFitness());
            p.getProperties().put(EntityType.Particle.VELOCITY, vIter.next().getCandidateSolution());

            p.setNeighbourhoodBest(nBest);
            p.calculateFitness();
        }

        return (List<E>) offspring;
    }

    public int getNumberOfParents() {
        return crossoverStrategy.getNumberOfParents();
    }

    public void setCrossoverStrategy(DiscreteCrossoverStrategy crossoverStrategy) {
        this.crossoverStrategy = crossoverStrategy;
    }

    public DiscreteCrossoverStrategy getCrossoverStrategy() {
        return crossoverStrategy;
    }

}
