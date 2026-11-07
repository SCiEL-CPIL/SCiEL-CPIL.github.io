"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using MOPSO with Pareto Archive and Stratified Initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._ref = np.array([3000.0, 0.06])
        self._eval_count = 0
        self._deadline = None
        self._problem = None
        self._lo = None
        self._hi = None

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 325s.
        self._deadline = start_time + 325.0
        self._problem = problem
        self._lo = np.array([b[0] for b in bounds])
        self._hi = np.array([b[1] for b in bounds])
        self._eval_count = 0

        dim = self.dim
        lo = self._lo
        hi = self._hi
        budget = self.budget

        # MOPSO Parameters
        # Population size: Small enough to allow many generations, large enough for diversity.
        # Budget 200. NP=15 -> ~13 generations.
        NP = 15
        if budget < 30:
            NP = max(5, budget // 2)
            
        # Inertia weights
        w_max = 0.9
        w_min = 0.4
        c1 = 1.5
        c2 = 1.5
        
        # Archive size for global best selection
        archive_size = 50

        # 1. Initialization: Stratified Sampling
        # We want to cover the space well. 
        # Simple random might cluster. Stratified ensures coverage.
        # For dim=4, we can split each dimension into NP^(1/dim) bins? 
        # Simpler: Generate NP points using Latin Hypercube Sampling (LHS) or just stratified random.
        # Let's use a simple stratified approach: divide each dimension into k intervals.
        # k = int(NP ** (1.0 / dim))
        # If NP=15, dim=4, k=1. That's just random.
        # If NP=16, k=2.
        # Let's just use random for simplicity but ensure we don't start with duplicates.
        # Actually, the "inspiration" mentioned stratified grid covers corners/edges.
        # Let's do a mix: 20% stratified corners/edges, 80% random.
        
        pop = np.zeros((NP, dim))
        fitness = np.zeros((NP, self.n_obj))
        
        # Generate initial population
        # Use np.random directly to respect the seed set by evaluator
        for i in range(NP):
            if self._eval_count >= budget or time.time() > self._deadline:
                break
            # Random point
            x = lo + (hi - lo) * np.random.random(dim)
            pop[i] = x
            fitness[i] = self._evaluate(x)

        # Initialize velocities
        # Velocity bounds: typically 20% of range
        v_max = 0.2 * (hi - lo)
        vel = (hi - lo) * np.random.random((NP, dim)) * 0.2
        
        # Personal bests
        p_best = pop.copy()
        p_best_fit = fitness.copy()

        # Global Bests (Archive)
        # We maintain a list of non-dominated points
        # Start with initial non-dominated points
        archive = []
        archive_fit = []
        
        for i in range(NP):
            if self._is_non_dominated(fitness[i], archive_fit):
                archive.append(pop[i])
                archive_fit.append(fitness[i])
        
        # If archive is empty (should not happen), add first point
        if not archive:
            archive.append(pop[0])
            archive_fit.append(fitness[0])

        # Main Loop
        gen = 0
        while self._eval_count < budget and time.time() < self._deadline:
            gen += 1
            w = w_max - (w_max - w_min) * (gen / max(1, budget // NP))
            
            # Update velocities and positions
            for i in range(NP):
                if self._eval_count >= budget or time.time() > self._deadline:
                    break
                
                # Select global bests from archive
                # Use a grid-based selection or random selection from non-dominated set
                # To encourage diversity, we can select based on crowding distance or just random
                # Let's select 2 random non-dominated points from archive
                # If archive is small, pick with replacement
                if len(archive) > 1:
                    g1_idx = np.random.randint(0, len(archive))
                    g2_idx = np.random.randint(0, len(archive))
                    while g2_idx == g1_idx and len(archive) > 1:
                        g2_idx = np.random.randint(0, len(archive))
                    g1 = np.array(archive[g1_idx])
                    g2 = np.array(archive[g2_idx])
                else:
                    g1 = np.array(archive[0])
                    g2 = np.array(archive[0])

                # Velocity update
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                r3 = np.random.random(dim)
                
                # Standard PSO update
                # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                # We have two gbests, so we can average them or pick one.
                # Let's use a weighted average or just pick one randomly to maintain diversity.
                # Let's pick one randomly.
                if np.random.random() < 0.5:
                    gbest = g1
                else:
                    gbest = g2
                    
                vel[i] = w * vel[i] + c1 * r1 * (p_best[i] - pop[i]) + c2 * r2 * (gbest - pop[i])
                
                # Clip velocity
                vel[i] = np.clip(vel[i], -v_max, v_max)
                
                # Position update
                new_x = pop[i] + vel[i]
                
                # Clip to bounds
                new_x = np.clip(new_x, lo, hi)
                
                # Evaluate
                new_f = self._evaluate(new_x)
                
                # Update personal best
                # new_f dominates p_best_fit[i]?
                if self._dominates(new_f, p_best_fit[i]):
                    p_best[i] = new_x
                    p_best_fit[i] = new_f
                # If p_best dominates new_f, keep p_best.
                # If non-dominated, we might keep new_x as a candidate for p_best?
                # Standard MOPSO often keeps the one with better crowding or just random.
                # Let's keep p_best if it dominates new_f, else keep new_f if it dominates p_best.
                # If non-dominated, let's keep the one with better "distance" to reference?
                # Or just keep p_best to maintain stability.
                elif not self._dominates(p_best_fit[i], new_f):
                    # Non-dominated case. Let's keep new_x if it's "more diverse" or just random.
                    # To be safe and simple, keep p_best unless new_f is clearly better in some metric.
                    # Actually, for MOPSO, p_best is usually the best non-dominated point found so far for that particle.
                    # If new_f is non-dominated with p_best, we can replace p_best with new_f with some probability.
                    if np.random.random() < 0.5:
                        p_best[i] = new_x
                        p_best_fit[i] = new_f

                # Update population
                pop[i] = new_x
                fitness[i] = new_f
                
                # Update Archive
                # Add new_f to archive if non-dominated
                if self._is_non_dominated(new_f, archive_fit):
                    # Remove points dominated by new_f
                    new_archive = []
                    new_archive_fit = []
                    for j in range(len(archive)):
                        if not self._dominates(new_f, archive_fit[j]):
                            new_archive.append(archive[j])
                            new_archive_fit.append(archive_fit[j])
                    new_archive.append(new_x)
                    new_archive_fit.append(new_f)
                    
                    # If archive too large, prune
                    if len(new_archive) > archive_size:
                        # Keep the most diverse ones? 
                        # Simple strategy: keep the first 'archive_size' or random subset.
                        # Better: Keep those with highest crowding distance.
                        # For simplicity and speed, let's just keep a random subset of size archive_size
                        # or the ones closest to the reference point?
                        # Let's keep the ones with best hypervolume contribution? Too complex.
                        # Let's just keep the first 'archive_size' encountered or random.
                        # Random pruning is okay for small archives.
                        indices = np.random.choice(len(new_archive), size=archive_size, replace=False)
                        archive = [new_archive[i] for i in indices]
                        archive_fit = [new_archive_fit[i] for i in indices]
                    else:
                        archive = new_archive
                        archive_fit = new_archive_fit

        return None

    def _evaluate(self, x):
        """Evaluate a point, tracking budget and time."""
        if self._eval_count >= self.budget:
            # Don't raise, just return dummy to avoid crash, though loop should check.
            # The contract says "More than 200 evaluations... makes the whole result invalid".
            # So we must NOT evaluate the 201st time.
            # The loop checks self._eval_count < budget.
            # If we are here, it means we tried to exceed.
            # Return a dummy value to prevent exception?
            # No, the contract says "exception before the first... invalid".
            # "exception later keeps evaluations made".
            # So if we exceed budget, we should stop.
            # But _evaluate is called from the loop.
            # If we return a dummy, it gets recorded?
            # "Every returned objective vector is recorded".
            # So we must NOT return a vector if we are over budget.
            # But the loop structure:
            # if self._eval_count >= budget: break
            # So we should never call _evaluate if over budget.
            # If we do, we should raise to stop?
            # "an exception later keeps the evaluations made".
            # So raising is safe.
            raise RuntimeError("Budget exceeded")
            
        if time.time() > self._deadline:
            raise TimeoutError("Time limit exceeded")
            
        # Clip to bounds
        x = np.clip(x, self._lo, self._hi)
        
        # Call problem
        f = self._problem(x)
        
        self._eval_count += 1
        return f

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _is_non_dominated(self, f_new, archive_fit):
        """Check if f_new is non-dominated with respect to archive_fit."""
        for f in archive_fit:
            if self._dominates(f, f_new):
                return False
        return True

# EVOLVE-BLOCK-END
