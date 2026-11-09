import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Differential Evolution (DE) with an 
    external archive for Pareto front approximation.
    
    Strategy:
    1. Initialize a small population using Latin Hypercube Sampling (LHS) 
       for good initial spread.
    2. Iterate using DE mutation (DE/rand/1/bin) and crossover.
    3. Evaluate offspring.
    4. Update an external archive with non-dominated points.
    5. The population is maintained by selecting the best non-dominated 
       points from the union of the current population and the archive, 
       ensuring diversity and progress towards the front.
    6. Budget management: Strictly count evaluations to stay <= 200.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo

        # State
        eval_count = 0
        archive = []  # List of (x, f) tuples for non-dominated solutions
        # Population will be a subset of the archive or newly generated points
        # We maintain a population of size P to drive DE
        # Given budget 200, and dim 5, a population of ~15-20 is reasonable for DE.
        # Let's use P = 16.
        # Initial cost: 16 evals.
        # Each iteration: P evals for offspring.
        # 16 (init) + 184 (remaining) / 16 ~ 11 iterations.
        # This is a bit low for DE convergence, but with a good archive and 
        # re-seeding from the archive, it can work.
        # Alternatively, smaller population P=10?
        # 10 init + 190/10 = 19 iterations. Better.
        pop_size = 10
        
        # Helper to check non-dominance
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 in all objectives and < in at least one
            if np.all(f1 <= f2) and np.any(f1 < f2):
                return True
            return False

        def is_non_dominated(f, archive_f):
            # Check if f is non-dominated with respect to archive_f (list of arrays)
            for af in archive_f:
                if dominates(af, f):
                    return False
            return True

        def update_archive(x, f):
            nonlocal archive
            # If dominated by existing, discard
            for ax, af in archive:
                if dominates(af, f):
                    return
            
            # Remove existing points dominated by new point
            new_archive = []
            for ax, af in archive:
                if not dominates(f, af):
                    new_archive.append((ax, af))
            
            new_archive.append((x, f))
            archive = new_archive

        # 1. Initialization: Latin Hypercube Sampling
        # Generate pop_size points
        population_x = np.zeros((pop_size, dim))
        population_f = np.zeros((pop_size, n_obj))
        
        for d in range(dim):
            indices = np.arange(pop_size)
            offsets = np.random.random(pop_size)
            u = (indices + offsets) / pop_size
            perm = np.random.permutation(pop_size)
            population_x[:, d] = u[perm]
            population_x[:, d] = lo[d] + population_x[:, d] * range_vec[d]

        for i in range(pop_size):
            if eval_count >= budget:
                break
            f = np.array(problem(population_x[i]))
            population_f[i] = f
            eval_count += 1
            update_archive(population_x[i], f)

        # Sort initial archive by hypervolume contribution or just keep as is
        # For DE, we need a population. We will use the current population_x/f.
        # However, to improve diversity, we can periodically refresh population from archive.
        
        # DE Parameters
        F = 0.8  # Mutation factor
        CR = 0.9 # Crossover rate

        # Main Loop
        while eval_count < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            # We will generate pop_size offspring
            # Check if we have enough budget for a full generation
            if eval_count + pop_size > budget:
                # Generate only as many as we can afford
                num_to_gen = budget - eval_count
            else:
                num_to_gen = pop_size
            
            if num_to_gen <= 0:
                break

            # DE Mutation: DE/rand/1/bin
            # For each individual i in population
            # We need 3 distinct random indices r1, r2, r3 != i
            # To simplify and ensure we use the current population, 
            # we pick random indices from the current population array.
            
            new_x_list = []
            new_f_list = []

            for i in range(num_to_gen):
                if eval_count >= budget:
                    break
                
                # Select r1, r2, r3 distinct from i and each other
                # Since pop_size is small (10), we can just pick random indices
                # and retry if collision, or just pick any 3 distinct from range(pop_size)
                # If pop_size < 4, DE/rand/1 doesn't work well. But pop_size=10 is fine.
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                
                # Mutation
                v = population_x[r1] + F * (population_x[r2] - population_x[r3])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover
                x_trial = population_x[i].copy()
                j_rand = np.random.randint(0, dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        x_trial[j] = v[j]
                    else:
                        x_trial[j] = population_x[i][j]
                
                # Evaluate
                f_trial = np.array(problem(x_trial))
                eval_count += 1
                
                new_x_list.append(x_trial)
                new_f_list.append(f_trial)
                
                # Update archive immediately to keep it fresh
                update_archive(x_trial, f_trial)

            # Selection: Replace population
            # We want to maintain diversity. 
            # Strategy: The new population consists of the best non-dominated points 
            # from the union of (current population + new offspring).
            # However, DE usually does 1-to-1 selection. 
            # For MO, we can do:
            # If offspring dominates parent, replace.
            # Else if parent dominates offspring, keep parent.
            # Else, keep both? But population size is fixed.
            
            # A robust MO-DE selection:
            # 1. Combine current population and offspring.
            # 2. Find non-dominated set.
            # 3. If non-dominated set size > pop_size, use crowding distance or hypervolume to select.
            # 4. If size < pop_size, fill with dominated points closest to front? 
            #    Or just keep the existing ones.
            
            # Simpler approach for small budget:
            # Just replace the population with the offspring if they are better, 
            # or mix. 
            # Given the budget is tight, let's just set the population to the 
            # newly generated offspring, but ensure we don't lose the best points.
            # Actually, a common trick is to keep the population as the 
            # non-dominated points from the *archive* if available, 
            # or the current population if not.
            
            # Let's try a simple elitism:
            # For each i, if f_trial[i] dominates population_f[i], replace.
            # Otherwise, if population_f[i] dominates f_trial[i], keep.
            # Otherwise, keep the one with better hypervolume contribution? 
            # Too expensive. Let's keep the parent if not dominated by child.
            # This might stagnate.
            
            # Better: Use the Archive to re-seed the population if it stagnates.
            # But for now, let's just update population where child is better or equal.
            # To encourage exploration, if neither dominates, keep the parent.
            
            for i in range(num_to_gen):
                # Only compare if i is within the range of the original population
                if i < pop_size:
                    if dominates(new_f_list[i], population_f[i]):
                        population_x[i] = new_x_list[i]
                        population_f[i] = new_f_list[i]
                    # else keep parent
        
            # Optional: If we have archive points that are not in population, 
            # we could inject them to maintain diversity.
            # But with pop_size=10 and budget 200, we have ~18 generations.
            # The archive will grow. 
            # Let's periodically (every 5 generations) reset population to 
            # the best non-dominated points in the archive (up to pop_size).
            
            # We need a generation counter
            if not hasattr(self, '_gen_count'):
                self._gen_count = 0
            self._gen_count += 1
            
            if self._gen_count % 5 == 0 and len(archive) >= pop_size:
                # Select top pop_size from archive based on hypervolume contribution
                # Approximation: Just take random non-dominated points from archive?
                # Or sort by sum of objectives?
                # Let's sort by sum of normalized objectives (rough knee).
                # Or just pick pop_size random points from archive to re-exploration.
                # Random re-seeding from archive helps escape local traps.
                indices = np.random.choice(len(archive), size=pop_size, replace=False)
                for i in range(pop_size):
                    ax, af = archive[indices[i]]
                    population_x[i] = ax
                    population_f[i] = af

        return None
