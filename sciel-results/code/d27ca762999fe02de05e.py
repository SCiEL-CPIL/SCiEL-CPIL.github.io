"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: Multi-Start Differential Evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0
        rng = np.random.default_rng(42)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Archive to store all non-dominated points found so far
        # This is crucial for hypervolume: we keep the "best" of everything we've seen
        archive_X = []
        archive_F = []
        
        used = 0

        def add_to_archive(x, f):
            nonlocal used
            # Check if f is dominated by any existing point in archive
            # Or if it dominates any existing point
            new_archive_X = []
            new_archive_F = []
            dominated_any = False
            is_dominated = False
            
            # Optimization: If archive is small, check all. If large, check all is still O(N) which is fine for 200 budget
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if np.all(af <= f) and np.any(af < f):
                    is_dominated = True
                    break
                elif np.all(f <= af) and np.any(f < af):
                    dominated_any = True
                    # Don't add ax, af to new list
                else:
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            if not is_dominated:
                new_archive_X.append(x)
                new_archive_F.append(f)
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            return not is_dominated

        # 1. Initialization: Spread out initial population
        # Use a mix of random and boundary points
        init_size = 20
        X_init = np.random.uniform(lo, hi, (init_size, dim))
        # Add some corners
        for i in range(min(8, init_size)):
             X_init[i] = np.where(rng.random(dim) < 0.5, lo, hi)
        
        F_init = []
        for x in X_init:
            f = problem(x)
            used += 1
            F_init.append(f)
            add_to_archive(x, f)
        
        if used >= budget:
             return np.array(archive_X)

        # 2. Differential Evolution Loop
        # We will generate individuals one by one to strictly respect the budget
        # and allow dynamic adjustment if needed, though DE usually works in batches.
        # Given the small budget (200), we can afford to check time frequently.
        
        # Standard DE parameters
        F_factor = 0.8  # Mutation scaling factor
        CR = 0.9        # Crossover rate
        
        # We will maintain a "current" population of size P=10 for the DE step
        # But the archive is the true memory.
        # Actually, for small budgets, a simpler approach is often better:
        # Generate a candidate, if it's non-dominated, add to archive.
        # Use the archive to pick parents for DE.
        
        # Let's stick to a population-based DE but with a global archive.
        pop_size = 10
        # Select initial DE population from archive if possible, or keep init
        if len(archive_X) >= pop_size:
            # Pick diverse points from archive for the working population
            # Simple strategy: pick random non-dominated points
            indices = rng.choice(len(archive_X), size=pop_size, replace=False)
            X_pop = np.array([archive_X[i] for i in indices])
            F_pop = np.array([archive_F[i] for i in indices])
        else:
            X_pop = X_init[:pop_size]
            F_pop = F_init[:pop_size]

        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate one offspring at a time to maximize flexibility
            # 1. Select 3 distinct parents from current population
            if len(X_pop) < 3:
                 # If pop is too small, just do random restarts or local search
                 # But we initialized with 10, so this shouldn't happen unless logic fails
                 break
                 
            i1, i2, i3 = rng.choice(len(X_pop), size=3, replace=False)
            
            # 2. Mutation: v = x_i1 + F * (x_i2 - x_i3)
            v = X_pop[i1] + F_factor * (X_pop[i2] - X_pop[i3])
            
            # 3. Crossover: Create trial vector u
            u = np.empty_like(v)
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    u[j] = v[j]
                else:
                    u[j] = X_pop[i1][j]
            
            # 4. Clip to bounds
            u = np.clip(u, lo, hi)
            
            # 5. Evaluate
            if used >= budget:
                break
            f_u = problem(u)
            used += 1
            
            # 6. Selection: Update archive and population
            # Check if u dominates any in population or is non-dominated
            # For the global archive, we just add if non-dominated
            added = add_to_archive(u, f_u)
            
            # For the local DE population, we replace the worst or dominated one
            # Simple strategy: If u dominates x_i1, replace x_i1.
            # Else, if u is non-dominated, maybe replace the point with worst crowding distance?
            # Or just replace x_i1 if it's better in some sense?
            # Let's use a simple tournament or replacement:
            # If u dominates x_i1, replace.
            # Else, if x_i1 dominates u, discard u from pop (but it might be in archive).
            # Else, keep both? But pop size is fixed.
            # Let's replace x_i1 if u is "better" in terms of distance to origin (normalized) 
            # or just randomly if non-dominated to maintain diversity.
            
            # Better: If u dominates any member of X_pop, replace the dominated one.
            replaced = False
            for k in range(len(X_pop)):
                if np.all(f_u <= F_pop[k]) and np.any(f_u < F_pop[k]):
                    X_pop[k] = u
                    F_pop[k] = f_u
                    replaced = True
                    break
            
            if not replaced:
                # If no domination, check if u is non-dominated w.r.t population
                # If so, replace the member with the largest "regret" or just random?
                # Let's replace x_i1 if u is non-dominated w.r.t pop, to explore new area.
                # Actually, standard DE replaces x_i1 if u is better (dominates) or equal.
                # For MOEA, we often keep a larger archive.
                # Let's just replace x_i1 if u is non-dominated w.r.t the population.
                is_dom_by_pop = False
                for k in range(len(X_pop)):
                    if np.all(F_pop[k] <= f_u) and np.any(F_pop[k] < f_u):
                        is_dom_by_pop = True
                        break
                
                if not is_dom_by_pop:
                    # u is non-dominated w.r.t pop. Replace x_i1 to inject new point.
                    X_pop[i1] = u
                    F_pop[i1] = f_u

        # Final polish: If we have budget left, do local search on the best archive points
        if used < budget and time.time() < deadline:
            # Sort archive by sum of normalized objectives (simple scalarization)
            # Reference point for normalization: max in archive or bounds?
            # Use max from archive
            if archive_F:
                F_arr = np.array(archive_F)
                max_F = F_arr.max(axis=0)
                min_F = F_arr.min(axis=0)
                # Avoid division by zero
                denom = max_F - min_F
                denom[denom < 1e-10] = 1.0
                
                # Calculate scalar score
                scores = np.sum((F_arr - min_F) / denom, axis=1)
                # Get top K points from archive for local search
                K = min(5, len(archive_X))
                top_indices = np.argsort(scores)[:K]
                
                for idx in top_indices:
                    if used >= budget or time.time() > deadline:
                        break
                    x_best = archive_X[idx]
                    f_best = archive_F[idx]
                    
                    # Local search: try small steps
                    for _ in range(5):
                        if used >= budget or time.time() > deadline:
                            break
                        # Random direction
                        direction = rng.standard_normal(dim)
                        direction /= np.linalg.norm(direction)
                        # Step size: adaptive? Start small.
                        step_size = 0.01 * span
                        x_cand = x_best + direction * step_size
                        x_cand = np.clip(x_cand, lo, hi)
                        
                        f_cand = problem(x_cand)
                        used += 1
                        
                        # If candidate dominates best, update
                        if np.all(f_cand <= f_best) and np.any(f_cand < f_best):
                            x_best = x_cand
                            f_best = f_cand
                            # Add to archive
                            add_to_archive(x_best, f_best)
                            # Update score for this point? Not strictly needed if we just track best
                        else:
                            # Just add to archive if non-dominated
                            add_to_archive(x_cand, f_cand)

        return np.array(archive_X)
# EVOLVE-BLOCK-END
