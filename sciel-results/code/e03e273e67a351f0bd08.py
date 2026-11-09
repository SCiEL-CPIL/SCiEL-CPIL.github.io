"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 with hypervolume-aware search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s/330s limits

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        budget = self.budget
        n_obj = self.n_obj
        dim = self.dim
        ref = np.array([3.0, 5.0, 7.0])

        # Track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            fx = problem(x)
            all_X.append(x.copy())
            all_F.append(np.array(fx))
            used += 1
            return fx

        # Phase 1: Initial random sampling
        n_init = min(budget, 40)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        for x in X_init:
            if time.time() > deadline or used >= budget:
                break
            eval_point(x)

        # Helper: check if a point is dominated by any in a set
        def is_dominated(f, F_set):
            for g in F_set:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False

        # Helper: get non-dominated indices
        def get_nd_indices(F_arr):
            n = len(F_arr)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        # Phase 2: Main optimization loop with adaptive local search
        # Use a population-based approach with hypervolume-aware selection
        pop_size = 15
        F_arr = np.array(all_F)
        X_arr = np.array(all_X)
        
        # Initialize population from non-dominated points
        nd_idx = get_nd_indices(F_arr)
        if len(nd_idx) == 0:
            nd_idx = list(range(min(pop_size, len(F_arr))))
        
        # Select up to pop_size non-dominated points for population
        pop_idx = nd_idx[:pop_size]
        while len(pop_idx) < pop_size and len(pop_idx) < len(F_arr):
            for i in range(len(F_arr)):
                if i not in pop_idx:
                    pop_idx.append(i)
                    break
        
        pop_X = X_arr[pop_idx].copy()
        pop_F = F_arr[pop_idx].copy()

        # Adaptive step size
        base_step = 0.1 * (hi - lo)
        min_step = 0.01 * (hi - lo)
        step_scale = base_step

        # Main loop
        while used < budget and time.time() < deadline:
            # Select parent using hypervolume contribution heuristic
            # Points that contribute more to hypervolume are more valuable
            # Approximate by distance to reference point (closer = better)
            nd_in_pop = get_nd_indices(pop_F)
            if len(nd_in_pop) == 0:
                nd_in_pop = list(range(len(pop_F)))
            
            # Weight by proximity to reference point (smaller distance = higher weight)
            dists = np.sum((pop_F[nd_in_pop] - ref) ** 2, axis=1)
            weights = 1.0 / (dists + 1e-6)
            weights /= weights.sum()
            
            parent_idx = nd_in_pop[rng.choice(len(nd_in_pop), p=weights)]
            parent = pop_X[parent_idx].copy()
            parent_f = pop_F[parent_idx].copy()

            # Mutation: adaptive step size with random direction
            # Occasionally use larger jumps for exploration
            if rng.random() < 0.1:
                scale = base_step * (2.0 + rng.random())
            else:
                scale = step_scale * (0.5 + rng.random())
            
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction) + 1e-10
            child = parent + scale * direction
            child = np.clip(child, lo, hi)

            fc = eval_point(child)
            if fc is None:
                break

            # Update population: replace worst point if child is non-dominated
            child_nd = not is_dominated(fc, pop_F)
            if child_nd:
                # Find the worst point in population to replace
                # Use normalized distance to reference point
                F_norm = (pop_F - np.min(pop_F, axis=0)) / (np.max(pop_F, axis=0) - np.min(pop_F, axis=0) + 1e-10)
                worst_idx = int(np.argmax(np.sum(F_norm, axis=1)))
                
                # Replace if child is better than worst
                if np.any(fc < pop_F[worst_idx]):
                    pop_X[worst_idx] = child
                    pop_F[worst_idx] = fc
                    
                    # Adapt step size: shrink if improvement was small
                    improvement = np.sum(np.maximum(0, pop_F[worst_idx] - fc))
                    if improvement < 0.01:
                        step_scale = max(min_step, step_scale * 0.9)
                    else:
                        step_scale = min(base_step, step_scale * 1.05)
            else:
                # Child is dominated; slightly shrink step
                step_scale = max(min_step, step_scale * 0.98)

            # Periodic exploration: add random point every 30 iterations
            if (used - n_init) % 30 == 0 and used < budget and time.time() < deadline:
                exp = lo + (hi - lo) * rng.random(dim)
                fe = eval_point(exp)
                if fe is not None:
                    # Add to population if non-dominated
                    if not is_dominated(fe, pop_F):
                        F_norm = (pop_F - np.min(pop_F, axis=0)) / (np.max(pop_F, axis=0) - np.min(pop_F, axis=0) + 1e-10)
                        worst_idx = int(np.argmax(np.sum(F_norm, axis=1)))
                        if np.any(fe < pop_F[worst_idx]):
                            pop_X[worst_idx] = exp
                            pop_F[worst_idx] = fe

        # Phase 3: Final local refinement of best non-dominated points
        F_all = np.array(all_F)
        X_all = np.array(all_X)
        
        final_nd = get_nd_indices(F_all)
        if len(final_nd) > 0 and used < budget and time.time() < deadline:
            # Sort by distance to reference point
            nd_points = F_all[final_nd]
            dists = np.sum((nd_points - ref) ** 2, axis=1)
            sorted_idx = np.argsort(dists)
            
            # Refine top non-dominated points
            n_to_refine = min(len(final_nd), max(1, (budget - used) // 8))
            
            for i in range(n_to_refine):
                if used >= budget or time.time() > deadline:
                    break
                
                orig_idx = final_nd[sorted_idx[i]]
                base_x = X_all[orig_idx].copy()
                base_f = F_all[orig_idx].copy()
                
                # Local search with decreasing step sizes
                local_steps = [0.03, 0.015, 0.005] * (hi - lo)
                
                for step in local_steps:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Try several perturbations at this step size
                    n_tries = min(3, budget - used)
                    for _ in range(n_tries):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        direction = rng.standard_normal(dim)
                        direction /= np.linalg.norm(direction) + 1e-10
                        candidate = base_x + step * direction
                        candidate = np.clip(candidate, lo, hi)
                        
                        fc = eval_point(candidate)
                        if fc is None:
                            break
                        
                        # Update base if candidate is better and non-dominated
                        if np.any(fc < base_f) and not is_dominated(fc, [base_f]):
                            base_x = candidate
                            base_f = fc
                            X_all[orig_idx] = base_x
                            F_all[orig_idx] = base_f

        return X_all

# EVOLVE-BLOCK-END
