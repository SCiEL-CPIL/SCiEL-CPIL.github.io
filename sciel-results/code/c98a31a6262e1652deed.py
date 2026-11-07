"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Differential Evolution with Hypervolume Contribution Selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # Reference point for hypervolume calculation
        # Based on task description: [3000.0, 0.06]
        ref_point = np.array([3000.0, 0.06])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = np.array(problem(x), dtype=np.float64)
            except Exception:
                return None
            if not np.all(np.isfinite(f)):
                f = np.array([1e10, 1e10])
            all_X.append(x)
            all_F.append(f)
            used += 1
            return f

        # 1. Initialization
        # Use a population size that allows for several generations.
        # Budget 200. Let's use population size 20.
        # 200 / 20 = 10 generations.
        pop_size = 20
        
        # Initialize population with random points
        # Ensure we don't exceed budget
        init_count = min(pop_size, budget)
        pop_X = []
        pop_F = []
        
        for _ in range(init_count):
            x = lo + span * np.random.random(dim)
            f = eval_point(x)
            if f is None:
                break
            pop_X.append(x)
            pop_F.append(f)

        if len(pop_X) == 0:
            return None

        pop_X = np.array(pop_X)
        pop_F = np.array(pop_F)

        # Helper: Calculate Hypervolume Contribution of a point given a set of non-dominated points
        # For 2D, the contribution of a point p to the hypervolume of set S (with reference r)
        # is the area of the box defined by p and r that is not covered by any other point in S.
        # Specifically, for 2D, if we sort non-dominated points by f1 ascending, 
        # the contribution of point i is (f_{i+1,1} - f_{i,1}) * (r_2 - f_{i,2}) 
        # with f_{n,1} = r_1.
        
        def calculate_hv_contributions(F_nd, ref):
            """
            F_nd: (N, 2) array of non-dominated points.
            ref: (2,) array of reference point.
            Returns: (N,) array of hypervolume contributions.
            """
            if len(F_nd) == 0:
                return np.array([])
            
            # Sort by first objective ascending
            idx = np.argsort(F_nd[:, 0])
            F_sorted = F_nd[idx]
            
            contributions = np.zeros(len(F_sorted))
            
            # The "next" first objective value for the last point is the reference
            f1_next = np.empty(len(F_sorted))
            f1_next[:-1] = F_sorted[1:, 0]
            f1_next[-1] = ref[0]
            
            # Width in f1 direction
            widths = f1_next - F_sorted[:, 0]
            # Height in f2 direction
            heights = ref[1] - F_sorted[:, 1]
            
            # Area
            contributions = widths * heights
            
            # Map back to original order
            inv_idx = np.empty(len(idx), dtype=int)
            inv_idx[idx] = np.arange(len(idx))
            contrib_original = np.zeros(len(F_nd))
            contrib_original[inv_idx] = contributions
            
            return contrib_original

        def get_non_dominated_indices(F):
            if len(F) == 0:
                return np.array([], dtype=int)
            n = len(F)
            le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
            lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
            dominated = np.any(le & lt, axis=0)
            return np.where(~dominated)[0]

        # 2. Main DE Loop
        # Adaptive mutation factor F
        # Standard DE uses F in [0.5, 1.0]. We can shrink it over time.
        
        while used < budget and time.time() < deadline:
            # Get current non-dominated set from population
            nd_idx = get_non_dominated_indices(pop_F)
            
            if len(nd_idx) == 0:
                # Fallback if something went wrong
                nd_idx = np.arange(len(pop_F))
            
            # Calculate contributions for selection
            F_nd = pop_F[nd_idx]
            contribs = calculate_hv_contributions(F_nd, ref_point)
            
            # Select parents based on contribution
            # Points with higher contribution are more valuable to the front
            # We want to explore around them.
            
            # Normalize contributions for probability
            total_contrib = np.sum(contribs)
            if total_contrib < 1e-10:
                # All contributions are zero? Use uniform
                probs = np.ones(len(nd_idx)) / len(nd_idx)
            else:
                probs = contribs / total_contrib
            
            # Select two distinct parents for mutation
            # We need at least 3 points for DE (v = x_r1 + F*(x_r2 - x_r3))
            # If we have fewer, we just pick random distinct indices.
            
            n_gen = (budget - used) // pop_size
            # Shrink mutation factor
            # Start with 0.8, end with 0.2
            progress = 1.0 - (budget - used) / max(1, (budget - init_count))
            F_mut = 0.8 - 0.6 * progress
            
            # Generate offspring
            new_X = []
            new_F = []
            
            for i in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random parents from non-dominated set
                # If nd_idx has < 3, sample with replacement or from full pop
                if len(nd_idx) >= 3:
                    # Sample 3 distinct indices from nd_idx
                    r_idx = np.random.choice(len(nd_idx), size=3, replace=False)
                    r1, r2, r3 = nd_idx[r_idx[0]], nd_idx[r_idx[1]], nd_idx[r_idx[2]]
                else:
                    # Sample from full population with replacement
                    r_idx = np.random.choice(len(pop_F), size=3, replace=True)
                    r1, r2, r3 = r_idx[0], r_idx[1], r_idx[2]
                    
                # Ensure r1, r2, r3 are distinct if possible
                # If len(nd_idx) >= 3, they are distinct.
                # If not, we might have duplicates. That's okay for a fallback.
                
                v = pop_X[r1] + F_mut * (pop_X[r2] - pop_X[r3])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Evaluate
                f_v = eval_point(v)
                if f_v is None:
                    continue
                
                new_X.append(v)
                new_F.append(f_v)
                
                # Selection: Replace x_i if v dominates x_i
                # Or if v is non-dominated and has better HV contribution?
                # Standard DE: Replace if v dominates x_i.
                if np.all(f_v <= pop_F[i]) and np.any(f_v < pop_F[i]):
                    pop_X[i] = v
                    pop_F[i] = f_v
            # End of generation
        # End of loop

        return None

# EVOLVE-BLOCK-END
