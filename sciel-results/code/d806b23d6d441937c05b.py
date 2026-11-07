# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 using a hybrid approach:
    1. Direct geometric construction of the known Pareto front (2-simplex).
    2. Local search / evolutionary refinement to ensure coverage and handle
       any potential numerical issues or mapping inaccuracies.
    
    The DTLZ1 problem has a known Pareto front defined by:
    f0 + f1 + f2 = 0.5 * sum(x_i) for i in [3..n-1]
    with f_i >= 0.
    
    For 7 variables (dim=7), the last 4 variables (indices 3,4,5,6) determine
    the sum. The first 3 variables determine the distribution of the objectives
    on the simplex.
    
    Actually, the standard DTLZ1 definition is:
    g(x) = 100 * (n - 1 + sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for i in 2..n-1)
    f0 = 0.5 * x0 * (1 + g(x))
    f1 = 0.5 * x1 * (1 - x0) * (1 + g(x))
    f2 = 0.5 * (1 - x1) * (1 + g(x))
    
    The Pareto front is achieved when g(x) = 0, which requires x_i = 0.5 for i >= 2.
    Then:
    f0 = 0.5 * x0
    f1 = 0.5 * x1 * (1 - x0)
    f2 = 0.5 * (1 - x1)
    with x0, x1 in [0, 1].
    
    Note: f0 + f1 + f2 = 0.5 * x0 + 0.5 * x1 * (1 - x0) + 0.5 * (1 - x1)
    = 0.5 * (x0 + x1 - x0*x1 + 1 - x1)
    = 0.5 * (1 + x0 - x0*x1)
    
    Wait, let me re-check. The standard DTLZ1 Pareto front is a 2-simplex where
    f0 + f1 + f2 = 0.5. This is when g(x)=0.
    
    Actually, for g(x)=0, we need x_i = 0.5 for all i >= 2.
    Then:
    f0 = 0.5 * x0 * (1 + 0) = 0.5 * x0
    f1 = 0.5 * x1 * (1 - x0) * (1 + 0) = 0.5 * x1 * (1 - x0)
    f2 = 0.5 * (1 - x1) * (1 + 0) = 0.5 * (1 - x1)
    
    Sum: f0 + f1 + f2 = 0.5 * x0 + 0.5 * x1 * (1 - x0) + 0.5 * (1 - x1)
    = 0.5 * [x0 + x1 - x0*x1 + 1 - x1]
    = 0.5 * [1 + x0 - x0*x1]
    
    This is NOT constant 0.5 unless x0*x1 = x0, which means x1=1 or x0=0.
    
    Let me look up the correct DTLZ1 definition again.
    
    Standard DTLZ1:
    g(x) = 100 * (m + sum_{i=2}^{n-1} (x_i^2 - 10*cos(20*pi*x_i)))
    where m is the number of Pareto-optimal solutions (for 3 objectives, m=2? No, m is typically n - k + 1 where k is number of objectives).
    
    Actually, for DTLZ1 with 3 objectives:
    f0 = 0.5 * x0 * (1 + g(x))
    f1 = 0.5 * x1 * (1 - x0) * (1 + g(x))
    f2 = 0.5 * (1 - x1) * (1 + g(x))
    
    The Pareto front is when g(x) = 0. The minimum of g(x) is 0, achieved when
    x_i = 0.5 for all i in {2, ..., n-1}.
    
    With g(x) = 0:
    f0 = 0.5 * x0
    f1 = 0.5 * x1 * (1 - x0)
    f2 = 0.5 * (1 - x1)
    
    The sum f0 + f1 + f2 = 0.5 * x0 + 0.5 * x1 - 0.5 * x0 * x1 + 0.5 - 0.5 * x1
    = 0.5 * x0 - 0.5 * x0 * x1 + 0.5
    = 0.5 * (1 + x0 * (1 - x1))
    
    This is not constant. So the Pareto front is NOT a flat simplex in objective space?
    
    Wait, I think I'm confusing DTLZ1 with DTLZ2 or another variant. Let me reconsider.
    
    Actually, upon reflection, the DTLZ1 Pareto front IS a 2-simplex, but it's not
    f0 + f1 + f2 = constant. Let me re-examine.
    
    For DTLZ1, the Pareto front in objective space is defined by:
    f0 + f1 + f2 = 0.5
    with f_i >= 0.
    
    This is achieved when x_i = 0.5 for i >= 2, and x0, x1 are chosen such that:
    0.5 * x0 + 0.5 * x1 * (1 - x0) + 0.5 * (1 - x1) = 0.5
    
    Let me verify: if x0 = 1, x1 = 0: f0 = 0.5, f1 = 0, f2 = 0.5. Sum = 1.0. Not 0.5.
    
    I think my understanding is wrong. Let me just use a robust search approach
    that doesn't rely on the exact front geometry, but instead uses a strong
    multi-objective evolutionary algorithm with good diversity maintenance.
    
    Given the budget of 200 evaluations and 7 dimensions, I'll use:
    1. A population-based approach with crowding distance for diversity.
    2. Tournament selection based on non-domination.
    3. Gaussian perturbation for mutation.
    4. Initial population spread across the search space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Population size
        pop_size = min(40, self.budget // 5)
        if pop_size < 10:
            pop_size = 10
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_X = X.copy()
        all_F = F.copy()
        
        def non_dominated_sort(F):
            """Return non-dominated indices."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]
        
        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for k in range(self.n_obj):
                vals = F[indices, k]
                idx_sorted = np.argsort(vals)
                cd[idx_sorted[0]] = np.inf
                cd[idx_sorted[-1]] = np.inf
                val_range = vals[idx_sorted[-1]] - vals[idx_sorted[0]]
                if val_range > 1e-12:
                    for i in range(1, n - 1):
                        cd[idx_sorted[i]] += (vals[idx_sorted[i + 1]] - vals[idx_sorted[i - 1]]) / val_range
            return cd
        
        while used < self.budget:
            # Get non-dominated front
            nd_idx = non_dominated_sort(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(pop_size)
            
            # Compute crowding distance on non-dominated front
            cd = crowding_distance(F, nd_idx)
            
            # Select parent: prefer high crowding distance
            parent_idx = nd_idx[np.argmax(cd)]
            parent = X[parent_idx].copy()
            
            # Mutate: Gaussian perturbation with adaptive step size
            # Step size decreases as we use more budget
            progress = used / self.budget
            step_size = 0.15 * (1 - 0.5 * progress)
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Add to archive
            all_X = np.vstack([all_X, child])
            all_F = np.vstack([all_F, fc])
            
            # Replace worst in population based on domination + crowding
            # Find a point to replace: prefer dominated points with low crowding
            dominated_mask = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_mask[i] = True
                        break
            
            if np.any(dominated_mask):
                # Among dominated, pick one with lowest crowding-like measure
                dom_idx = np.where(dominated_mask)[0]
                # Use sum of objectives as simple fitness
                replace_idx = dom_idx[np.argmax(np.sum(F[dom_idx], axis=1))]
            else:
                # All non-dominated, replace one with lowest crowding distance
                cd_all = crowding_distance(F, np.arange(pop_size))
                replace_idx = np.argmin(cd_all)
            
            X[replace_idx] = child
            F[replace_idx] = fc
        
        return all_X
# EVOLVE-BLOCK-END
