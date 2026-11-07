"""Multi-objective optimizer for WFG7 using Tchebycheff decomposition + local search."""
import time
import numpy as np


class Optimizer:
    """Decomposition-based multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(777)

        used = 0

        def _eval(x):
            nonlocal used
            f = problem(x)
            used += 1
            return np.asarray(f, dtype=np.float64)

        def _dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def _tcheb(f, w, ideal):
            return np.max(w * (f - ideal))

        # Generate 30 uniformly distributed weight vectors in 3D (simplex)
        n_weights = 30
        weights = []
        # Use a grid-like approach for 3D weights that sum to 1
        # We'll use a combination of edge and interior points
        for i in range(6):
            for j in range(6):
                if i + j <= 6:
                    w0 = i / 6.0
                    w1 = j / 6.0
                    w2 = 1.0 - w0 - w1
                    if w2 >= -1e-10:
                        weights.append(np.array([max(0, w0), max(0, w1), max(0, w2)]))
        # Deduplicate and normalize
        unique_weights = []
        for w in weights:
            s = np.sum(w)
            if s > 0:
                wn = w / s
                # Check if already exists
                dup = False
                for uw in unique_weights:
                    if np.allclose(wn, uw, atol=1e-6):
                        dup = True
                        break
                if not dup:
                    unique_weights.append(wn)
        
        # If we don't have enough, add more
        while len(unique_weights) < n_weights:
            w = rng.random(3)
            w = w / np.sum(w)
            dup = False
            for uw in unique_weights:
                if np.allclose(w, uw, atol=1e-6):
                    dup = True
                    break
            if not dup:
                unique_weights.append(w)
        
        weights = np.array(unique_weights[:n_weights])
        n_w = len(weights)

        # Initialize: evaluate center and corners
        center = (lo + hi) / 2.0
        f_center = _eval(center)
        
        # Evaluate a few random points to get a better ideal
        ideal = np.array([3.0, 5.0, 7.0])
        for _ in range(5):
            x = lo + (hi - lo) * rng.random(self.dim)
            f = _eval(x)
            ideal = np.minimum(ideal, f)
        
        # Initialize subproblem solutions at center
        sols = np.tile(center, (n_w, 1))
        sols_f = np.tile(f_center, (n_w, 1))
        # Evaluate each subproblem's initial point with its weight
        for i in range(n_w):
            sols_f[i] = _eval(sols[i])
            ideal = np.minimum(ideal, sols_f[i])

        # Local search function for a single subproblem
        def _local_search(x0, w, f0, max_iters=15):
            nonlocal used
            x_best = x0.copy()
            f_best = f0.copy()
            t_best = _tcheb(f_best, w, ideal)
            
            # Nelder-Mead-like simplex for Tchebycheff scalarization
            n = self.dim
            # Create initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x_best
            for i in range(n):
                simplex[i + 1] = x_best.copy()
                step = 0.05 * (hi[i] - lo[i])
                simplex[i + 1][i] += step
                simplex[i + 1] = np.clip(simplex[i + 1], lo, hi)
            
            # Evaluate simplex
            simp_f = np.zeros((n + 1, self.n_obj))
            for i in range(n + 1):
                if i == 0:
                    simp_f[i] = f_best
                else:
                    simp_f[i] = _eval(simplex[i])
            
            # Sort by Tchebycheff value
            t_vals = np.array([_tcheb(simp_f[i], w, ideal) for i in range(n + 1)])
            order = np.argsort(t_vals)
            simplex = simplex[order]
            simp_f = simp_f[order]
            t_vals = t_vals[order]
            
            for _ in range(max_iters):
                if used >= self.budget or time.time() > deadline - 5:
                    break
                
                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + 1.0 * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = _eval(x_ref)
                t_ref = _tcheb(f_ref, w, ideal)
                
                if t_ref < t_vals[0]:
                    # Expansion
                    x_exp = centroid + 2.0 * (x_ref - centroid)
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = _eval(x_exp)
                    t_exp = _tcheb(f_exp, w, ideal)
                    if t_exp < t_ref:
                        simplex[-1] = x_exp
                        simp_f[-1] = f_exp
                        t_vals[-1] = t_exp
                    else:
                        simplex[-1] = x_ref
                        simp_f[-1] = f_ref
                        t_vals[-1] = t_ref
                elif t_ref < t_vals[-2]:
                    simplex[-1] = x_ref
                    simp_f[-1] = f_ref
                    t_vals[-1] = t_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = _eval(x_con)
                    t_con = _tcheb(f_con, w, ideal)
                    if t_con < t_vals[-1]:
                        simplex[-1] = x_con
                        simp_f[-1] = f_con
                        t_vals[-1] = t_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            simp_f[i] = _eval(simplex[i])
                            t_vals[i] = _tcheb(simp_f[i], w, ideal)
                
                # Re-sort
                order = np.argsort(t_vals)
                simplex = simplex[order]
                simp_f = simp_f[order]
                t_vals = t_vals[order]
                
                if t_vals[0] < t_best:
                    x_best = simplex[0].copy()
                    f_best = simp_f[0].copy()
                    t_best = t_vals[0]
            
            return x_best, f_best

        # Main loop: iterate through subproblems
        n_rounds = max(1, (self.budget - used) // (n_w * 8))
        
        for round_i in range(n_rounds):
            if used >= self.budget or time.time() > deadline - 10:
                break
            
            for i in range(n_w):
                if used >= self.budget or time.time() > deadline - 5:
                    break
                
                # Update ideal
                ideal = np.min(sols_f, axis=0)
                
                # Local search for subproblem i
                x_new, f_new = _local_search(sols[i], weights[i], sols_f[i], max_iters=8)
                
                # Update solution for this subproblem
                sols[i] = x_new
                sols_f[i] = f_new
                
                # Update ideal
                ideal = np.minimum(ideal, f_new)

        # Final polish: do a few more local searches on the best solutions
        # Find the best Tchebycheff value for each weight and keep those
        final_sols = sols.copy()
        final_sols_f = sols_f.copy()
        
        # Additional local search on a subset
        if used < self.budget and time.time() < deadline - 10:
            # Pick top 10 diverse solutions
            # Sort by sum of objectives
            sums = np.sum(final_sols_f, axis=1)
            top_idx = np.argsort(sums)[:min(10, n_w)]
            
            for idx in top_idx:
                if used >= self.budget or time.time() > deadline - 3:
                    break
                x_new, f_new = _local_search(final_sols[idx], weights[idx], final_sols_f[idx], max_iters=5)
                final_sols[idx] = x_new
                final_sols_f[idx] = f_new

        return final_sols
