import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit

        # Track all evaluated points for hypervolume
        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        # Phase 1: Halton sequence for space-filling initialization
        # Use enough points to get good coverage
        n_init = min(50, budget)
        
        if n_init >= budget:
            # Just do random sampling if budget is very small
            for _ in range(budget):
                x = lo + (hi - lo) * np.random.random(dim)
                eval_point(x)
            return np.array(all_X)

        # Generate Halton sequence points
        halton_points = self._halton_sequence(n_init, dim)
        # Map to [lo, hi]
        X_init = lo + halton_points * (hi - lo)
        
        for x in X_init:
            if used >= budget:
                break
            eval_point(x)

        # Phase 2: Identify non-dominated points and refine with Nelder-Mead
        F_arr = np.array(all_F)
        X_arr = np.array(all_X)
        
        # Get non-dominated indices
        nd_indices = self._get_non_dominated_indices(F_arr)
        
        if len(nd_indices) == 0:
            return X_arr

        # Sort non-dominated points by their "quality" - use sum of normalized objectives
        # Lower sum means better
        nd_F = F_arr[nd_indices]
        # Normalize each objective to [0,1] range based on current min/max
        f_min = F_arr.min(axis=0)
        f_max = F_arr.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-14] = 1.0
        nd_F_norm = (nd_F - f_min) / f_range
        nd_scores = nd_F_norm.sum(axis=1)
        
        # Pick top k non-dominated points for refinement
        k_refine = min(5, len(nd_indices))
        top_k_local = np.argsort(nd_scores)[:k_refine]
        refine_indices = nd_indices[top_k_local]

        # Phase 3: Nelder-Mead refinement on each selected point
        # We'll use a simple coordinate descent / Nelder-Mead hybrid
        remaining_budget = budget - used
        
        # Distribute budget among refinement points
        evals_per_point = max(10, remaining_budget // max(1, k_refine))
        
        for idx in refine_indices:
            if used >= budget - 5:  # Leave some buffer
                break
            if time.time() > deadline:
                break
                
            x_start = X_arr[idx].copy()
            f_start = F_arr[idx].copy()
            
            # Nelder-Mead style optimization
            best_x, best_f = self._nelder_mead(
                eval_point, x_start, f_start, lo, hi, 
                max_evals=evals_per_point, deadline=deadline
            )
            
            # If we found a better point, add it
            if best_f is not None and np.all(best_f <= f_start) and np.any(best_f < f_start):
                # Already added by eval_point
                pass

        # Phase 4: Use remaining budget for additional space-filling or random search
        if used < budget - 2:
            if time.time() < deadline - 5:
                # Add more Halton points or random points to fill remaining budget
                n_extra = budget - used
                # Use a different offset for Halton to get new points
                extra_halton = self._halton_sequence(n_extra, dim, offset=0.5)
                X_extra = lo + extra_halton * (hi - lo)
                
                for x in X_extra:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    eval_point(x)

        return np.array(all_X)

    def _halton_sequence(self, n, dim, offset=0.0):
        """
        Generate n points in dim dimensions using the Halton sequence.
        Returns array of shape (n, dim) with values in [0, 1).
        """
        # First n primes for Halton bases
        primes = self._first_primes(dim)
        
        points = np.zeros((n, dim))
        for d in range(dim):
            base = primes[d]
            for i in range(n):
                points[i, d] = self._halton_value(i + 1 + int(offset * 1000), base)
        
        return points

    def _halton_value(self, i, base):
        """
        Compute the i-th value in the Halton sequence with given base.
        """
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _first_primes(self, n):
        """
        Return the first n prime numbers.
        """
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def _get_non_dominated_indices(self, F):
        """
        Return indices of non-dominated points in F.
        """
        n = F.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        
        return np.where(~is_dominated)[0]

    def _nelder_mead(self, eval_func, x_start, f_start, lo, hi, max_evals=50, deadline=None):
        """
        Simple Nelder-Mead style optimization for multi-objective.
        We minimize the sum of objectives as a scalar surrogate.
        Returns (best_x, best_f).
        """
        dim = len(x_start)
        n_obj = len(f_start)
        
        # Initialize simplex
        alpha = 0.05  # Step size for initial simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x_start
        f0 = f_start.copy()
        
        for i in range(dim):
            simplex[i + 1] = x_start.copy()
            step = alpha * (hi[i] - lo[i])
            simplex[i + 1, i] += step
            simplex[i + 1] = np.clip(simplex[i + 1], lo, hi)
        
        # Evaluate simplex
        f_values = [f0]
        for i in range(1, dim + 1):
            if deadline is not None and time.time() > deadline:
                break
            f_i = np.array(eval_func(simplex[i].copy()), dtype=np.float64)
            f_values.append(f_i)
        
        # We need to track evaluations - but eval_func already does that
        # Let's use a wrapper that doesn't double-count
        # Actually, eval_func already increments used and records, so we just call it
        
        # Nelder-Mead iterations
        # We'll minimize the sum of objectives as a scalar
        best_x = x_start.copy()
        best_f = f_start.copy()
        best_score = f_start.sum()
        
        for iteration in range(max_evals // (dim + 1)):
            if deadline is not None and time.time() > deadline:
                break
            
            # Sort simplex by objective sum
            scores = [fv.sum() for fv in f_values]
            order = np.argsort(scores)
            
            simplex_sorted = simplex[order]
            f_sorted = [f_values[i] for i in order]
            scores_sorted = [scores[i] for i in order]
            
            # Check for convergence
            if scores_sorted[0] >= scores_sorted[-1] - 1e-10:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex_sorted[:-1], axis=0)
            centroid = np.clip(centroid, lo, hi)
            
            # Reflection
            worst = simplex_sorted[-1]
            f_worst = f_sorted[-1]
            f_worst_score = scores_sorted[-1]
            
            x_ref = centroid + (centroid - worst)
            x_ref = np.clip(x_ref, lo, hi)
            f_ref = np.array(eval_func(x_ref.copy()), dtype=np.float64)
            f_ref_score = f_ref.sum()
            
            if f_ref_score < scores_sorted[0]:
                # Expansion
                x_exp = centroid + 2.0 * (x_ref - centroid)
                x_exp = np.clip(x_exp, lo, hi)
                f_exp = np.array(eval_func(x_exp.copy()), dtype=np.float64)
                f_exp_score = f_exp.sum()
                
                if f_exp_score < f_ref_score:
                    simplex_sorted[-1] = x_exp
                    f_sorted[-1] = f_exp
                    scores_sorted[-1] = f_exp_score
                else:
                    simplex_sorted[-1] = x_ref
                    f_sorted[-1] = f_ref
                    scores_sorted[-1] = f_ref_score
            elif f_ref_score < f_worst_score:
                # Accept reflection
                simplex_sorted[-1] = x_ref
                f_sorted[-1] = f_ref
                scores_sorted[-1] = f_ref_score
            else:
                # Contraction
                x_con = centroid + 0.5 * (worst - centroid)
                x_con = np.clip(x_con, lo, hi)
                f_con = np.array(eval_func(x_con.copy()), dtype=np.float64)
                f_con_score = f_con.sum()
                
                if f_con_score < f_worst_score:
                    simplex_sorted[-1] = x_con
                    f_sorted[-1] = f_con
                    scores_sorted[-1] = f_con_score
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex_sorted[i] = simplex_sorted[0] + 0.5 * (simplex_sorted[i] - simplex_sorted[0])
                        simplex_sorted[i] = np.clip(simplex_sorted[i], lo, hi)
                        f_shrink = np.array(eval_func(simplex_sorted[i].copy()), dtype=np.float64)
                        f_sorted[i] = f_shrink
                        scores_sorted[i] = f_shrink.sum()
            
            # Update simplex and f_values
            simplex = simplex_sorted
            f_values = f_sorted
            
            # Track best
            min_idx = np.argmin(scores_sorted)
            if scores_sorted[min_idx] < best_score:
                best_score = scores_sorted[min_idx]
                best_x = simplex[min_idx].copy()
                best_f = f_values[min_idx].copy()
        
        return best_x, best_f
