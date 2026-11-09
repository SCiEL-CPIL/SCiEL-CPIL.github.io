"""Multi-objective optimizer on RE21 using HypE with Sobol initialization and coordinate-descent polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin under 330s kill
        B = int(self.budget)
        d = self.dim
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        # Reference point for hypervolume calculation (as per task)
        ref = np.array([3000.0, 0.06])

        # Archive stores (x, f) for non-dominated points
        xs = []
        fs = []
        
        def dominates(f1, f2):
            """Returns True if f1 strictly dominates f2 (minimization)"""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(x_new, f_new):
            """
            Adds x_new, f_new to the archive if it is non-dominated.
            Removes any existing points dominated by the new point.
            Returns True if the point was added to the archive.
            """
            if len(fs) == 0:
                xs.append(x_new)
                fs.append(f_new)
                return True
            
            # Check if new point is dominated by any existing point
            for f_exist in fs:
                if dominates(f_exist, f_new):
                    return False
            
            # Remove any existing points dominated by the new point
            new_xs = []
            new_fs = []
            for i in range(len(fs)):
                if not dominates(f_new, fs[i]):
                    new_xs.append(xs[i])
                    new_fs.append(fs[i])
            
            new_xs.append(x_new)
            new_fs.append(f_new)
            
            xs.clear()
            fs.clear()
            for i in range(new_xs):
                xs.append(new_xs[i])
                fs.append(new_fs[i])
            return True

        def hypervolume_2d(points, ref_pt):
            """
            Calculate 2D hypervolume for a list of non-dominated points.
            """
            if len(points) == 0:
                return 0.0
            p = np.array(points)
            # Filter points strictly better than ref
            mask = (p[:, 0] < ref_pt[0]) & (p[:, 1] < ref_pt[1])
            p = p[mask]
            if len(p) == 0:
                return 0.0
            
            # Sort by f1 ascending
            p = p[np.argsort(p[:, 0])]
            f1 = p[:, 0]
            f2 = p[:, 1]
            
            total_hv = 0.0
            for i in range(len(p)):
                x1 = f1[i]
                x2 = f1[i+1] if i < len(p)-1 else ref_pt[0]
                y1 = f2[i]
                y2 = f2[i+1] if i < len(p)-1 else ref_pt[1]
                
                w = max(0.0, x2 - x1)
                h = max(0.0, y1 - y2)
                total_hv += w * h
                
            return total_hv

        def marginal_hv_contribution(idx):
            """Compute the marginal HV contribution of point idx in the archive."""
            if len(fs) <= 1:
                return hypervolume_2d([fs[idx]], ref)
            
            # HV with the point
            hv_with = hypervolume_2d(fs, ref)
            # HV without the point
            fs_without = fs[:idx] + fs[idx+1:]
            hv_without = hypervolume_2d(fs_without, ref)
            return max(0.0, hv_with - hv_without)

        def select_parent_by_hv():
            """Select a parent proportional to its marginal HV contribution."""
            n = len(xs)
            if n == 0:
                return None
            if n == 1:
                return 0
            
            # Compute marginal contributions
            contribs = []
            for i in range(n):
                c = marginal_hv_contribution(i)
                contribs.append(c)
            
            total = sum(contribs)
            if total <= 0:
                return np.random.randint(n)
            
            # Normalize to probabilities
            probs = np.array(contribs) / total
            # Add small epsilon to avoid zero probabilities
            probs = probs + 1e-10
            probs = probs / probs.sum()
            
            return np.random.choice(n, p=probs)

        def sobol_sample(n, dim):
            """Generate quasi-random Sobol-like samples in [0,1]^dim using Halton sequence."""
            # Use Halton sequence for low-discrepancy sampling
            def halton(index, base):
                result = 0.0
                f = 1.0 / base
                i = index
                while i > 0:
                    result += f * (i % base)
                    i //= base
                    f /= base
                return result
            
            samples = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    # Use different primes for each dimension
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    samples[i, j] = halton(i + 1, primes[j % len(primes)])
            return samples

        # Initial Sampling with Sobol/Halton for better coverage
        n_init = min(20, B)
        if n_init > 0:
            sobol_pts = sobol_sample(n_init, d)
            for i in range(n_init):
                if time.time() > deadline:
                    break
                x = lo + span * sobol_pts[i]
                f = problem(x)
                f = np.array(f, dtype=np.float64)
                if np.all(np.isfinite(f)):
                    update_archive(x, f)
                    B -= 1

        # Main Evolutionary Loop
        pop_size = 10
        
        while B > 0 and time.time() < deadline:
            # Generate candidates
            candidates = []
            
            # 1. Mutation from existing archive using HV-based selection
            if len(xs) > 0:
                for _ in range(pop_size):
                    if len(xs) > 2 and np.random.random() < 0.6:
                        # Differential Evolution style with HV-based parent selection
                        i1 = select_parent_by_hv()
                        i2 = select_parent_by_hv()
                        i3 = select_parent_by_hv()
                        F = 0.5 + np.random.random() * 0.5
                        v = xs[i1] + F * (xs[i2] - xs[i3])
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
                    else:
                        # Local mutation around an HV-weighted point
                        i = select_parent_by_hv()
                        # Adaptive step size: smaller as we converge
                        elapsed_frac = (time.time() - start_time) / (deadline - start_time)
                        base_sigma = 0.15 * span
                        sigma = base_sigma * (1.0 - 0.5 * elapsed_frac)
                        v = xs[i] + np.random.normal(0, 1, d) * sigma
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
            
            # 2. Random exploration (fewer as we progress)
            elapsed_frac = (time.time() - start_time) / (deadline - start_time)
            n_random = max(1, int(5 * (1.0 - 0.5 * elapsed_frac)))
            for _ in range(n_random):
                if B <= 0:
                    break
                x = lo + span * np.random.random(d)
                candidates.append(x)
            
            # Evaluate candidates
            for x in candidates:
                if B <= 0 or time.time() > deadline:
                    break
                f = problem(x)
                f = np.array(f, dtype=np.float64)
                if not np.all(np.isfinite(f)):
                    continue
                update_archive(x, f)
                B -= 1

        # Final polish: deterministic coordinate-descent local search on top HV contributors
        if len(xs) > 0 and B > 0 and time.time() < deadline:
            # Sort archive by marginal HV contribution (descending) and polish top points
            n_polish = min(len(xs), 10)
            # Select top points by HV contribution
            contribs = []
            for i in range(len(xs)):
                c = marginal_hv_contribution(i)
                contribs.append((c, i))
            contribs.sort(reverse=True)
            top_indices = [idx for _, idx in contribs[:n_polish]]
            
            for idx in top_indices:
                if B <= 0 or time.time() > deadline:
                    break
                
                # Coordinate descent: try perturbing each coordinate
                x_best = xs[idx].copy()
                f_best = fs[idx].copy()
                
                for coord in range(d):
                    if B <= 0 or time.time() > deadline:
                        break
                    
                    # Try positive and negative perturbations
                    for direction in [1, -1]:
                        if B <= 0 or time.time() > deadline:
                            break
                        
                        # Adaptive step size for coordinate descent
                        elapsed_frac = (time.time() - start_time) / (deadline - start_time)
                        step_size = 0.05 * span[coord] * (1.0 - 0.5 * elapsed_frac)
                        
                        x_try = x_best.copy()
                        x_try[coord] += direction * step_size
                        x_try = np.clip(x_try, lo, hi)
                        
                        f_try = problem(x_try)
                        f_try = np.array(f_try, dtype=np.float64)
                        if not np.all(np.isfinite(f_try)):
                            continue
                        
                        B -= 1
                        
                        # Check if this improves the hypervolume contribution
                        if not dominates(f_best, f_try):
                            # Update best if non-dominated or strictly better in at least one objective
                            if np.any(f_try < f_best):
                                x_best = x_try
                                f_best = f_try
                
                # Update archive with improved point
                if np.any(f_best < fs[idx]) or not dominates(fs[idx], f_best):
                    update_archive(x_best, f_best)

        # Return the archive
        if len(fs) > 0:
            return np.array(fs)
        else:
            return np.array([])

# EVOLVE-BLOCK-END
