"""Multi-objective optimizer on RE21 using HypE (hypervolume estimation) with Sobol initialization."""
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
            for i in range(len(new_xs)):
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

        def sobol_sample(n, dim):
            """Generate quasi-random Sobol-like samples in [0,1]^dim using Halton sequence."""
            samples = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    base = j + 2  # primes: 2, 3, 5, 7
                    f = 1.0
                    r = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    samples[i, j] = r
            return samples

        # Initial Sampling using stratified/Sobol-like design
        n_init = min(30, B)
        init_samples = sobol_sample(n_init, d)
        for i in range(n_init):
            if time.time() > deadline or B <= 0:
                break
            x = lo + span * init_samples[i]
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            if np.all(np.isfinite(f)):
                update_archive(x, f)
                B -= 1

        # HypE-style selection
        def select_parent_hype():
            """Select parent using HypE-style weighted sampling based on HV contribution."""
            n = len(xs)
            if n == 0:
                return None
            if n == 1:
                return 0
            
            # Compute marginal contributions
            contribs = np.array([marginal_hv_contribution(i) for i in range(n)])
            
            # Add small epsilon to avoid zero weights
            contribs = contribs + 1e-10
            probs = contribs / contribs.sum()
            
            return np.random.choice(n, p=probs)

        # Main Evolutionary Loop
        pop_size = 8
        
        while B > 0 and time.time() < deadline:
            # Generate candidates
            candidates = []
            
            # 1. Differential Evolution with HypE-based parent selection
            if len(xs) > 2:
                for _ in range(pop_size):
                    if np.random.random() < 0.7:
                        # DE/rand/1
                        i1 = select_parent_hype()
                        i2 = select_parent_hype()
                        i3 = select_parent_hype()
                        # Ensure distinct indices if possible
                        attempts = 0
                        while len(set([i1, i2, i3])) < 3 and attempts < 10:
                            i1 = select_parent_hype()
                            i2 = select_parent_hype()
                            i3 = select_parent_hype()
                            attempts += 1
                        
                        F = 0.5 + np.random.random() * 0.5
                        v = xs[i1] + F * (xs[i2] - xs[i3])
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
                    else:
                        # Local mutation
                        i = select_parent_hype()
                        elapsed_frac = min(1.0, (time.time() - start_time) / (deadline - start_time))
                        base_sigma = 0.1 * span
                        sigma = base_sigma * (1.0 - 0.7 * elapsed_frac)
                        v = xs[i] + np.random.normal(0, 1, d) * sigma
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
            
            # 2. Random exploration (decreasing over time)
            elapsed_frac = min(1.0, (time.time() - start_time) / (deadline - start_time))
            n_random = max(1, int(3 * (1.0 - 0.5 * elapsed_frac)))
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

        # Final polish: tight local search around best archive points
        if len(xs) > 0 and B > 0 and time.time() < deadline:
            n_polish = min(len(xs), 8)
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
                for trial in range(4):
                    if B <= 0 or time.time() > deadline:
                        break
                    sigma = 0.015 * span
                    v = xs[idx] + np.random.normal(0, 1, d) * sigma
                    v = np.clip(v, lo, hi)
                    f = problem(v)
                    f = np.array(f, dtype=np.float64)
                    if np.all(np.isfinite(f)):
                        update_archive(v, f)
                        B -= 1

        # Return the archive
        if len(fs) > 0:
            return np.array(fs)
        else:
            return np.array([])

# EVOLVE-BLOCK-END
