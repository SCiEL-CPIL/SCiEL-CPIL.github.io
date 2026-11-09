"""Adaptive covariance local search with Sobol global scan, time-guarded."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Sobol global scan followed by adaptive-covariance local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol(self, n, dim, seed):
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
            pts = sampler.random(n)
            return pts
        except Exception:
            def halton(index, base):
                f, r = 1.0, 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            pts = np.empty((n, dim))
            for d in range(dim):
                for i in range(n):
                    pts[i, d] = halton(i + 1, d + 2)
            return pts

    def _adaptive_local(self, func, x0, calls_left, deadline, best_state, max_iter=150):
        """
        Adaptive local search that adjusts step sizes per dimension based on
        progress. This helps with anisotropic functions like Bent Cigar.
        """
        dim = self.dim
        # Initialize step sizes: start moderate
        steps = np.ones(dim) * 0.5
        # Track progress: for each dimension, track if we're making progress
        last_best = best_state[0]
        progress_count = np.zeros(dim)
        
        x = x0.copy()
        fx = func(x)
        calls_left[0] -= 1
        if fx < best_state[0]:
            best_state[0] = fx
            best_state[1] = x.copy()
        
        for iteration in range(max_iter):
            if time.time() >= deadline or calls_left[0] <= 1:
                break
            
            # Try moving in each dimension with adaptive step
            improved = False
            for d in range(dim):
                if time.time() >= deadline or calls_left[0] <= 1:
                    break
                
                # Try positive and negative direction
                for direction in [1, -1]:
                    if calls_left[0] <= 1:
                        break
                    x_trial = x.copy()
                    x_trial[d] += direction * steps[d]
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_left[0] -= 1
                    
                    if f_trial < fx:
                        # Progress in this direction
                        x = x_trial.copy()
                        fx = f_trial
                        if fx < best_state[0]:
                            best_state[0] = fx
                            best_state[1] = x.copy()
                        progress_count[d] += 1
                        improved = True
                        # Keep step size or slightly increase
                        steps[d] = min(steps[d] * 1.1, 2.0)
                    else:
                        # No progress: shrink step size
                        steps[d] = max(steps[d] * 0.7, 0.01)
                        # Try a smaller step in the other direction
                        if calls_left[0] <= 1:
                            break
                        x_trial2 = x.copy()
                        x_trial2[d] -= direction * steps[d]
                        x_trial2 = np.clip(x_trial2, -5, 5)
                        f_trial2 = func(x_trial2)
                        calls_left[0] -= 1
                        if f_trial2 < fx:
                            x = x_trial2.copy()
                            fx = f_trial2
                            if fx < best_state[0]:
                                best_state[0] = fx
                                best_state[1] = x.copy()
                            progress_count[d] += 1
                            improved = True
                            steps[d] = min(steps[d] * 1.1, 2.0)
            
            # If no improvement in any dimension, try a random perturbation
            if not improved:
                if calls_left[0] <= 1:
                    break
                # Random small perturbation
                rng = np.random.default_rng(iteration + 1)
                x_trial = x + rng.normal(0, 0.1, dim)
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls_left[0] -= 1
                if f_trial < fx:
                    x = x_trial.copy()
                    fx = f_trial
                    if fx < best_state[0]:
                        best_state[0] = fx
                        best_state[1] = x.copy()
                # Reset some step sizes if stuck
                steps[progress_count < 2] = np.maximum(steps[progress_count < 2] * 0.5, 0.01)
            
            # Periodically reset progress count
            if iteration % 20 == 0 and iteration > 0:
                progress_count *= 0.5
        
        return x, fx

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget
        calls_left = [budget]

        def guarded(x):
            if calls_left[0] <= 0:
                return float('inf')
            calls_left[0] -= 1
            return float(func(x))

        rng_seed = 12345

        # Phase 1: Sobol global scan
        n_scan = min(150, budget // 3)
        if n_scan < 20:
            n_scan = 20
        u = self._sobol(n_scan, dim, rng_seed)
        pts = -5.0 + 10.0 * u
        pts = np.clip(pts, -5, 5)

        # Evaluate scan points, tracking values
        vals = np.empty(n_scan)
        best_val = float('inf')
        best_idx = 0
        for i in range(n_scan):
            if time.time() >= deadline or calls_left[0] <= 0:
                break
            vals[i] = guarded(pts[i])
            if vals[i] < best_val:
                best_val = vals[i]
                best_idx = i

        if best_val == float('inf'):
            best_val = guarded(np.zeros(dim))
            best_idx = 0
            pts[0] = np.zeros(dim)

        best_state = [best_val, pts[best_idx].copy()]

        # Phase 2: Adaptive local refinement from top candidates
        # Sort by value and try top few
        order = np.argsort(vals)
        top_k = min(5, n_scan)
        
        for rank in range(top_k):
            idx = order[rank]
            if time.time() >= deadline or calls_left[0] <= 2 * dim + 5:
                break
            # Skip if this is the same as the best
            if np.allclose(pts[idx], best_state[1]):
                continue
            # Only try if plausibly good (within some factor of best)
            if vals[idx] > best_state[0] * 2.0 + 5.0 and best_state[0] > 0:
                continue
            if vals[idx] > best_state[0] + 10.0 and best_state[0] <= 0:
                continue
            
            self._adaptive_local(guarded, pts[idx].copy(), calls_left, deadline, best_state, max_iter=100)
        
        # Final polish on the best
        if calls_left[0] > 2 * dim + 5 and time.time() < deadline:
            self._adaptive_local(guarded, best_state[1].copy(), calls_left, deadline, best_state, max_iter=150)

        return best_state[1], best_state[0]
# EVOLVE-BLOCK-END
