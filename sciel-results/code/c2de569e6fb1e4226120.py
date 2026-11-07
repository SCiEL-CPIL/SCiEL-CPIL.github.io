import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290

        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Broad random sampling to get a good initial estimate
        # Sample 20 points uniformly in [-4, 4]^dim
        n_initial = 20
        for _ in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            eval_func(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Use the best point to estimate the rotation matrix
        # Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # For rotated Rastrigin: f(x) = 10*n + sum((R*x + s)_i^2 - 10*cos(2*pi*(R*x + s)_i))
        # where R is orthogonal and s is a shift.
        # The minima occur when R*x + s is at integer points.
        # We can try to estimate R and s by evaluating points near the best.
        
        # Strategy: Use the best point as anchor, then do a lattice search
        # around it in the rotated frame.
        # First, let's do a coarse grid search in the neighborhood of the best point.
        
        x_best = best_x.copy()
        f_best = best_f
        
        # Coarse local search: evaluate points on a grid around x_best
        # Step size 0.5, radius 1.5 (so 7^5 = 16807 points, too many)
        # Let's use a smarter approach: coordinate-wise with adaptive steps
        
        # Phase 3: Multi-start local search with Rastrigin-aware jumps
        # Key insight: In the rotated frame, minima are at integer points.
        # If we're near a minimum in the rotated frame, moving to the nearest
        # integer in each coordinate (in the rotated frame) should get us closer.
        # But we don't know the rotation. So we use a heuristic:
        # Try rounding each coordinate to nearest integer, and combinations thereof.
        
        # First, do a quick coordinate descent to get closer to a local min
        x_cur = x_best.copy()
        f_cur = f_best
        step = 0.2
        for _ in range(20):
            if calls >= budget or time.time() > deadline:
                break
            improved = False
            for d in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                for sign in [1, -1]:
                    if calls >= budget or time.time() > deadline:
                        break
                    y = x_cur.copy()
                    y[d] += sign * step
                    y = np.clip(y, -5, 5)
                    fy = eval_func(y)
                    if fy < f_cur:
                        x_cur, f_cur = y, fy
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 0.01:
                    break
        
        if f_cur < best_f:
            best_f = f_cur
            best_x = x_cur.copy()
        
        # Phase 4: Lattice-based search - try to snap to nearby "integer" points
        # in a local coordinate system. Since we don't know the rotation,
        # we try all combinations of rounding coordinates to nearest integers
        # with small perturbations.
        
        # Generate candidate points: for each coordinate, try floor, round, ceil
        # and combinations of these.
        base = x_cur
        candidates = []
        
        # All combinations of rounding each coordinate to floor/round/ceil
        for mask in range(3 ** dim):
            if calls >= budget or time.time() > deadline:
                break
            y = base.copy()
            m = mask
            for d in range(dim):
                digit = m % 3
                m //= 3
                if digit == 0:
                    y[d] = np.floor(y[d])
                elif digit == 1:
                    y[d] = np.round(y[d])
                else:
                    y[d] = np.ceil(y[d])
            y = np.clip(y, -5, 5)
            candidates.append(y)
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for y in candidates:
            key = tuple(np.round(y, 6))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(y)
        
        # Sort by distance to x_cur and evaluate closest first
        unique_candidates.sort(key=lambda y: np.linalg.norm(y - x_cur))
        
        for y in unique_candidates:
            if calls >= budget or time.time() > deadline:
                break
            fy = eval_func(y)
            if fy < f_cur:
                f_cur = fy
                x_cur = y.copy()
                # If we found a better point, do another round of coordinate descent
                step = 0.1
                for _ in range(10):
                    if calls >= budget or time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        for sign in [1, -1]:
                            if calls >= budget or time.time() > deadline:
                                break
                            z = x_cur.copy()
                            z[d] += sign * step
                            z = np.clip(z, -5, 5)
                            fz = eval_func(z)
                            if fz < f_cur:
                                x_cur, f_cur = z, fz
                                improved = True
                                break
                    if not improved:
                        step *= 0.5
                        if step < 0.005:
                            break
        
        if f_cur < best_f:
            best_f = f_cur
            best_x = x_cur.copy()
        
        # Phase 5: Final fine-tuning
        if calls < budget and time.time() < deadline:
            step = 0.05
            for _ in range(20):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_cur.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f_cur:
                            x_cur, f_cur = y, fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.005:
                        break
        
        if f_cur < best_f:
            best_f = f_cur
            best_x = x_cur.copy()
        
        return best_x, best_f
