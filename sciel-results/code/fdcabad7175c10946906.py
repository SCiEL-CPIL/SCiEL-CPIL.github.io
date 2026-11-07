"""Optimized solver for BBOB f23 (Katsuura) with multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _halton(self, n, dim, skip=1):
        """Generate Halton sequence points in [0,1]^dim."""
        points = []
        for i in range(skip, skip + n):
            point = []
            for j in range(dim):
                base = j + 2
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                point.append(r)
            points.append(point)
        return np.array(points)

    def _nelder_mead(self, func, x0, max_iter, initial_step):
        """Nelder-Mead simplex method."""
        dim = len(x0)
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            if xi[i] != 0:
                xi[i] *= 1.0 + initial_step
            else:
                xi[i] += initial_step
            simplex.append(xi)
        
        fvals = [func(np.clip(x, -5, 5)) for x in simplex]
        
        for _ in range(max_iter):
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]
            
            # Reflection
            xr = centroid + (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + 2 * (centroid - worst)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + 0.5 * (worst - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                if fc < worst_f:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
        
        return np.array(simplex), fvals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin
        dim = self.dim
        budget = self.budget
        
        # Generate Halton sequence for initial points
        n_initial = 10
        halton_pts = self._halton(n_initial, dim)
        # Map to [-5, 5]
        init_points = halton_pts * 10 - 5
        
        # Also add center and some random points
        rng = np.random.default_rng(42)
        extra_points = rng.uniform(-5, 5, size=(5, dim))
        center = np.zeros(dim)
        
        all_starts = list(init_points) + list(extra_points) + [center]
        
        # Evaluate all starting points
        best_x = all_starts[0]
        best_f = func(np.clip(best_x, -5, 5))
        
        # Track which starts we've processed
        processed = set()
        
        # Phase 1: Quick evaluation of all starts
        for i, x0 in enumerate(all_starts):
            if time.time() > deadline - 60:
                break
            x0_clipped = np.clip(x0, -5, 5)
            f0 = func(x0_clipped)
            if f0 < best_f:
                best_f = f0
                best_x = x0_clipped
            processed.add(i)
        
        # Phase 2: Nelder-Mead refinement from best starts
        # Sort starts by initial function value
        start_fvals = []
        for i, x0 in enumerate(all_starts):
            if i in processed:
                x0_clipped = np.clip(x0, -5, 5)
                start_fvals.append((x0_clipped, best_f if i == 0 else None))
        
        # Re-evaluate to get proper ordering
        start_results = []
        for i, x0 in enumerate(all_starts):
            x0_clipped = np.clip(x0, -5, 5)
            f0 = func(x0_clipped)
            start_results.append((f0, x0_clipped))
        
        start_results.sort(key=lambda x: x[0])
        
        # Refine top starts with Nelder-Mead
        n_refine = min(5, len(start_results))
        for idx in range(n_refine):
            if time.time() > deadline - 30:
                break
            f0, x0 = start_results[idx]
            if f0 < best_f:
                best_f = f0
                best_x = x0
            
            # Nelder-Mead with adaptive iteration count
            max_iter = 80
            initial_step = 0.5
            simplex, fvals = self._nelder_mead(func, x0, max_iter, initial_step)
            
            # Find best in simplex
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx]
        
        # Phase 3: If budget remains, do more refinement
        if time.time() < deadline - 10:
            # Second round of refinement from current best
            simplex, fvals = self._nelder_mead(func, best_x, 50, 0.1)
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx]
        
        return best_x, best_f
# EVOLVE-BLOCK-END
