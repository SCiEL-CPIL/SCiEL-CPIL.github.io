import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        """Compute the Halton sequence value for a given index and base."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_sequence(self, n, dim, offset=0):
        """Generate n points in [0,1)^dim using Halton sequence."""
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                points[i, j] = self._halton(i + 1 + offset, j + 2)
        return points

    def _nelder_mead(self, func, x0, max_iter=500, tol=1e-10, deadline=None):
        """
        Adaptive Nelder-Mead simplex method.
        Returns (best_x, best_f, calls_used).
        """
        dim = len(x0)
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.1

        fvals = np.array([func(simplex[i]) for i in range(dim + 1)])
        calls = dim + 1

        # Adaptive step size
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        best_x = simplex[np.argmin(fvals)]
        best_f = fvals.min()

        for iteration in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Check convergence
            if np.max(fvals) - np.min(fvals) < tol:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                calls += 1
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
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                calls += 1
                if fc <= min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                    if deadline is not None and time.time() > deadline:
                        break

            if deadline is not None and time.time() > deadline:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Systematic Halton exploration
        # Use first ~100 evaluations for broad coverage
        n_halton = min(100, budget // 5)
        halton_points = self._halton_sequence(n_halton, dim)
        # Map from [0,1)^dim to [-5,5]^dim
        halton_points = halton_points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        calls = 0

        # Evaluate Halton points
        for i in range(n_halton):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multi-trajectory (1+1)-ES refinement
        if best_x is not None and calls < budget:
            # Initialize 5 trajectories from the best Halton point with different scales
            n_traj = 5
            step_sizes = [0.1, 0.5, 1.0, 2.0, 4.0]
            x = [best_x.copy() for _ in range(n_traj)]
            sigma = step_sizes.copy()
            best_f_traj = [best_f] * n_traj
            recent_hist = [[] for _ in range(n_traj)]
            calls_since_global_improve = 0

            # Main loop for ES
            while calls < budget:
                if time.time() > deadline:
                    break
                
                # 1. Determine which trajectory to update
                scores = []
                for i in range(n_traj):
                    if len(recent_hist[i]) == 0:
                        scores.append(0.0)
                        continue
                    hist = recent_hist[i][-20:]
                    if len(hist) >= 2:
                        f_start = hist[0][1]
                        f_end = hist[-1][1]
                        imp = f_start - f_end
                    else:
                        imp = 0.0
                    scores.append(imp)
                
                # Add noise to prevent deterministic cycling
                scores_noisy = [s + rng.normal(0, 0.1) for s in scores]
                
                # Selection: 80% best, 20% random
                if rng.random() < 0.8:
                    idx = np.argmax(scores_noisy)
                else:
                    idx = rng.integers(0, n_traj)
                
                # 2. Generate offspring for selected trajectory
                z = rng.standard_normal(dim)
                y = x[idx] + sigma[idx] * z
                y = np.clip(y, -5, 5)
                
                # 3. Evaluate
                fy = func(y)
                calls += 1
                
                # Update history
                recent_hist[idx].append((calls, fy))
                if len(recent_hist[idx]) > 20:
                    recent_hist[idx] = recent_hist[idx][-20:]
                    
                # 4. Update Global Best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    calls_since_global_improve = 0
                else:
                    calls_since_global_improve += 1
                    
                # 5. Update Trajectory State
                if fy < best_f_traj[idx]:
                    # Improvement
                    best_f_traj[idx] = fy
                    x[idx] = y.copy()
                    sigma[idx] *= 1.05 # Expand
                else:
                    # No improvement
                    sigma[idx] *= 0.95 # Contract
                    
                sigma[idx] = np.clip(sigma[idx], 0.01, 10.0)
                
                # 6. Global Stagnation Handling
                # If no global improvement for 100 calls, restart a random trajectory from global best + noise
                if calls_since_global_improve >= 100:
                    restart_idx = rng.integers(0, n_traj)
                    perturbation = 2.0 * rng.standard_normal(dim)
                    x[restart_idx] = np.clip(best_x + perturbation, -5, 5)
                    sigma[restart_idx] = 1.0 # Reset to medium scale
                    best_f_traj[restart_idx] = best_f
                    recent_hist[restart_idx] = []
                    calls_since_global_improve = 0

        # Phase 3: Final Nelder-Mead polish if budget remains
        remaining = budget - calls
        if best_x is not None and remaining > 10 and time.time() < deadline - 5:
            nm_deadline = deadline - 5
            x_nm, f_nm, nm_calls = self._nelder_mead(
                func, best_x, max_iter=50, tol=1e-12, deadline=nm_deadline
            )
            calls += nm_calls
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm

        return best_x, best_f
