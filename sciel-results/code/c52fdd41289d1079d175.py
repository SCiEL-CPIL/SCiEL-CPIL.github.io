"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """BOBYQA-style derivative-free trust-region optimizer with quadratic model fitting."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # State
        x_best = np.full(n, np.nan)
        f_best = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= total_budget:
                return None
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                val = func(x_clipped)
                calls += 1
                return float(val)
            except Exception:
                return None

        # Initial point: center of the box
        x0 = np.zeros(n)
        f0 = safe_call(x0)
        if f0 is None:
            return x0, f0
        x_best = x0.copy()
        f_best = f0

        # Build initial interpolation set
        # Start with center and axis points
        points = [x0.copy()]
        fvals = [f0]

        # Add points along each axis
        initial_step = 2.0
        for i in range(n):
            for sign in [1.0, -1.0]:
                xi = x0.copy()
                xi[i] += sign * initial_step
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    break
                points.append(xi)
                fvals.append(fi)
                if fi < f_best:
                    f_best = fi
                    x_best = xi.copy()

        n_pts = len(points)
        if n_pts < n + 1:
            # Not enough points, fall back to simple ES
            self._fallback_es(func, x_best, f_best, total_budget - calls, n, start_time, deadline)
            return x_best, f_best

        # Trust region parameters
        rho = 0.5  # initial trust region radius
        rho_min = 1e-6
        rho_max = 5.0
        gamma = 0.2  # shrink factor for trust region
        beta = 2.0  # grow factor for trust region

        # Quadratic model coefficients
        # We'll fit f(x) ≈ c0 + c1^T (x-x0) + (x-x0)^T C (x-x0)
        # Using the points we have, we solve for the coefficients

        max_iter = 200
        for iteration in range(max_iter):
            # Check time and budget
            if time.time() > deadline or calls >= total_budget:
                break

            # Build design matrix for quadratic model
            # Points are in `points`, values in `fvals`
            # Reference point for the model: use current best
            x_ref = x_best.copy()

            # Build system: for each point p_j,
            # f(p_j) = c0 + c1^T (p_j - x_ref) + (p_j - x_ref)^T C (p_j - x_ref)
            # We have n_pts equations and 1 + n + n*(n+1)/2 unknowns
            # Let m = n*(n+1)//2 be the number of symmetric matrix entries

            m = n * (n + 1) // 2
            total_unknowns = 1 + n + m

            if n_pts < total_unknowns:
                # Not enough points for full quadratic model
                # Use linear model or add more points
                # For simplicity, use a linear model
                # f(p_j) ≈ c0 + c1^T (p_j - x_ref)
                # Solve least squares
                A = np.zeros((n_pts, n + 1))
                A[:, 0] = 1.0
                for j in range(n_pts):
                    A[j, 1:] = points[j] - x_ref

                b = np.array(fvals)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    c0 = sol[0]
                    c1 = sol[1:]
                    # Gradient direction: -c1
                    # Hessian: zero (linear model)
                    C = np.zeros((n, n))
                except Exception:
                    break
            else:
                # Full quadratic model
                # Vectorize symmetric matrix C
                # A_j = [1, (p_j - x_ref), vec((p_j - x_ref)(p_j - x_ref)^T)]
                A = np.zeros((n_pts, total_unknowns))
                A[:, 0] = 1.0
                for j in range(n_pts):
                    d = points[j] - x_ref
                    A[j, 1:1 + n] = d
                    # Fill in vec(d d^T) for symmetric matrix
                    idx = 0
                    for i in range(n):
                        for k in range(i, n):
                            A[j, 1 + n + idx] = d[i] * d[k]
                            idx += 1

                b = np.array(fvals)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    c0 = sol[0]
                    c1 = sol[1:1 + n]
                    # Reconstruct C from vec
                    C = np.zeros((n, n))
                    idx = 0
                    for i in range(n):
                        for k in range(i, n):
                            C[i, k] = sol[1 + n + idx]
                            C[k, i] = sol[1 + n + idx]
                            idx += 1
                except Exception:
                    break

            # Find the minimizer of the quadratic model within the trust region
            # min c0 + c1^T d + d^T C d  s.t. ||d|| <= rho
            # First try unconstrained minimizer: d* = -C^{-1} c1 (if C is positive definite)
            # If C is not PD, use projected gradient or trust-region step

            # Regularize C for stability
            C_reg = C + 1e-8 * np.eye(n)

            # Try to solve C_reg d = -c1
            try:
                d_unconstrained = np.linalg.solve(C_reg, -c1)
            except np.linalg.LinAlgError:
                try:
                    d_unconstrained = np.linalg.lstsq(C_reg, -c1, rcond=None)[0]
                except Exception:
                    d_unconstrained = -c1 * 0.1

            # Check if unconstrained minimizer is within trust region
            norm_d = np.linalg.norm(d_unconstrained)
            if norm_d > rho:
                # Project to trust region boundary
                d_unconstrained = d_unconstrained * (rho / norm_d)

            # Compute the model value at the proposed step
            d_proposed = d_unconstrained
            x_proposed = np.clip(x_best + d_proposed, -5.0, 5.0)
            f_model_proposed = c0 + c1 @ d_proposed + d_proposed @ C @ d_proposed

            # Evaluate the actual function
            f_actual = safe_call(x_proposed)
            if f_actual is None:
                break

            # Compute actual reduction
            f_actual_reduction = f_best - f_actual
            f_model_reduction = f_best - f_model_proposed

            # Ratio of actual to predicted reduction
            if f_model_reduction > 1e-12:
                rho_ratio = f_actual_reduction / f_model_reduction
            else:
                rho_ratio = 0.0 if f_actual_reduction < 0 else 1.0

            # Update trust region and accept/reject step
            if rho_ratio > 0.75:
                # Excellent step
                x_best = x_proposed.copy()
                f_best = f_actual
                rho = min(rho * beta, rho_max)
            elif rho_ratio > 0.25:
                # Good step
                x_best = x_proposed.copy()
                f_best = f_actual
                # Keep rho same
            else:
                # Poor step, shrink trust region
                rho = max(rho * gamma, rho_min)

            # Add the new point to the interpolation set
            points.append(x_proposed.copy())
            fvals.append(f_actual)

            # Keep the interpolation set size manageable
            # If too many points, remove the least useful ones
            max_points = 2 * total_unknowns
            if len(points) > max_points:
                # Remove points that are farthest from the current best
                # or have the highest function values
                # Simple strategy: keep the best max_points points
                combined = list(zip(points, fvals))
                combined.sort(key=lambda x: x[1])
                points = [p for p, f in combined[:max_points]]
                fvals = [f for p, f in combined[:max_points]]

            # Check for convergence
            if rho < rho_min * 1.1:
                # Trust region is very small, try a random restart in the neighborhood
                # or break
                pass

            # Check stagnation: if no improvement in last several iterations
            # (simplified: just continue)

        # If we have remaining budget, do a local polish with CMA-ES or simple ES
        remaining = total_budget - calls
        if remaining > 10 and time.time() < deadline - 10:
            self._polish(func, x_best, f_best, remaining, n, start_time, deadline)

        return x_best, f_best

    def _fallback_es(self, func, x_best, f_best, remaining, n, start_time, deadline):
        """Simple (1+1)-ES fallback."""
        rng = np.random.default_rng(0)
        sigma = 1.0
        calls_used = 0
        for _ in range(remaining):
            if time.time() > deadline:
                break
            y = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
            try:
                fy = func(y)
                calls_used += 1
            except Exception:
                break
            if fy < f_best:
                x_best, f_best = y.copy(), fy
                sigma *= 1.2
            else:
                sigma *= 0.9
        return x_best, f_best

    def _polish(self, func, x_best, f_best, remaining, n, start_time, deadline):
        """Local polish using a small population CMA-ES-like approach."""
        rng = np.random.default_rng(123)
        pop_size = min(10, remaining // 2)
        if pop_size < 2:
            return

        # Initialize population around best point
        sigma = 0.5
        pop = [x_best + sigma * rng.standard_normal(n) for _ in range(pop_size)]
        pop = [np.clip(p, -5, 5) for p in pop]

        for gen in range(remaining // pop_size):
            if time.time() > deadline:
                break
            # Evaluate population
            evals = []
            for p in pop:
                try:
                    fp = func(p)
                    evals.append(fp)
                except Exception:
                    evals.append(np.inf)

            # Find best in population
            idx_best = np.argmin(evals)
            if evals[idx_best] < f_best:
                x_best = pop[idx_best].copy()
                f_best = evals[idx_best]

            # Update population: keep best, replace others
            pop = [x_best.copy()]
            for _ in range(pop_size - 1):
                new_p = x_best + sigma * rng.standard_normal(n)
                pop.append(np.clip(new_p, -5, 5))

            # Adapt sigma
            sigma *= 0.95

        return x_best, f_best
# EVOLVE-BLOCK-END
