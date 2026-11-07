"""Rewritten optimizer for BBOB f14 (Different powers)."""
import numpy as np
import time


class Optimizer:
    """Derivative-free sequential quadratic programming (DF-SQP) with
    trust-region subproblem, combined with a lightweight global restart
    strategy for robustness on shifted/rotated instances."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            fx = float(func(np.asarray(x, dtype=float)))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Global exploration with random restarts + local search
        # Use a mixture of broad random sampling and local refinement.

        rng = np.random.default_rng(42)

        # Initial diverse set of points
        n_init = 20
        points = []
        for _ in range(n_init):
            x = rng.uniform(-4.5, 4.5, n)
            fx = eval_func(x)
            points.append((x, fx))

        # Find best initial point
        points.sort(key=lambda t: t[1])
        x_cur = points[0][0].copy()
        f_cur = points[0][1]

        # Phase 2: DF-SQP local search from the best point
        # Build a local quadratic model using nearby evaluations

        def dfsqp_local(x0, f0, max_iter=100):
            """Local DF-SQP: build quadratic model from finite differences
            and solve trust-region subproblem iteratively."""
            nonlocal call_count, best_x, best_f, deadline
            x = x0.copy()
            fx = f0
            dim = n
            delta = 0.1  # finite difference step
            trust_radius = 1.0

            for it in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break

                # Build gradient and Hessian estimate via central differences
                # Use a sparse set of points to keep cost low
                grad = np.zeros(dim)
                hess = np.zeros((dim, dim))

                # First-order: central differences for gradient
                for i in range(dim):
                    epi = np.zeros(dim)
                    epi[i] = delta
                    xp = x + epi
                    xm = x - epi
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    fxp = eval_func(xp)
                    fxm = eval_func(xm)
                    if fxp == np.inf or fxm == np.inf:
                        break
                    grad[i] = (fxp - fxm) / (2 * delta)

                if call_count >= budget or time.time() > deadline:
                    break

                # Second-order: estimate Hessian from directional second derivatives
                # Use only diagonal and a few off-diagonal terms to save budget
                for i in range(dim):
                    epi = np.zeros(dim)
                    epi[i] = delta
                    xp2 = x + 2 * epi
                    xm2 = x - 2 * epi
                    xp2 = np.clip(xp2, -5, 5)
                    xm2 = np.clip(xm2, -5, 5)
                    fxp2 = eval_func(xp2)
                    fxm2 = eval_func(xm2)
                    if fxp2 == np.inf or fxm2 == np.inf:
                        break
                    # Second derivative: (f(x+2d) - 2f(x) + f(x-2d)) / (2d)^2
                    hess[i, i] = (fxp2 - 2 * fx + fxm2) / (4 * delta * delta)
                    if hess[i, i] < 0:
                        hess[i, i] = 0.01  # ensure positive semi-definite diagonal

                if call_count >= budget or time.time() > deadline:
                    break

                # Estimate off-diagonal Hessian terms for a few pairs
                for i in range(min(dim, 3)):
                    for j in range(i + 1, min(dim, 3)):
                        epi = np.zeros(dim)
                        epj = np.zeros(dim)
                        epi[i] = delta
                        epj[j] = delta
                        # f(x + epi + epj) - f(x + epi) - f(x + epj) + f(x)
                        xpp = np.clip(x + epi + epj, -5, 5)
                        xpi = np.clip(x + epi, -5, 5)
                        xpj = np.clip(x + epj, -5, 5)
                        fpp = eval_func(xpp)
                        fpi = eval_func(xpi)
                        fpj = eval_func(xpj)
                        if fpp == np.inf or fpi == np.inf or fpj == np.inf:
                            break
                        hess_ij = (fpp - fpi - fpj + fx) / (delta * delta)
                        hess[i, j] = hess_ij
                        hess[j, i] = hess_ij

                if call_count >= budget or time.time() > deadline:
                    break

                # Solve trust-region subproblem: min g^T s + 0.5 s^T H s
                # subject to ||s|| <= trust_radius
                # Use a simple gradient projection approach
                H = hess + np.eye(dim) * 1e-4  # ensure positive definite
                try:
                    # Solve (H + lambda*I) s = -g for appropriate lambda
                    # Start with small lambda
                    lam = 1e-6
                    s = None
                    for _ in range(30):
                        A = H + lam * np.eye(dim)
                        try:
                            s = np.linalg.solve(A, -grad)
                        except np.linalg.LinAlgError:
                            s = np.linalg.lstsq(A, -grad, rcond=None)[0]
                        if np.linalg.norm(s) <= trust_radius:
                            break
                        lam *= 2
                    if s is None:
                        s = -grad / (np.linalg.norm(grad) + 1e-10) * trust_radius
                    # Project to trust region
                    s_norm = np.linalg.norm(s)
                    if s_norm > trust_radius:
                        s = s / s_norm * trust_radius
                except Exception:
                    s = -grad / (np.linalg.norm(grad) + 1e-10) * trust_radius

                # Take step
                x_new = np.clip(x + s, -5, 5)
                f_new = eval_func(x_new)

                if f_new == np.inf:
                    trust_radius *= 0.5
                    continue

                # Update trust region
                predicted_reduction = -0.5 * (grad @ s + 0.5 * (s @ H @ s))
                actual_reduction = fx - f_new
                if actual_reduction > 0.1 * predicted_reduction:
                    x, fx = x_new, f_new
                    trust_radius = min(2 * trust_radius, 5.0)
                else:
                    trust_radius *= 0.5

                # Check convergence
                if np.linalg.norm(grad) < 1e-8:
                    break

            return x, fx

        # Run local search from best point
        x_cur, f_cur = dfsqp_local(x_cur, f_cur)

        # Phase 3: If we have budget left, do a few more restarts from
        # promising regions
        if call_count < budget:
            # Generate a few new random points near the current best
            for _ in range(5):
                if call_count >= budget or time.time() > deadline:
                    break
                x_new = x_cur + rng.normal(0, 0.5, n)
                x_new = np.clip(x_new, -5, 5)
                f_new = eval_func(x_new)
                if f_new < f_cur:
                    x_cur, f_cur = dfsqp_local(x_new, f_new, max_iter=50)

        return best_x, best_f
