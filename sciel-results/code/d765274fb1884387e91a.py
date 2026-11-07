"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Halton exploration + Sequential Quadratic Approximation local search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        n = self.dim
        budget = self.budget

        def safe_call(x):
            try:
                val = func(np.asarray(x, dtype=float))
                return float(val)
            except Exception:
                return float('inf')

        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim):
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            pt = np.zeros(dim)
            for d in range(dim):
                pt[d] = -5.0 + 10.0 * halton(index + 1, primes[d % len(primes)])
            return pt

        # Phase 1: Deterministic Halton sequence for uniform space-filling
        best_x = None
        best_f = float('inf')
        calls = 0
        eval_history = []  # list of (x, f) for quadratic approximation

        # Evaluate first ~150 points using Halton sequence
        n_halton = min(150, budget // 3)
        for idx in range(n_halton):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = halton_point(idx, n)
            f = safe_call(pt)
            calls += 1
            eval_history.append((pt.copy(), f))
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: Sequential Quadratic Approximation (SQP) local search
        # Fit a quadratic model to recent evaluations and step toward the minimum
        max_sqp_steps = budget - calls
        sqp_radius = 2.0
        no_improve_count = 0
        max_no_improve = 5

        while calls < budget and time.time() < deadline:
            if no_improve_count >= max_no_improve:
                # Reset radius and try from a new Halton point
                sqp_radius = 2.0
                no_improve_count = 0
                # Find a new starting point from Halton sequence
                for idx in range(n_halton, min(n_halton + 50, budget - calls)):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    pt = halton_point(idx, n)
                    f = safe_call(pt)
                    calls += 1
                    eval_history.append((pt.copy(), f))
                    if f < best_f:
                        best_f = f
                        best_x = pt.copy()
                continue

            # Use the most recent evaluations (up to 2*n+1 points) near current best
            recent = []
            for x, f in reversed(eval_history):
                if np.linalg.norm(x - best_x) < sqp_radius * 1.5:
                    recent.append((x, f))
                if len(recent) >= 2 * n + 1:
                    break

            if len(recent) < n + 1:
                # Not enough points, expand search
                sqp_radius *= 0.5
                no_improve_count += 1
                # Generate a new point in the direction of best gradient estimate
                if len(recent) >= 2:
                    # Simple gradient estimate from the two best points
                    recent_sorted = sorted(recent, key=lambda t: t[1])
                    if len(recent_sorted) >= 2 and recent_sorted[0][1] != recent_sorted[1][1]:
                        grad_est = (recent_sorted[0][0] - recent_sorted[1][0]) / max(1e-10, recent_sorted[1][1] - recent_sorted[0][1])
                        norm_grad = np.linalg.norm(grad_est)
                        if norm_grad > 1e-10:
                            step = -sqp_radius * 0.1 * grad_est / norm_grad
                            new_pt = best_x + step
                            new_pt = np.clip(new_pt, -5, 5)
                            f = safe_call(new_pt)
                            calls += 1
                            eval_history.append((new_pt.copy(), f))
                            if f < best_f:
                                best_f = f
                                best_x = new_pt.copy()
                                no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                continue

            # Fit quadratic model: f(x) ≈ f0 + g^T(x-x0) + 0.5*(x-x0)^T H (x-x0)
            # Use least squares to fit coefficients
            x0 = best_x
            X = np.array([x - x0 for x, f in recent])
            y = np.array([f - best_f for x, f in recent])
            
            # Build design matrix for quadratic terms
            m = len(X)
            A = np.zeros((m, n + n * (n + 1) // 2))
            for i in range(m):
                A[i, :n] = X[i]  # linear terms
                j = n
                for a in range(n):
                    for b in range(a, n):
                        A[i, j] = X[i, a] * X[i, b]
                        j += 1
            
            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
            except np.linalg.LinAlgError:
                no_improve_count += 1
                continue

            # Extract gradient and Hessian
            g = coeffs[:n]
            H_flat = coeffs[n:]
            H = np.zeros((n, n))
            j = 0
            for a in range(n):
                for b in range(a, n):
                    H[a, b] = H_flat[j]
                    H[b, a] = H_flat[j]
                    j += 1

            # Regularize Hessian for stability
            H += 1e-4 * np.eye(n)

            # Find minimum of quadratic model: solve (H) dx = -g
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                except:
                    no_improve_count += 1
                    continue

            # Limit step size
            step_norm = np.linalg.norm(dx)
            if step_norm > sqp_radius:
                dx = dx / step_norm * sqp_radius

            new_pt = best_x + dx
            new_pt = np.clip(new_pt, -5, 5)
            f = safe_call(new_pt)
            calls += 1
            eval_history.append((new_pt.copy(), f))

            if f < best_f - 1e-10:
                best_f = f
                best_x = new_pt.copy()
                no_improve_count = 0
                # Shrink radius slightly on success
                sqp_radius = max(0.1, sqp_radius * 0.8)
            else:
                no_improve_count += 1
                sqp_radius = max(0.1, sqp_radius * 0.7)

        return best_x, best_f
# EVOLVE-BLOCK-END
