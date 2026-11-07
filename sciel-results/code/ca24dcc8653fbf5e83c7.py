import numpy as np
import time


class Optimizer:
    """Derivative-free trust-region optimizer with quadratic model fitting."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        def safe_call(x):
            try:
                val = func(x)
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                return np.inf

        rng = np.random.default_rng(123)

        # Initial point
        x = rng.uniform(-4, 4, n)
        fx = safe_call(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Trust region radius
        trust_radius = 1.0
        min_trust = 1e-4
        max_trust = 8.0

        # History for quadratic model
        history_x = [x.copy()]
        history_f = [fx]

        # Stagnation tracking
        no_improve_count = 0
        stagnation_window = 30

        for iteration in range(budget - 1):
            # Time guard
            if time.time() > deadline - 5.0:
                break
            if calls >= budget:
                break

            # Build quadratic model from history
            # Use last min(15, len(history)) points
            k = min(15, len(history_x))
            X = np.array(history_x[-k:])
            F = np.array(history_f[-k:])
            center = X[0]

            # Center the data
            D = X - center
            # Quadratic model: f(x) ≈ f0 + g^T (x-c) + 0.5 (x-c)^T H (x-c)
            # Features: [1, d_1, ..., d_n, d_1^2, d_1*d_2, ..., d_n^2]
            # But we'll use a simpler parameterization: f = a + b^T d + d^T C d
            # where C is symmetric. We fit a, b, C via least squares.
            # Number of parameters: 1 + n + n*(n+1)/2
            n_params = 1 + n + n * (n + 1) // 2

            if k >= n + 1:
                # Build feature matrix
                # Features: [1, d_1, ..., d_n, d_1^2, d_1*d_2, ..., d_n^2]
                # For symmetric C, we use: sum_{i<=j} C_{ij} d_i d_j
                # with C_{ij} = C_{ji}, and diagonal terms C_{ii} d_i^2
                # Off-diagonal: 2*C_{ij} d_i d_j for i != j
                # Let's use the standard: f = a + b^T d + 0.5 d^T H d
                # Parameters: a (1), b (n), H upper-triangular (n*(n+1)/2)
                # Feature for H_{ij} (i<=j): d_i * d_j, but for i!=j the coefficient is H_{ij} and it appears as 2*H_{ij}*d_i*d_j in 0.5*d^T*H*d
                # So: 0.5*d^T*H*d = sum_i 0.5*H_{ii}*d_i^2 + sum_{i<j} H_{ij}*d_i*d_j
                # Let's build the feature matrix properly.

                # Indices for upper triangular H
                h_indices = [(i, j) for i in range(n) for j in range(i, n)]
                n_h = len(h_indices)

                # Feature matrix: columns are [1, d_1..d_n, d_i*d_j for (i,j) in h_indices]
                # For the H part: 0.5*d^T*H*d = sum_{(i,j) in h_indices} coeff_{ij} * feature_{ij}
                # where feature_{ii} = d_i^2, coeff = 0.5*H_{ii}
                #       feature_{ij} = d_i*d_j (i<j), coeff = H_{ij}
                # So we can write: 0.5*d^T*H*d = sum_{(i,j)} c_{ij} * phi_{ij}(d)
                # where phi_{ii} = d_i^2, phi_{ij} = d_i*d_j for i<j
                # and c_{ii} = 0.5*H_{ii}, c_{ij} = H_{ij} for i<j

                A = np.zeros((k, 1 + n + n_h))
                A[:, 0] = 1.0
                A[:, 1:1+n] = D
                for idx, (i, j) in enumerate(h_indices):
                    A[:, 1 + n + idx] = D[:, i] * D[:, j]

                # Target: F - F[0] (relative to center)
                y = F - F[0]

                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = np.zeros(1 + n + n_h)

                # Extract model parameters
                a = coeffs[0]  # should be ~0 since we centered
                b = coeffs[1:1+n]  # gradient
                # Reconstruct H from coeffs[1+n:]
                H = np.zeros((n, n))
                for idx, (i, j) in enumerate(h_indices):
                    c = coeffs[1 + n + idx]
                    if i == j:
                        H[i, i] = 2.0 * c  # since 0.5*H_{ii}*d_i^2 = c*d_i^2 => H_{ii} = 2c
                    else:
                        H[i, j] = c
                        H[j, i] = c

                # Eigen-decompose H to find directions of low curvature
                try:
                    eigvals, eigvecs = np.linalg.eigh(H)
                except np.linalg.LinAlgError:
                    eigvals = np.ones(n)
                    eigvecs = np.eye(n)

                # Find the direction of minimum curvature (largest step possible)
                # But we want to minimize, so we want to go in the direction that reduces f
                # The steepest descent direction in the model is -H^{-1} b if H is positive definite
                # Otherwise, use -b (gradient descent)

                # Check if H is positive definite
                if np.all(eigvals > 1e-8):
                    # Newton step: -H^{-1} b
                    try:
                        step_dir = np.linalg.solve(H, -b)
                    except np.linalg.LinAlgError:
                        step_dir = -b
                else:
                    # Use gradient descent direction
                    step_dir = -b

                # Normalize step direction
                step_norm = np.linalg.norm(step_dir)
                if step_norm > 1e-12:
                    step_dir = step_dir / step_norm
                else:
                    step_dir = rng.standard_normal(n)
                    step_dir /= np.linalg.norm(step_dir)

                # Scale step by trust region, but adapt based on curvature
                # If curvature is small along this direction, allow larger step
                curv_along_dir = float(step_dir @ H @ step_dir)
                if curv_along_dir < 1e-6:
                    # Very flat direction, can take larger step
                    effective_radius = min(trust_radius * 3.0, max_trust)
                elif curv_along_dir < 0.1:
                    effective_radius = trust_radius * 1.5
                else:
                    effective_radius = trust_radius

                step = effective_radius * step_dir
                # Clip to trust region
                if np.linalg.norm(step) > effective_radius:
                    step = step * (effective_radius / np.linalg.norm(step))

                # Also clip to box
                y = np.clip(x + step, -5, 5)

            else:
                # Not enough points, use random direction
                step_dir = rng.standard_normal(n)
                step_dir /= np.linalg.norm(step_dir)
                y = np.clip(x + trust_radius * step_dir, -5, 5)

            # Evaluate
            fy = safe_call(y)
            calls += 1

            # Update trust region based on improvement
            if fy < fx:
                # Successful step
                rho = (fx - fy) / max(np.linalg.norm(step), 1e-12)
                if rho > 0.5:
                    trust_radius = min(trust_radius * 1.5, max_trust)
                elif rho > 0.1:
                    trust_radius = min(trust_radius * 1.2, max_trust)
                else:
                    trust_radius = max(trust_radius * 0.8, min_trust)

                x = y
                fx = fy
                no_improve_count = 0

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

                # Add to history
                history_x.append(x.copy())
                history_f.append(fx)
                if len(history_x) > 20:
                    history_x.pop(0)
                    history_f.pop(0)
            else:
                # Unsuccessful step, shrink trust region
                trust_radius = max(trust_radius * 0.5, min_trust)
                no_improve_count += 1

                # Stagnation restart
                if no_improve_count >= stagnation_window:
                    # Restart from best point with new random perturbation
                    x = best_x + rng.standard_normal(n) * 0.5
                    x = np.clip(x, -5, 5)
                    fx = safe_call(x)
                    calls += 1
                    if calls >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    history_x = [x.copy()]
                    history_f = [fx]
                    trust_radius = 1.0
                    no_improve_count = 0

                    # Try a few random restarts if still stuck
                    for _ in range(3):
                        if calls >= budget or time.time() > deadline - 5.0:
                            break
                        xr = best_x + rng.standard_normal(n) * rng.uniform(0.5, 2.0)
                        xr = np.clip(xr, -5, 5)
                        fr = safe_call(xr)
                        calls += 1
                        if fr < fx:
                            x, fx = xr, fr
                        if fr < best_f:
                            best_f = fr
                            best_x = xr.copy()
                        history_x.append(x.copy())
                        history_f.append(fx)
                        if len(history_x) > 20:
                            history_x.pop(0)
                            history_f.pop(0)

        return best_x, best_f
