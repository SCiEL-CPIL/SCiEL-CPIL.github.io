import numpy as np
import time

class Optimizer:
    """Quadratic surrogate + Hooke-Jeeves refinement for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')
        history_x = []
        history_f = []

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            history_x.append(x.copy())
            history_f.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def fit_quadratic(points_x, points_f):
            """Fit a quadratic model and return the predicted minimizer."""
            n = len(points_x)
            if n < d + 1:
                return None
            
            # Build quadratic model: f(x) = c + g^T x + 0.5 x^T H x
            # We need at least d*(d+3)/2 + 1 points for a full quadratic
            # Use least squares if we have enough points
            
            num_params = 1 + d + d * (d + 1) // 2  # constant + linear + quadratic (symmetric)
            
            if n < num_params:
                # Not enough points for full quadratic, use linear model
                # f(x) = c + g^T x
                X = np.ones((n, d + 1))
                X[:, 1:] = points_x
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(X, points_f, rcond=None)
                    c = coeffs[0]
                    g = coeffs[1:]
                    # Minimize linear: no interior minimum unless g=0
                    # Just return current best
                    return None
                except:
                    return None
            
            # Full quadratic: parameterize symmetric H
            # f(x) = c + g^T x + 0.5 * sum_{i<=j} H_ij * x_i * x_j
            # Build design matrix
            def quad_basis(x):
                """Return [1, x_1, ..., x_d, x_1^2, x_1*x_2, ..., x_d^2]"""
                basis = [1.0]
                basis.extend(x)
                for i in range(d):
                    for j in range(i, d):
                        basis.append(x[i] * x[j])
                return basis
            
            A = np.array([quad_basis(x) for x in points_x])
            b = np.array(points_f)
            
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                return None
            
            # Extract c, g, H
            c = coeffs[0]
            g = coeffs[1:1+d].copy()
            # Quadratic terms: for i<=j, coeff is at index 1+d + (i*(2d-i+1)//2 + j - i)
            # Simpler: reconstruct H from coeffs
            H = np.zeros((d, d))
            idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    val = coeffs[idx]
                    H[i, j] = val
                    H[j, i] = val
                    idx += 1
            
            # Find minimum: solve (H) x = -g
            # f(x) = c + g^T x + 0.5 x^T H x
            # grad = g + H x = 0 => x = -H^{-1} g
            try:
                if np.linalg.cond(H) < 1e10:
                    x_opt = np.linalg.solve(H, -g)
                    # Check if H is positive definite
                    eigvals = np.linalg.eigvalsh(H)
                    if np.all(eigvals > 1e-10):
                        x_opt = np.clip(x_opt, -5, 5)
                        return x_opt
            except:
                pass
            
            return None

        def hooke_jees(start_x, max_iter=500):
            nonlocal eval_count
            if eval_count >= budget:
                return

            x = np.clip(start_x, -5, 5).copy()
            f = eval_func(x)
            if eval_count >= budget:
                return

            step_sizes = np.ones(d) * 0.3
            min_step = 1e-6

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                moved = False
                for i in range(d):
                    if eval_count >= budget or time.time() > deadline:
                        break

                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    f_plus = eval_func(x_plus)
                    if eval_count >= budget:
                        break

                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        step_sizes[i] *= 2.0
                        moved = True
                    else:
                        step_sizes[i] *= 0.5

                        x_minus = x.copy()
                        x_minus[i] -= step_sizes[i]
                        f_minus = eval_func(x_minus)
                        if eval_count >= budget:
                            break

                        if f_minus < f:
                            x = x_minus
                            f = f_minus
                            step_sizes[i] *= 2.0
                            moved = True
                        else:
                            step_sizes[i] *= 0.5

                if np.all(step_sizes < min_step):
                    break

        # Phase 1: Initial exploration with well-spaced points
        # Use a small set of points to get a sense of the landscape
        rng = np.random.default_rng(42)
        
        # Start with origin and a few strategic points
        init_points = [np.zeros(d)]
        # Add points along each axis
        for i in range(d):
            p = np.zeros(d)
            p[i] = 2.0
            init_points.append(p.copy())
        
        for pt in init_points:
            if eval_count >= budget - 50:
                break
            eval_func(pt)
        
        # Phase 2: Use quadratic surrogate to predict minimum
        if len(history_x) >= 6 and time.time() < deadline:
            # Use recent points for the fit
            n_fit = min(len(history_x), 20)
            recent_x = np.array(history_x[-n_fit:])
            recent_f = np.array(history_f[-n_fit:])
            
            x_pred = fit_quadratic(recent_x, recent_f)
            if x_pred is not None:
                # Evaluate predicted minimum
                f_pred = eval_func(x_pred)
                # Refine with Hooke-Jeeves from predicted point
                if eval_count < budget - 30:
                    hooke_jees(x_pred)
        
        # Phase 3: If budget remains, do more exploration + surrogate cycles
        while eval_count < budget - 20 and time.time() < deadline:
            # Try a random point near current best
            if eval_count < budget - 10:
                x_rand = best_x + rng.normal(0, 0.5, d)
                x_rand = np.clip(x_rand, -5, 5)
                eval_func(x_rand)
            
            # Try surrogate again with more data
            if len(history_x) >= 8 and time.time() < deadline:
                n_fit = min(len(history_x), 25)
                recent_x = np.array(history_x[-n_fit:])
                recent_f = np.array(history_f[-n_fit:])
                
                x_pred = fit_quadratic(recent_x, recent_f)
                if x_pred is not None:
                    f_pred = eval_func(x_pred)
                    if eval_count < budget - 20:
                        hooke_jees(x_pred, max_iter=300)
            
            # Check if we're making progress
            if eval_count >= budget - 10:
                break

        return best_x, best_f
