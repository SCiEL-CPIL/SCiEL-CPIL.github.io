import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        X = []
        Y = []

        def wrapped_func(x):
            nonlocal calls, best_f, best_x, X, Y
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                val = best_f
            if not np.isfinite(val):
                val = best_f
            X.append(np.array(x, dtype=float))
            Y.append(val)
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling to build a surrogate model
        # Sample a diverse set of points to understand the landscape structure
        n_init = 60
        # Mix of random, grid-like, and structured points
        init_points = []
        # Random points
        init_points.append(rng.uniform(-5, 5, (40, dim)))
        # Grid-like points (coarse)
        for i in range(10):
            for j in range(10):
                pt = np.zeros(dim)
                pt[0] = -5 + 10 * i / 9.0
                pt[1] = -5 + 10 * j / 9.0
                if dim > 2:
                    for k in range(2, dim):
                        pt[k] = rng.uniform(-5, 5)
                init_points.append(pt.reshape(1, -1))
        # Origin and structured
        init_points.append(np.zeros((1, dim)))
        init_points.append(np.ones((1, dim)) * 2.0)
        init_points.append(np.ones((1, dim)) * -2.0)
        
        init_points = np.vstack(init_points)
        # Shuffle and take first n_init
        idx = rng.permutation(len(init_points))
        init_points = init_points[idx[:n_init]]
        
        for x in init_points:
            if calls >= budget - 20 or time.time() > deadline - 10:
                break
            wrapped_func(x)

        # Phase 2: Bayesian Optimization loop with GP surrogate
        # We use a simple GP implementation with RBF kernel
        # To keep it fast, we limit the number of points used in the GP to the most recent or best ones
        # Since dim=5 is small, GP on 500 points is feasible but might be slow if not careful.
        # We will use a subset of points for the GP to keep it fast, or use a simplified acquisition.
        
        # Actually, with 500 budget, a full GP on all points is O(n^3) which is 500^3 = 125M ops per iteration, too slow.
        # We need to be careful.
        # Alternative: Use a simpler model or limit the GP size.
        # Let's use a GP with a maximum of 100 points (the best and most recent).
        
        max_gp_points = 100
        alpha = 1e-6
        beta = 1e-6
        
        def gp_predict(X_train, Y_train, X_pred, length_scale=1.0, sigma_f=1.0, noise=1e-8):
            """Simple GP prediction with RBF kernel."""
            n_train = len(X_train)
            n_pred = len(X_pred)
            
            # Kernel matrix
            def rbf(X1, X2, ls):
                # Compute squared distances
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-sq_dists / (2 * ls**2))
            
            K = rbf(X_train, X_train, length_scale) + noise * np.eye(n_train)
            Ks = rbf(X_train, X_pred, length_scale)
            
            # Solve K * alpha = Y
            try:
                alpha_vec = np.linalg.solve(K, Y_train)
            except np.linalg.LinAlgError:
                # Fallback to least squares if singular
                alpha_vec = np.linalg.lstsq(K, Y_train, rcond=None)[0]
            
            # Predict mean and variance
            mu = Ks.T @ alpha_vec
            v = np.linalg.solve(K, Ks)
            K_ss = rbf(X_pred, X_pred, length_scale)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mu, var

        def expected_improvement(mu, var, best_y, xi=0.01):
            """Expected Improvement acquisition function."""
            sigma = np.sqrt(var)
            imp = best_y - mu - xi
            z = imp / sigma
            # Standard normal CDF and PDF
            cdf = 0.5 * (1 + np.vectorize(lambda x: np.math.erf(x / np.sqrt(2)))(z))
            pdf = np.vectorize(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi))(z / np.sqrt(2)) # Wait, z is already standardized?
            # Standard formula: EI = (mu - best_y) * Phi(z) + sigma * phi(z) where z = (mu - best_y) / sigma
            # Here we want to minimize, so EI = (best_y - mu) * Phi(-z) + sigma * phi(z)
            # Let's use the standard minimization EI:
            # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
            z = imp / sigma
            cdf_z = 0.5 * (1 + np.vectorize(lambda x: np.math.erf(x / np.sqrt(2)))(z))
            pdf_z = np.vectorize(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi))(z)
            ei = imp * cdf_z + sigma * pdf_z
            return ei

        # Main BO loop
        while calls < budget - 10 and time.time() < deadline - 5:
            # Select points for GP: best 50% and most recent 50%
            n_current = len(X)
            if n_current < 10:
                # Not enough data, just sample randomly
                x_next = rng.uniform(-5, 5, dim)
                wrapped_func(x_next)
                continue
            
            # Sort by Y
            sorted_idx = np.argsort(Y)
            n_best = max(10, n_current // 2)
            best_idx = sorted_idx[:n_best]
            recent_idx = np.arange(n_current - n_best, n_current)
            gp_idx = np.unique(np.concatenate([best_idx, recent_idx]))
            
            X_gp = np.array(X)[gp_idx]
            Y_gp = np.array(Y)[gp_idx]
            
            # Normalize Y for better GP performance
            y_mean = np.mean(Y_gp)
            y_std = np.std(Y_gp) + 1e-8
            Y_gp_norm = (Y_gp - y_mean) / y_std
            
            # Sample candidate points
            n_candidates = 200
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Predict on candidates
            try:
                mu, var = gp_predict(X_gp, Y_gp_norm, candidates, length_scale=1.5, sigma_f=1.0, noise=1e-8)
            except Exception:
                # Fallback to random if GP fails
                x_next = rng.uniform(-5, 5, dim)
                wrapped_func(x_next)
                continue
            
            # Compute EI
            best_y_norm = np.min(Y_gp_norm)
            ei = expected_improvement(mu, var, best_y_norm, xi=0.01)
            
            # Select best candidate
            best_candidate_idx = np.argmax(ei)
            x_next = candidates[best_candidate_idx]
            
            # Evaluate
            wrapped_func(x_next)
            
            # Check for time
            if time.time() > deadline - 20:
                break

        # Phase 3: Local polish with Nelder-Mead from best point
        if calls < budget - 20 and time.time() < deadline - 10:
            # Simple Nelder-Mead implementation
            x0 = best_x.copy()
            # Initial simplex
            simplex = [x0]
            for i in range(dim):
                perturbed = x0.copy()
                perturbed[i] += 0.5 if x0[i] < 4.5 else -0.5
                simplex.append(perturbed)
            
            simplex = np.array(simplex)
            f_simplex = np.array([wrapped_func(x) for x in simplex])
            
            max_iter = 50
            for _ in range(max_iter):
                if calls >= budget - 10 or time.time() > deadline - 5:
                    break
                
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                # Check convergence
                if np.max(f_simplex) - np.min(f_simplex) < 1e-8:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                f_reflect = wrapped_func(x_reflect)
                
                if f_reflect < f_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    x_expand = centroid + gamma * (centroid - simplex[-1])
                    f_expand = wrapped_func(x_expand)
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_simplex[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_simplex[-1] = f_reflect
                elif f_reflect < f_simplex[-2]:
                    simplex[-1] = x_reflect
                    f_simplex[-1] = f_reflect
                else:
                    # Contraction
                    rho = 0.5
                    x_contract = centroid + rho * (simplex[-1] - centroid)
                    f_contract = wrapped_func(x_contract)
                    if f_contract < f_simplex[-1]:
                        simplex[-1] = x_contract
                        f_simplex[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = wrapped_func(simplex[i])
                
                # Update best
                if f_simplex[0] < best_f:
                    best_f = f_simplex[0]
                    best_x = simplex[0].copy()

        return best_x, best_f
