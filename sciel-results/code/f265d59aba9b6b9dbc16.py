import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Initial point
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        
        history = [(x, fx)]
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Pre-compute feature indices for quadratic terms
        quad_indices = []
        for i in range(n):
            for j in range(i, n):
                quad_indices.append((i, j))
        
        def get_quadratic_model(points):
            X = np.array([p[0] for p in points])
            Y = np.array([p[1] for p in points])
            m = len(X)
            
            num_quad = len(quad_indices)
            num_params = 1 + n + num_quad
            
            A = np.zeros((m, num_params))
            A[:, 0] = 1.0
            A[:, 1:1+n] = X
            
            idx = 1 + n
            for i, j in quad_indices:
                A[:, idx] = X[:, i] * X[:, j]
                idx += 1
            
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
            except np.linalg.LinAlgError:
                return None, None

            c = coeffs[0]
            g = coeffs[1:1+n].copy()
            H_flat = coeffs[1+n:]
            
            H = np.zeros((n, n))
            idx = 0
            for i, j in quad_indices:
                val = H_flat[idx]
                H[i, j] = val
                if i != j:
                    H[j, i] = val
                idx += 1
                
            # Regularize Hessian to be PD
            try:
                eigvals = np.linalg.eigvalsh(H)
                min_eig = np.min(eigvals)
                if min_eig < 1e-6:
                    H += (1e-5 - min_eig) * np.eye(n)
            except np.linalg.LinAlgError:
                H += 1e-5 * np.eye(n)
            
            try:
                x_min = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                return None, None
                
            return x_min, c

        stagnation_counter = 0
        
        while calls < budget:
            # Use the best 30 points for fitting
            history_sorted = sorted(history, key=lambda p: p[1])
            fit_points = history_sorted[:min(30, len(history))]
            
            if len(fit_points) < 2*n + 1:
                # Random walk
                step = 1.0 * rng.standard_normal(n)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_x, best_f = x, fx
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
                history.append((x, fx))
                if len(history) > 100:
                    history = history_sorted[:100]
                continue
            
            x_min, c = get_quadratic_model(fit_points)
            
            if x_min is None:
                # Fallback
                step = 0.5 * rng.standard_normal(n)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_x, best_f = x, fx
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
                history.append((x, fx))
                if len(history) > 100:
                    history = history_sorted[:100]
                continue
                
            x_proposed = np.clip(x_min, -5, 5)
            
            # If the proposed step is very small, it might be a local min of the model
            # but we might be stuck.
            if np.linalg.norm(x_proposed - x) < 1e-4:
                # Random jitter
                step = 0.1 * rng.standard_normal(n)
                y = np.clip(x + step, -5, 5)
            else:
                y = x_proposed
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_x, best_f = x, fx
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                history.append((x, fx))
                if len(history) > 100:
                    history = history_sorted[:100]
            else:
                # Try a smaller step towards x_min or random
                step = 0.1 * rng.standard_normal(n)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_x, best_f = x, fx
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                    history.append((x, fx))
                    if len(history) > 100:
                        history = history_sorted[:100]
                else:
                    stagnation_counter += 1
                
            # Stagnation check
            if stagnation_counter > 30:
                # Restart from best with perturbation
                x = best_x.copy()
                x = np.clip(x + 1.0 * rng.standard_normal(n), -5, 5)
                fx = func(x)
                calls += 1
                history.append((x, fx))
                stagnation_counter = 0
                if len(history) > 100:
                    history_sorted = sorted(history, key=lambda p: p[1])
                    history = history_sorted[:100]

        return best_x, best_f
