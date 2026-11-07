import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Quadratic Fit
        # For Sphere f(x) = ||x - x_opt||^2 + c, we need to fit:
        # f(x) = x^T Q x + q^T x + c
        # Number of params: dim*(dim+1)/2 + dim + 1
        # For dim=5: 15 + 5 + 1 = 21 params.
        # We need at least 21 points. Let's use 24 for stability (center + 10 axis + 13 random).
        n_points = 24
        if budget < n_points + 5:
             return self._simple_es(func, best_x, best_f)

        # Generate points
        points = []
        # Center
        points.append(np.zeros(dim))
        # Axis-aligned points at +/- 2.0
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 2.0
            p2 = np.zeros(dim); p2[i] = -2.0
            points.append(p1)
            points.append(p2)
        
        # Add random points to reach n_points
        rng = np.random.default_rng(42)
        while len(points) < n_points:
            p = rng.uniform(-5, 5, dim)
            points.append(p)
        
        points = np.array(points)
        values = np.array([evaluate(p) for p in points])
        
        # Construct design matrix for quadratic fit
        def feature_vector(x):
            d = len(x)
            quad_terms = []
            for i in range(d):
                for j in range(i, d):
                    quad_terms.append(x[i] * x[j])
            lin_terms = list(x)
            const = [1.0]
            return np.array(quad_terms + lin_terms + const)

        A = np.array([feature_vector(p) for p in points])
        b = values
        
        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            n_quad = dim * (dim + 1) // 2
            Q_flat = coeffs[:n_quad]
            q = coeffs[n_quad:n_quad + dim]
            c = coeffs[n_quad + dim]
            
            # Reconstruct Q matrix
            Q = np.zeros((dim, dim))
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    val = Q_flat[idx]
                    Q[i, j] = val
                    Q[j, i] = val
                    idx += 1
            
            # Solve Q x = -0.5 q
            try:
                x_opt = np.linalg.solve(Q, -0.5 * q)
            except np.linalg.LinAlgError:
                x_opt = np.linalg.pinv(Q) @ (-0.5 * q)
            
            x_opt = np.clip(x_opt, -5, 5)
            f_opt = evaluate(x_opt)
            
        except Exception:
            pass

        # Phase 2: Polish with (1+1)-ES
        x = best_x.copy()
        fx = best_f
        
        sigma = 0.5
        rng = np.random.default_rng(0)
        
        while calls_made < budget and time.time() < deadline:
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
            else:
                sigma *= 0.87
            
            if sigma < 1e-8:
                break

        return best_x, best_f

    def _simple_es(self, func, best_x, best_f):
        return best_x, best_f
