import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Helper: Fit a quadratic model to points and find its minimum ---
        def fit_quadratic_and_minimize(points, values):
            """
            Given 2n+1 points (n+1 around a center), fit a quadratic model.
            Returns the predicted minimum point.
            """
            m = len(points)
            # We need at least 2n+1 points for a full quadratic in n dims
            if m < 2 * n + 1:
                return None
            
            # Use the first point as the local origin for numerical stability
            x0 = points[0]
            
            # Build the design matrix for the quadratic model
            # f(x) = c + b^T (x-x0) + (x-x0)^T Q (x-x0)
            # We solve for b and Q (symmetric)
            # Number of parameters: 1 (c) + n (b) + n*(n+1)/2 (Q) = 1 + n + n*(n+1)/2
            
            # Actually, let's use a simpler approach for small n:
            # Use finite differences to estimate gradient and Hessian at x0
            # Or use least squares with a full quadratic basis
            
            # Let's use the standard approach: evaluate at x0 and x0 +/- h_i e_i
            # But we already have points. Let's construct the system.
            
            # For n=5, we can use scipy or manual linear algebra.
            # Let's try to fit using the points directly.
            
            # Construct basis functions for each point relative to x0
            # phi(x) = [1, x1-x0_1, ..., xn-x0_n, (x1-x0_1)^2, (x1-x0_1)(x2-x0_2), ..., (xn-x0_n)^2]
            # Number of terms: 1 + n + n(n+1)/2
            
            def basis(x):
                d = x - x0
                terms = [1.0]
                terms.extend(d)
                for i in range(n):
                    for j in range(i, n):
                        terms.append(d[i] * d[j])
                return np.array(terms)
            
            # Build A matrix and y vector
            A = np.array([basis(p) for p in points])
            y = np.array(values)
            
            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
            except:
                return None
            
            # Reconstruct quadratic form
            # coeffs[0] = c
            # coeffs[1:n+1] = b
            # coeffs[n+1:] = Q terms (upper triangular)
            
            b = coeffs[1:n+1]
            
            # Build Q matrix from coeffs
            Q = np.zeros((n, n))
            idx = n + 1
            for i in range(n):
                for j in range(i, n):
                    val = coeffs[idx]
                    if i == j:
                        Q[i, i] = val
                    else:
                        Q[i, j] = val
                        Q[j, i] = val
                    idx += 1
            
            # Find minimum: solve (2Q) dx = -b
            # dx = -0.5 * Q^{-1} b
            try:
                if np.linalg.matrix_rank(Q) < n:
                    # Singular, use pseudo-inverse
                    Q_inv = np.linalg.pinv(Q)
                else:
                    Q_inv = np.linalg.inv(Q)
                
                dx = -0.5 * (Q_inv @ b)
                x_min = x0 + dx
                return x_min
            except:
                return None

        # --- Phase 1: Global Exploration with (1+lambda)-ES and Paraboloid Jumps ---
        
        # Initial points: spread out in the box
        np.random.seed(42)
        
        # Start with a few diverse points
        init_points = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.random.uniform(-4, 4, n),
            np.random.uniform(-4, 4, n),
        ]
        
        for p in init_points:
            if calls_made >= total_budget:
                break
            evaluate(p)
        
        # Main loop: iterate until budget is nearly exhausted
        # Strategy:
        # 1. Fit a local quadratic model around the current best point using 2n+1 points
        # 2. Jump to the predicted minimum
        # 3. If the jump doesn't improve significantly, do a small local step
        
        # Reserve some budget for final polish
        polish_budget = 30
        main_budget_limit = total_budget - polish_budget
        
        # Current search center
        center = best_x.copy() if best_x is not None else np.zeros(n)
        step_size = 1.0
        
        while calls_made < main_budget_limit:
            if time.time() > deadline:
                break
            
            # Try to fit a quadratic model around the current best point
            # We need 2n+1 points: center and +/- step_size * e_i
            
            points = []
            values = []
            
            # Center
            if calls_made < main_budget_limit:
                f_center = evaluate(center)
                points.append(center.copy())
                values.append(f_center)
            
            # Perturbations
            for i in range(n):
                if calls_made >= main_budget_limit:
                    break
                
                # Positive
                x_pos = center.copy()
                x_pos[i] += step_size
                x_pos = np.clip(x_pos, -5, 5)
                f_pos = evaluate(x_pos)
                points.append(x_pos.copy())
                values.append(f_pos)
                
                if calls_made >= main_budget_limit:
                    break
                
                # Negative
                x_neg = center.copy()
                x_neg[i] -= step_size
                x_neg = np.clip(x_neg, -5, 5)
                f_neg = evaluate(x_neg)
                points.append(x_neg.copy())
                values.append(f_neg)
            
            # Fit quadratic and find predicted minimum
            if len(points) >= 2 * n + 1:
                x_pred = fit_quadratic_and_minimize(points, values)
                
                if x_pred is not None:
                    # Clip to box
                    x_pred = np.clip(x_pred, -5, 5)
                    
                    # Evaluate the predicted minimum
                    if calls_made < main_budget_limit:
                        f_pred = evaluate(x_pred)
                        
                        # If it improved, move center there and reduce step size
                        if f_pred < best_f:
                            center = x_pred.copy()
                            step_size *= 0.5
                        else:
                            # If not improved, try a smaller step or random restart
                            step_size *= 0.7
                            
                            # If step size is too small, try a random jump
                            if step_size < 0.01:
                                center = np.random.uniform(-4, 4, n)
                                step_size = 1.0
            else:
                # Not enough points, just do a random step
                center = np.random.uniform(-4, 4, n)
                step_size = 1.0
        
        # --- Phase 2: Local Polish ---
        if best_x is not None and calls_made < total_budget:
            # Use a finer grid around the best point
            polish_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
            
            for step in polish_steps:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                improved = False
                for dim_idx in range(n):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    
                    # Try positive
                    x_pos = best_x.copy()
                    x_pos[dim_idx] += step
                    x_pos = np.clip(x_pos, -5, 5)
                    f_pos = evaluate(x_pos)
                    if f_pos < best_f:
                        improved = True
                    
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    
                    # Try negative
                    x_neg = best_x.copy()
                    x_neg[dim_idx] -= step
                    x_neg = np.clip(x_neg, -5, 5)
                    f_neg = evaluate(x_neg)
                    if f_neg < best_f:
                        improved = True
                
                if not improved and step < 0.01:
                    break
        
        # Final check
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
