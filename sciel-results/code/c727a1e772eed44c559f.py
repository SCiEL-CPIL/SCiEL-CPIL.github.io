"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global quadratic fit using a small number of strategically placed points."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Explicit deadline guard: 300s limit, use 295s for safety
        start_time = time.time()
        deadline = start_time + 295
        
        best_x = None
        best_fx = float('inf')
        calls = 0
        
        def evaluate(x):
            nonlocal best_x, best_fx, calls
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        
        # Strategy:
        # The function is a rotated, shifted, and offset sphere (quadratic).
        # f(x) = (x-c)^T A (x-c) + f0
        # We can fit the quadratic parameters exactly using a small number of points.
        # The quadratic form has dim*(dim+1)/2 coefficients for the Hessian (symmetric),
        # dim coefficients for the linear term, and 1 for the constant.
        # Total parameters: dim*(dim+1)/2 + dim + 1.
        # For dim=5: 15 + 5 + 1 = 21 parameters.
        # We need at least 21 points to fit the system. To be robust, we use more.
        # A structured set of points:
        # 1. Origin: (0, ..., 0)
        # 2. Axis points: +/- h * e_i for i=1..dim (2*dim points)
        # 3. Pairwise points: h * (e_i + e_j) for i<j (dim*(dim-1)/2 points)
        # Total: 1 + 2*dim + dim*(dim-1)/2 = 1 + 10 + 10 = 21 points.
        # This is exactly the number of parameters.
        
        # Let's generate the points:
        # p0 = 0
        # p_i = h * e_i, p_{dim+i} = -h * e_i
        # p_{2*dim + k} = h * (e_i + e_j) for i<j
        
        # We will evaluate these points.
        # Then we set up the linear system:
        # f(x) = x^T Q x + b^T x + c
        # where Q is symmetric.
        
        h = 1.0
        
        points = []
        points.append(np.zeros(dim))
        
        # Axis points
        for i in range(dim):
            p_plus = np.zeros(dim)
            p_plus[i] = h
            points.append(p_plus)
            
            p_minus = np.zeros(dim)
            p_minus[i] = -h
            points.append(p_minus)
        
        # Pairwise points
        for i in range(dim):
            for j in range(i+1, dim):
                p = np.zeros(dim)
                p[i] = h
                p[j] = h
                points.append(p)
        
        # Evaluate all these points
        f_values = []
        for p in points:
            if calls >= self.budget:
                break
            if time.time() >= deadline:
                break
            f_values.append(evaluate(p))
        
        # If we didn't get enough points, fallback to random search
        if len(f_values) < len(points):
            # Fallback: simple random search
            for _ in range(self.budget - calls):
                if time.time() >= deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
            return best_x, best_fx
        
        # Now fit the quadratic using finite differences approach for robustness
        # Instead of solving a large linear system, we can use the function values
        # to directly estimate the gradient and Hessian at the origin, then
        # solve for the minimizer.
        
        # For a quadratic f(x) = x^T Q x + b^T x + c:
        # f(0) = c
        # f(h e_i) = Q_ii h^2 + b_i h + c
        # f(-h e_i) = Q_ii h^2 - b_i h + c
        # f(h(e_i + e_j)) = Q_ii h^2 + Q_jj h^2 + 2 Q_ij h^2 + b_i h + b_j h + c
        
        c_val = f_values[0]  # f(0)
        
        # Extract Q_ii and b_i from axis points
        Q_diag = np.zeros(dim)
        b_vec = np.zeros(dim)
        
        for i in range(dim):
            f_plus = f_values[1 + 2*i]      # f(h e_i)
            f_minus = f_values[1 + 2*i + 1] # f(-h e_i)
            
            # f_plus + f_minus = 2 Q_ii h^2 + 2c
            # f_plus - f_minus = 2 b_i h
            Q_diag[i] = (f_plus + f_minus - 2*c_val) / (2*h**2)
            b_vec[i] = (f_plus - f_minus) / (2*h)
        
        # Extract Q_ij (i < j) from pairwise points
        Q_off_diag = np.zeros((dim, dim))
        idx = 1 + 2*dim  # start index for pairwise points
        
        for i in range(dim):
            for j in range(i+1, dim):
                f_pair = f_values[idx]  # f(h(e_i + e_j))
                idx += 1
                
                # f_pair = Q_ii h^2 + Q_jj h^2 + 2 Q_ij h^2 + b_i h + b_j h + c
                # => 2 Q_ij h^2 = f_pair - Q_ii h^2 - Q_jj h^2 - b_i h - b_j h - c
                Q_ij = (f_pair - Q_diag[i]*h**2 - Q_diag[j]*h**2 - b_vec[i]*h - b_vec[j]*h - c_val) / (2*h**2)
                Q_off_diag[i, j] = Q_ij
                Q_off_diag[j, i] = Q_ij
        
        # Assemble full Q matrix
        Q = np.diag(Q_diag) + Q_off_diag
        
        # Find the minimum: 2 Q x + b = 0 => x* = -0.5 Q^{-1} b
        # Use regularization for stability
        Q_reg = Q + 1e-6 * np.eye(dim)
        
        try:
            x_star = np.linalg.solve(Q_reg, -0.5 * b_vec)
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            try:
                x_star = np.linalg.lstsq(Q_reg, -0.5 * b_vec, rcond=None)[0]
            except:
                # Ultimate fallback: random search
                for _ in range(self.budget - calls):
                    if time.time() >= deadline:
                        break
                    x = rng.uniform(-5, 5, dim)
                    evaluate(x)
                return best_x, best_fx
        
        # Clip to box
        x_star = np.clip(x_star, -5, 5)
        
        # Evaluate the predicted minimum
        fx_star = evaluate(x_star)
        
        # Use remaining budget for a simple (1+1)-ES polish starting from x_star
        if calls < self.budget:
            x = x_star
            fx = fx_star
            sigma = 0.1
            for _ in range(self.budget - calls):
                if time.time() >= deadline:
                    break
                if calls >= self.budget:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                    
                if sigma < 1e-6:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
