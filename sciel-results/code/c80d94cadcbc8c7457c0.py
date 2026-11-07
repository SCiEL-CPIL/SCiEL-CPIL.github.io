"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global Quadratic Model Fitting Optimizer."""

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
            if time.time() >= deadline:
                return float('inf')
            if calls >= self.budget:
                return float('inf')
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        
        # Strategy:
        # 1. Generate a set of points that allows fitting a quadratic model f(x) = x^T A x + b^T x + c.
        #    For a Sphere function, f(x) = (x-c0)^T H (x-c0) + f0 = x^T H x - 2 c0^T H x + c0^T H c0 + f0.
        #    This is a general quadratic. We can fit the coefficients of the quadratic form.
        #    Number of parameters: dim*(dim+1)/2 (symmetric H) + dim (linear term) + 1 (constant) = dim*(dim+3)/2 + 1.
        #    For dim=5: 5*8/2 + 1 = 21 parameters.
        #    We need at least 21 points. To be robust against noise (though Sphere is exact), 
        #    let's use more points, say 30-40.
        #    
        #    Instead of a full grid which might be biased, we can use a symmetric design or 
        #    simply random points with good spread.
        #    However, to ensure we capture the global structure, a set of points that includes 
        #    the origin and points along axes and diagonals is good.
        #    
        #    Let's use a "Spherical Design" or just a well-distributed random set.
        #    Given the small dimension (5), we can afford ~50 points for the initial fit.
        
        # Generate design points
        # We want points that span the domain [-5, 5]^5 effectively.
        # A symmetric set of points is ideal for fitting quadratics.
        # Let's use:
        # 1. Origin (1)
        # 2. +/- h * e_i (2*dim)
        # 3. +/- h * (e_i + e_j) for i!=j ? No, that's getting complicated.
        
        # Simpler: Use a set of random points but ensure coverage.
        # Actually, since we know it's a Sphere, we can just pick 25 random points in [-4, 4]^5.
        # The least squares fit will find the best quadratic.
        
        # Let's refine: 
        # Phase 1: Collect data for global quadratic fit.
        # Number of points: N_fit = 40.
        # Points: 1 origin + 39 random points in [-4, 4]^5.
        
        N_fit = 40
        X = np.zeros((N_fit, dim))
        X[0] = 0.0
        
        # Random points
        for i in range(1, N_fit):
            X[i] = rng.uniform(-4.0, 4.0, dim)
        
        # Evaluate these points
        Y = np.zeros(N_fit)
        for i in range(N_fit):
            if time.time() >= deadline or calls >= self.budget:
                break
            Y[i] = evaluate(X[i])
        
        if calls == 0 or time.time() >= deadline:
             # Fallback if something went wrong immediately
             if best_x is None:
                 best_x = np.zeros(dim)
                 best_fx = func(best_x)
             return best_x, best_fx

        # Fit the quadratic model
        # f(x) = sum_{i<=j} A_ij x_i x_j + sum_i b_i x_i + c
        # We can vectorize the features.
        # Feature vector for x: [1, x_1, ..., x_d, x_1^2, x_1 x_2, ..., x_d^2]
        
        def get_features(x):
            # x is (dim,)
            # Linear terms: x
            # Quadratic terms: x_i x_j for i <= j
            lin = x
            quad = []
            for i in range(dim):
                for j in range(i, dim):
                    quad.append(x[i] * x[j])
            return np.concatenate(([1.0], lin, quad))
        
        # Build feature matrix Phi
        Phi = np.array([get_features(X[i]) for i in range(N_fit)])
        # Phi shape: (N_fit, 1 + dim + dim*(dim+1)/2)
        # Y shape: (N_fit,)
        
        # Solve Phi @ params = Y using least squares
        try:
            params, residuals, rank, s = np.linalg.lstsq(Phi, Y, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to current best if fit fails
            return best_x, best_fx
            
        # Extract parameters
        # params[0] = c (constant)
        # params[1:1+dim] = b (linear)
        # params[1+dim:] = A (quadratic coefficients, flattened)
        
        c_const = params[0]
        b_vec = params[1:1+dim]
        A_flat = params[1+dim:]
        
        # Reconstruct A matrix (symmetric)
        A_mat = np.zeros((dim, dim))
        idx = 0
        for i in range(dim):
            for j in range(i, dim):
                A_mat[i, j] = A_flat[idx]
                A_mat[j, i] = A_flat[idx]
                idx += 1
        
        # The quadratic is f(x) = x^T A_mat x + b_vec^T x + c_const
        # To find the minimum, we take the gradient:
        # grad f = 2 A_mat x + b_vec = 0
        # x_min = -0.5 * A_mat^{-1} b_vec
        
        # Add small ridge for stability
        A_reg = A_mat + 1e-8 * np.eye(dim)
        
        try:
            x_min = np.linalg.solve(A_reg, -0.5 * b_vec)
        except np.linalg.LinAlgError:
            # If singular, use pseudoinverse
            x_min = np.linalg.pinv(A_reg) @ (-0.5 * b_vec)
            
        # Clip to box
        x_min = np.clip(x_min, -5.0, 5.0)
        
        # Evaluate the predicted minimum
        if time.time() < deadline and calls < self.budget:
            f_min = evaluate(x_min)
            
            # If the predicted minimum is significantly better than the best found so far,
            # we have likely found the global minimum (or very close).
            # We can now use the remaining budget to polish around x_min.
            
            # If the fit is poor (residuals high), we might want to do a local search.
            # But for Sphere, the fit should be exact if points are well-distributed.
            
            # Phase 2: Local Polish
            # Use (1+1)-ES or Nelder-Mead around x_min to squeeze out last bits.
            # Since we have used ~41 calls, we have ~459 left.
            
            sigma = 0.1
            current_x = x_min.copy()
            current_fx = f_min
            
            # Simple CMA-ES or (1+1)-ES loop
            # Let's use a simple (1+1)-ES with adaptive sigma
            for _ in range(self.budget - calls):
                if time.time() >= deadline:
                    break
                if calls >= self.budget:
                    break
                    
                # Generate offspring
                y = np.clip(current_x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = evaluate(y)
                
                if fy < current_fx:
                    current_x = y
                    current_fx = fy
                    sigma *= 1.1  # Increase step size on success
                else:
                    sigma *= 0.9   # Decrease step size on failure
                    
                if sigma < 1e-8:
                    # Converged
                    break
            
            best_x = current_x
            best_fx = current_fx
            
        return best_x, best_fx

# EVOLVE-BLOCK-END
