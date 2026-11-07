"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Quadratic model-based optimizer with full Hessian estimation for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 295.0

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def call_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count += 1
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        # Phase 1: Build a full quadratic model using a 2nd-order design
        # We need to estimate c, g (dim), and H (dim x dim symmetric)
        # Total params: 1 + dim + dim*(dim+1)/2
        # For dim=5: 1 + 5 + 15 = 21 params.
        # We need at least 21 points.
        
        delta = 1.0
        design_points = []
        
        # Origin
        design_points.append(np.zeros(dim))
        
        # Coordinate axes
        for i in range(dim):
            p_plus = np.zeros(dim); p_plus[i] = delta
            p_minus = np.zeros(dim); p_minus[i] = -delta
            design_points.append(p_plus)
            design_points.append(p_minus)
            
        # Cross terms for off-diagonal Hessian elements
        # Use points delta * (e_i + e_j) / sqrt(2) to keep norm similar?
        # Actually, simple addition delta*(e_i + e_j) works for finite difference H_ij
        # H_ij approx (f(x+di+dj) - f(x+di) - f(x+dj) + f(x)) / delta^2
        # Let's add these points to the design to help the least squares fit capture correlations
        for i in range(dim):
            for j in range(i+1, dim):
                p_cross = np.zeros(dim)
                p_cross[i] = delta
                p_cross[j] = delta
                design_points.append(p_cross)
                
                # Also add the "minus" cross to improve symmetry and coverage
                p_cross_minus = np.zeros(dim)
                p_cross_minus[i] = delta
                p_cross_minus[j] = -delta
                design_points.append(p_cross_minus)

        # Total points: 1 + 10 + 20 = 31 points.
        # This is more than enough for 21 parameters, giving redundancy for better fit.
        
        # Evaluate all design points
        f_vals = []
        for x in design_points:
            if call_count >= budget or time.time() - start_time > timeout:
                break
            f_vals.append(call_func(x))
            
        # If we ran out of budget, we can't fit. Return best.
        if len(f_vals) < 21:
             return best_x, best_fx

        # Construct the design matrix M for least squares
        # Basis functions: 1, x_1..x_n, x_1^2..x_n^2, x_1x_2..x_4x_5
        n_params = 1 + dim + dim*(dim+1)//2
        
        M = np.zeros((len(f_vals), n_params))
        y = np.array(f_vals)
        
        for k, x in enumerate(design_points[:len(f_vals)]):
            M[k, 0] = 1.0
            M[k, 1:dim+1] = x
            # Quadratic terms
            idx = dim + 1
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        M[k, idx] = x[i] * x[i]
                    else:
                        # For cross terms x_i x_j, the coefficient in the quadratic form x^T H x 
                        # is H_ij * x_i * x_j. 
                        # However, standard quadratic expansion is sum H_ii x_i^2 + sum_{i!=j} H_ij x_i x_j.
                        # If we use basis x_i x_j, the parameter theta_idx corresponds to the coefficient.
                        # Note: In x^T H x, the term for i!=j appears twice (H_ij x_i x_j + H_ji x_j x_i = 2 H_ij x_i x_j).
                        # If we fit f(x) = c + g^T x + sum theta_q q_q(x),
                        # then H_ii = theta_{ii} and 2 H_ij = theta_{ij}.
                        # So we must remember to divide cross-terms by 2 when extracting H.
                        M[k, idx] = x[i] * x[j]
                    idx += 1
        
        # Solve least squares M * theta = y
        try:
            theta, residuals, rank, s = np.linalg.lstsq(M, y, rcond=None)
        except np.linalg.LinAlgError:
            return best_x, best_fx

        # Extract parameters
        c = theta[0]
        g = theta[1:dim+1]
        
        H = np.zeros((dim, dim))
        idx = dim + 1
        for i in range(dim):
            for j in range(i, dim):
                val = theta[idx]
                if i == j:
                    H[i, i] = val
                else:
                    # As derived above, 2 * H_ij = theta_idx for i != j
                    H[i, j] = val / 2.0
                    H[j, i] = val / 2.0
                idx += 1

        # Find the minimum of the quadratic model: 2 H x + g = 0 => x = -0.5 H^-1 g
        # Check if H is invertible (positive definite)
        try:
            # Add small regularization if needed, but Sphere should be well-conditioned
            # If H is singular, fall back to steepest descent
            x_min_model = np.linalg.solve(2.0 * H, -g)
        except np.linalg.LinAlgError:
            # Fallback: simple gradient step from origin
            x_min_model = -0.5 * g
            # Or just use origin
            pass
        
        x_min_model = np.clip(x_min_model, -5.0, 5.0)
        
        # Evaluate the predicted minimum
        if call_count < budget and time.time() - start_time <= timeout:
            f_pred = call_func(x_min_model)
            
        # Phase 2: Polish with Newton steps using the fitted Hessian
        # The Hessian of a sphere is constant, so H is valid anywhere.
        # We just need an accurate gradient at the current best location.
        
        x_current = best_x.copy()
        f_current = best_fx
        
        for it in range(5): # Up to 5 polish iterations
            if call_count >= budget or time.time() - start_time > timeout:
                break
            
            # Estimate gradient at x_current using central differences
            # Use a small step size for accuracy
            h = 1e-4
            grad_local = np.zeros(dim)
            
            for i in range(dim):
                if call_count + 1 >= budget: # Need 2 calls per dim
                    break
                if time.time() - start_time > timeout:
                    break
                    
                e_i = np.zeros(dim)
                e_i[i] = 1.0
                
                x_plus = x_current + h * e_i
                x_minus = x_current - h * e_i
                
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                f_plus = call_func(x_plus)
                f_minus = call_func(x_minus)
                
                grad_local[i] = (f_plus - f_minus) / (2.0 * h)
            
            # Check if gradient is negligible
            if np.linalg.norm(grad_local) < 1e-12:
                break
                
            # Newton step: x_new = x_current - H^-1 * grad / 2
            try:
                step = np.linalg.solve(2.0 * H, -grad_local)
                # Limit step size to prevent flying out of bounds or diverging if model is bad
                step_norm = np.linalg.norm(step)
                if step_norm > 5.0:
                    step = step / step_norm * 5.0
                x_new = x_current + step
            except np.linalg.LinAlgError:
                # If H is singular, use steepest descent
                x_new = x_current - 0.1 * grad_local

            x_new = np.clip(x_new, -5.0, 5.0)
            
            if call_count >= budget:
                break
                
            f_new = call_func(x_new)
            
            if f_new < f_current:
                x_current = x_new
                f_current = f_new
            else:
                # If no improvement, we are likely at the minimum or model is inaccurate
                break

        return best_x, best_fx

# EVOLVE-BLOCK-END
