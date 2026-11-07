import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initialize points
        # We will maintain a set of points and their function values
        # For quadratic fitting, we need at least dim*(dim+1)/2 + 1 points
        # Let's aim for a small population of 10-20 points for the surrogate
        
        points = []
        values = []
        
        # Initial random sampling to establish a baseline and diverse points
        # Use a fixed seed for reproducibility in the surrogate fit, 
        # but the task instances have hidden shifts/rotations, so we rely on the search dynamics.
        rng = np.random.default_rng(42)
        
        # Generate initial points
        # Spread them out to capture the curvature
        for i in range(min(15, budget)):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)
            
        if not points:
            # Fallback if we failed to even get one point (should not happen)
            return np.zeros(dim), float('inf')

        points = np.array(points)
        values = np.array(values)
        
        best_x = points[np.argmin(values)]
        best_f = np.min(values)
        
        # History of best values for AOCC calculation is handled by the evaluator,
        # but we need to return the best x. The evaluator records every call.
        
        # Main loop
        # We will use a mix of surrogate-guided steps and random restarts/exploration
        # to avoid getting stuck in local minima of the surrogate.
        
        call_count = len(points)
        
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # 1. Fit a quadratic model
            # Model: f(x) = c + g^T (x - x0) + 0.5 (x - x0)^T H (x - x0)
            # We fit around the current best point or a centroid.
            # Let's use the best point as the center for the expansion to focus on the minimum.
            x0 = best_x
            
            # Prepare design matrix for least squares
            # Variables: [1, x_i, x_i*x_j (i<=j)]
            n_vars = 1 + dim + dim*(dim+1)//2
            
            if len(points) >= n_vars:
                # Use a subset of the most recent or diverse points if we have too many?
                # For dim=5, n_vars = 1 + 5 + 15 = 21.
                # We have ~15 initial points. We need a few more to fit well.
                # Let's wait until we have at least n_vars points before trusting the surrogate heavily.
                
                # Select points for fitting. Use all if <= 50, else random sample.
                # To ensure we capture the local structure, we prioritize points close to best_x.
                # But for curvature, we need spread.
                
                # Let's just use the last 30 points or all if fewer.
                idx = np.arange(len(points))
                if len(idx) > 30:
                    # Keep best_x and some random others
                    best_idx = np.argmin(values)
                    others = np.setdiff1d(idx, [best_idx])
                    sample_others = rng.choice(others, size=29, replace=False)
                    idx = np.concatenate([[best_idx], sample_others])
                
                X_fit = points[idx] - x0
                y_fit = values[idx]
                
                # Construct design matrix
                # Column 0: 1
                # Columns 1..dim: X_fit[:, i]
                # Columns dim+1..: X_fit[:, i] * X_fit[:, j]
                
                A = np.zeros((len(idx), n_vars))
                A[:, 0] = 1.0
                A[:, 1:dim+1] = X_fit
                
                # Quadratic terms
                col_idx = dim + 1
                for i in range(dim):
                    for j in range(i, dim):
                        A[:, col_idx] = X_fit[:, i] * X_fit[:, j]
                        col_idx += 1
                
                # Solve least squares: A * coeff = y
                try:
                    coeff, residuals, rank, sv = np.linalg.lstsq(A, y_fit, rcond=None)
                except np.linalg.LinAlgError:
                    coeff = None
                
                if coeff is not None:
                    # Extract gradient and Hessian
                    # coeff[0] = c
                    # coeff[1:dim+1] = g
                    # coeff[dim+1:] = H_upper_triangular_elements
                    
                    g = coeff[1:dim+1].copy()
                    
                    # Reconstruct Hessian
                    H = np.zeros((dim, dim))
                    col_idx = dim + 1
                    for i in range(dim):
                        for j in range(i, dim):
                            val = coeff[col_idx]
                            if i == j:
                                H[i, i] = val
                            else:
                                H[i, j] = val / 2.0
                                H[j, i] = val / 2.0
                            col_idx += 1
                    
                    # Regularize Hessian to ensure invertibility and positive definiteness (for minimization)
                    # Add small identity to handle noise/ill-conditioning
                    H_reg = H + 1e-4 * np.eye(dim)
                    
                    # Check if Hessian is well-behaved
                    # If eigenvalues are too small or negative, the model might be poor or we're on a saddle.
                    # For Bent Cigar, we expect one large positive eigenvalue and small others.
                    
                    try:
                        # Newton step: x_new = x0 - H^{-1} g
                        # To avoid huge steps, we can use a line search or limit step size.
                        step = np.linalg.solve(H_reg, g)
                        
                        # Limit step size to prevent jumping out of the box or over the minimum
                        # A step of 1.0 is reasonable for a 5D box.
                        step_norm = np.linalg.norm(step)
                        if step_norm > 2.0:
                            step = step / step_norm * 2.0
                            
                        x_new = x0 - step
                        x_new = np.clip(x_new, -5, 5)
                        
                        # Evaluate
                        f_new = func(x_new)
                        call_count += 1
                        
                        points.append(x_new)
                        values.append(f_new)
                        
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new
                        # If no improvement, we might still want to keep the point for future fits
                        # but we should probably diversify.
                        
                    except np.linalg.LinAlgError:
                        # Fallback to random step
                        x_new = x0 + 0.5 * rng.standard_normal(dim)
                        x_new = np.clip(x_new, -5, 5)
                        f_new = func(x_new)
                        call_count += 1
                        points.append(x_new)
                        values.append(f_new)
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new

            else:
                # Not enough points for a robust quadratic fit yet.
                # Do a random step around the best point to gather more data.
                # Step size decreases as we gather more points?
                # Let's use a moderate step.
                x_new = best_x + 1.0 * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                call_count += 1
                points.append(x_new)
                values.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new

            # Periodically, if we are stagnating (no improvement in last N calls),
            # do a larger random jump to escape local minima of the surrogate.
            # For now, the random initialization and step limits should handle some of this.
            # A more sophisticated approach would track stagnation.

        return best_x, best_f
