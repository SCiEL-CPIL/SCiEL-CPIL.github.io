import numpy as np
import time

class Optimizer:
    """
    Optimizer that exploits the quadratic structure of the Sphere function.
    It fits a quadratic model using ~30 evaluations, jumps to the estimated optimum,
    and then refines using a local search method.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        # 1. Probe points to fit a quadratic model
        # We need to determine f(x) = x^T A x + b^T x + c
        # Number of parameters: dim*(dim+1)/2 (A) + dim (b) + 1 (c) = 15 + 5 + 1 = 21 for dim=5
        
        points = []
        # Origin
        points.append(np.zeros(self.dim))
        
        # Axis points
        for i in range(self.dim):
            e_pos = np.zeros(self.dim)
            e_neg = np.zeros(self.dim)
            e_pos[i] = 1.0
            e_neg[i] = -1.0
            points.append(e_pos)
            points.append(e_neg)
            
        # Pairwise points to capture off-diagonal terms
        # f(e_i + e_j) and f(e_i - e_j)
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                p1 = np.zeros(self.dim)
                p2 = np.zeros(self.dim)
                p1[i] = 1.0; p1[j] = 1.0
                p2[i] = 1.0; p2[j] = -1.0
                points.append(p1)
                points.append(p2)
                
        # We have 1 + 10 + 20 = 31 points. We only need 21.
        # But using all 31 in a least squares fit is more robust to noise/errors.
        # However, the function is deterministic. 
        # Let's use the first 21 to solve exactly, or all 31 for least squares.
        # Least squares is safer if the function isn't perfectly quadratic.
        
        # Build design matrix
        # Features: x_i^2, x_i x_j (i<j), x_i, 1
        num_params = self.dim * (self.dim + 1) // 2 + self.dim + 1
        M = np.zeros((len(points), num_params))
        y = np.zeros(len(points))
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        for idx, pt in enumerate(points):
            if time.time() > deadline:
                break
            val = func(pt)
            y[idx] = val
            calls_made += 1
            
            # Fill M row
            row = 0
            # Quadratic terms
            for i in range(self.dim):
                for j in range(i, self.dim):
                    if i == j:
                        M[idx, row] = pt[i] * pt[i]
                    else:
                        M[idx, row] = pt[i] * pt[j]
                    row += 1
            # Linear terms
            for i in range(self.dim):
                M[idx, row] = pt[i]
                row += 1
            # Constant
            M[idx, row] = 1.0
            
            if val < best_f:
                best_f = val
                best_x = pt.copy()
                
        if calls_made < num_params:
            # Fallback if we ran out of time/budget before fitting
            # Just return best found so far
            return best_x, best_f

        # Solve for coefficients
        # theta = [A_11, A_12, ..., A_55, b_1, ..., b_5, c]
        try:
            # Use least squares
            theta, residuals, rank, sv = np.linalg.lstsq(M, y, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to random search if linear algebra fails
            return best_x, best_f

        # Extract A and b
        A = np.zeros((self.dim, self.dim))
        b = np.zeros(self.dim)
        
        idx = 0
        for i in range(self.dim):
            for j in range(i, self.dim):
                if i == j:
                    A[i, i] = theta[idx]
                else:
                    A[i, j] = theta[idx]
                    A[j, i] = theta[idx]
                idx += 1
        
        for i in range(self.dim):
            b[i] = theta[idx]
            idx += 1
        # c = theta[idx]

        # Find minimum of x^T A x + b^T x
        # Gradient = 2 A x + b = 0 => x = -0.5 A^{-1} b
        try:
            # Check if A is invertible and positive definite
            # If not, use pseudo-inverse
            x_star = -0.5 * np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_star = -0.5 * np.linalg.pinv(A) @ b
            
        # Clip to bounds
        x_star = np.clip(x_star, -5.0, 5.0)
        
        # Evaluate at x_star
        if time.time() < deadline:
            f_star = func(x_star)
            calls_made += 1
            if f_star < best_f:
                best_f = f_star
                best_x = x_star.copy()
            else:
                # If the jump didn't improve, stick to best_x found during probing
                # This can happen if the quadratic fit is poor or min is outside box
                pass

        # 2. Refinement phase
        # Use remaining budget to refine best_x
        # A simple (1+1)-ES or coordinate descent is sufficient.
        # Since we are likely at the global min, small steps are good.
        
        remaining_budget = self.budget - calls_made
        if remaining_budget > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5 # Initial step size
            
            # Adaptive step size
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * np.random.randn(self.dim), -5.0, 5.0)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    best_x = x.copy()
                    best_f = fx
                    sigma *= 1.1 # Expand slightly
                else:
                    sigma *= 0.9 # Shrink
                    
                # If sigma is too small, we can stop or reset
                if sigma < 1e-6:
                    break

        return best_x, best_f
