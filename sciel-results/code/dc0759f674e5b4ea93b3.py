"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Derivative-free Gauss-Newton solver with finite-difference Jacobian.
    
    The Bent Cigar function (BBOB f12) is highly ill-conditioned.
    Instead of sampling a distribution (CMA-ES) or moving a simplex (Nelder-Mead),
    we explicitly estimate the local gradient and Hessian structure (via Gauss-Newton)
    to take large, directed steps along the narrow valley.
    
    Key structural differences:
    1. Gradient estimation: We compute a finite-difference Jacobian to estimate the gradient.
    2. Gauss-Newton Step: We solve (J^T J) dx = -J^T r to find the step direction.
       For a scalar function f(x), J is the gradient g. So we solve (g g^T) dx = -g * f(x)?
       Wait, Gauss-Newton is for residuals. For a single scalar function, 
       the Hessian approximation is g g^T.
       The step is dx = - (g g^T + lambda I)^-1 g.
       This is essentially a scaled gradient step with adaptive line search.
       However, for ill-conditioned problems, this might not be enough.
       
       Let's refine: The Bent Cigar is f(x) = 100 * sum((x_i - x_{i+1})^2) + (x_1 - x*)^2 ... 
       Actually, BBOB f12 is:
       f(x) = 100 * (x_1 - x_2)^2 + ... + 100*(x_{n-1} - x_n)^2 + (x_1 - x*)^2
       (with shifts/rotations).
       
       The gradient is dense. A pure Gauss-Newton step for a scalar objective is just 
       gradient descent with a specific step size. 
       
       To capture the "valley" structure better, we can use a quasi-Newton method like BFGS,
       which maintains an approximation of the Hessian. This is a different design family 
       from CMA-ES (which adapts a full covariance matrix of the *search distribution*) 
       and NM (which moves a simplex). BFGS adapts the *inverse Hessian* of the objective.
       
       Let's implement a derivative-free BFGS with finite-difference gradients.
       This is a "quasi-Newton" approach.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        self.deadline = self.start_time + 290.0
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= self.budget:
                return float('inf') # Graceful degradation
            self.calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # Finite Difference Gradient Estimator
        eps = 1e-5
        def estimate_gradient(x):
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = wrapped_func(x_plus)
                
                x_minus = x.copy()
                x_minus[i] -= eps
                f_minus = wrapped_func(x_minus)
                
                grad[i] = (f_plus - f_minus) / (2 * eps)
            return grad

        # BFGS Parameters
        # Initial inverse Hessian approximation H = I
        H = np.eye(dim)
        
        # Initial point
        x = np.random.uniform(-4, 4, dim)
        f_x = wrapped_func(x)
        g_x = estimate_gradient(x)
        
        # Line search parameters
        c1 = 1e-4
        c2 = 0.9
        max_ls_iter = 20
        step_size = 1.0
        
        # BFGS Loop
        # We need to manage the budget carefully.
        # Each iteration: 1 gradient (2*dim calls) + line search (variable)
        # Let's estimate ~ 2*dim + 5 calls per iteration.
        # 500 budget / 15 calls/iter = ~33 iterations.
        
        while self.calls < budget and time.time() < self.deadline:
            # Check convergence
            if np.linalg.norm(g_x) < 1e-8:
                break
            
            # Compute search direction p = -H @ g
            p = -H @ g_x
            
            # Line search to find alpha
            alpha = step_size
            f_x_old = f_x
            g_x_old = g_x
            x_old = x.copy()
            
            found_alpha = False
            for _ in range(max_ls_iter):
                if self.calls >= budget or time.time() > self.deadline:
                    break
                    
                x_new = x + alpha * p
                x_new_clipped = np.clip(x_new, -5, 5)
                f_new = wrapped_func(x_new_clipped)
                
                # Armijo condition
                if f_new < f_x + c1 * alpha * (p @ g_x):
                    found_alpha = True
                    break
                else:
                    alpha *= 0.5
            
            if not found_alpha:
                # If line search fails, take a small step in negative gradient direction
                alpha = 1e-4
                x_new = x + alpha * (-g_x)
                x_new_clipped = np.clip(x_new, -5, 5)
                f_new = wrapped_func(x_new_clipped)
            
            if self.calls >= budget or time.time() > self.deadline:
                break

            # Update BFGS
            g_new = estimate_gradient(x_new_clipped)
            
            s = x_new_clipped - x_old
            y = g_new - g_x_old
            sy = s @ y
            
            if sy > 1e-10:
                # BFGS update
                # H_new = H + (s s^T / (s^T y)) - (H y y^T H) / (y^T H y)
                yH = H @ y
                yHy = y @ yH
                
                if yHy > 1e-10:
                    H = H + (np.outer(s, s) / sy) - (np.outer(yH, yH) / yHy)
                else:
                    # Reset H if it becomes singular or ill-conditioned
                    H = np.eye(dim)
            else:
                # If sy is too small, we might be at a minimum or the step was bad.
                # Reset H to identity to restart curvature estimation
                H = np.eye(dim)

            # Update state
            x = x_new_clipped
            f_x = f_new
            g_x = g_new
            
            # Periodically reset H to prevent drift in high dimensions or if it becomes ill-conditioned
            # This is a heuristic "restart" within the BFGS framework
            if self.calls % (2 * dim) == 0:
                # Check condition number of H
                try:
                    eigvals = np.linalg.eigvalsh(H)
                    if eigvals[0] < 1e-10 or eigvals[-1] > 1e10:
                        H = np.eye(dim)
                except:
                    H = np.eye(dim)

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
