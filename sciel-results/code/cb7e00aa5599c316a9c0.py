"""Derivative-Free Trust-Region optimizer with local quadratic surrogates for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DFTR optimizer: builds local quadratic models from directional probes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 275  # safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sparse global exploration to find a promising basin
        # Use a small number of random points to locate the general area of the minimum
        n_explore = min(20, budget)
        np.random.seed(42)
        for _ in range(n_explore):
            if total_calls >= budget or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            safe_call(x)

        if total_calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Derivative-Free Trust-Region (DFTR)
        # We use a set of 2*dim + 1 points (central + 2*dim directional) to fit a quadratic model.
        # This is the "simplex" or "coordinate" approach for derivative-free optimization.
        
        x0 = best_x.copy()
        f0 = best_f
        
        # Initial trust region radius
        # Start moderately small to ensure local linearity/quadraticity is valid
        delta = 0.5
        
        # Pre-compute directions: e_i and -e_i
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
            directions.append(-d)
        directions = np.array(directions)
        
        # We will iteratively:
        # 1. Evaluate function at x0 + delta * directions (if not already known)
        # 2. Fit a quadratic model f(x) ~= f0 + g^T (x-x0) + 0.5 (x-x0)^T H (x-x0)
        #    Using the 2*dim+1 points, we can solve for g and H.
        #    Actually, with 2*dim+1 points, we have 1 + dim + dim*(dim+1)/2 parameters.
        #    For dim=5: 1 + 5 + 15 = 21 parameters. We have 11 points.
        #    So we are under-determined. We need to assume H is diagonal or use a smaller model.
        #    Standard DFTR (like Nelder-Mead variant or Powell) often uses a linear model + step size,
        #    or a quadratic model with a larger stencil.
        #    
        #    Alternative: Use a "Rosenbrock-like" pattern or just a linear model with adaptive step.
        #    Given the "ill-conditioned" nature, a linear model (Nelder-Mead style) is robust.
        #    However, the prompt suggests "quadratic surrogates".
        #    
        #    Let's use a simpler, robust approach: 
        #    Local quadratic model using 2*dim+1 points is impossible to fully fit H.
        #    But we can fit a *linear* model (gradient) and use the curvature implied by the step.
        #    Or, we can use a "Powell" method which doesn't explicitly build a global H but updates conjugate directions.
        #    
        #    Let's implement a simplified DFTR that fits a linear model (gradient estimation) 
        #    and uses a trust region to determine the step.
        #    This is effectively a "Derivative-Free Gradient Descent" with trust region.
        
        # To honor "quadratic surrogates", we can fit a model of the form:
        # f(x) = f0 + g^T (x-x0) + 0.5 * (x-x0)^T D (x-x0)
        # where D is diagonal. This has dim + dim + 1 = 2*dim + 1 parameters.
        # We have exactly 2*dim + 1 points (center + 2*dim directions).
        # This is solvable!
        
        while total_calls < budget and time.time() < deadline:
            # 1. Build stencil
            stencil_x = [x0]
            stencil_f = [f0]
            
            for d in directions:
                x_probe = x0 + delta * d
                # Check if we have a recent eval for this point? 
                # For simplicity, we re-evaluate or keep a cache. 
                # Given budget 500 and dim 5, 11 calls per iteration is 45 iterations max.
                # This is efficient.
                f_probe = safe_call(x_probe)
                stencil_x.append(x_probe)
                stencil_f.append(f_probe)
                
                if total_calls >= budget or time.time() > deadline:
                    break
            
            if total_calls >= budget or time.time() > deadline:
                break

            # 2. Fit Quadratic Model (Diagonal Hessian)
            # Model: f(x) = f0 + g^T s + 0.5 s^T D s
            # where s = x - x0
            # For point i (direction d_i * delta):
            # f_i = f0 + delta * g^T d_i + 0.5 * delta^2 * (d_i^T D d_i)
            
            # Let's set up the linear system for g and diag(D).
            # Unknowns: g (dim), d_diag (dim). Total 2*dim.
            # Equations: 2*dim (from the 2*dim probe points).
            
            A = np.zeros((2 * dim, 2 * dim))
            b = np.zeros(2 * dim)
            
            for i in range(2 * dim):
                d = directions[i]
                s = delta * d
                f_val = stencil_f[i + 1]
                # f_val = f0 + g^T s + 0.5 s^T D s
                # g^T s = sum(g_j * s_j)
                # s^T D s = sum(d_j^2 * D_j * delta^2) = delta^2 * sum(d_j^2 * D_j)
                # Since d is a basis vector (or negative), d_j^2 is 1 for one j, 0 else.
                # So s^T D s = delta^2 * D_j where j is the axis of d.
                
                # Row i of A corresponds to coefficients of g and D.
                # A[i, :dim] = s (coefficients for g)
                # A[i, dim:] = 0.5 * delta^2 * d^2 (coefficients for D_diag)
                
                A[i, :dim] = s
                A[i, dim:] = 0.5 * delta**2 * (d**2)
                
                b[i] = f_val - f0
            
            try:
                sol = np.linalg.solve(A, b)
                g = sol[:dim]
                D_diag = sol[dim:]
            except np.linalg.LinAlgError:
                # If singular, fall back to simple gradient estimation
                g = np.zeros(dim)
                D_diag = np.zeros(dim)
                for i in range(dim):
                    pos_f = stencil_f[1 + 2*i]
                    neg_f = stencil_f[1 + 2*i + 1]
                    g[i] = (pos_f - neg_f) / (2 * delta)
                    D_diag[i] = (pos_f + neg_f - 2*f0) / (delta**2) if delta > 0 else 0

            # 3. Solve for the step that minimizes the model within the trust region
            # min 0.5 s^T D s + g^T s  s.t. ||s|| <= delta
            # This is a trust-region subproblem.
            # For diagonal D, we can solve this efficiently.
            
            # If D is positive definite, the unconstrained minimum is s* = -D^-1 g.
            # If ||s*|| <= delta, take s*.
            # Otherwise, solve for lambda in s = -(D + lambda I)^-1 g such that ||s|| = delta.
            
            # Ensure D is positive semi-definite for stability
            D_diag = np.maximum(D_diag, 1e-12)
            D = np.diag(D_diag)
            
            s_unconstrained = -np.linalg.solve(D, g)
            
            if np.linalg.norm(s_unconstrained) <= delta:
                s_step = s_unconstrained
            else:
                # Bisection on lambda
                lo, hi = 0.0, 1e12
                for _ in range(50):
                    mid = (lo + hi) / 2
                    s_mid = -np.linalg.solve(D + mid * np.eye(dim), g)
                    norm_s = np.linalg.norm(s_mid)
                    if norm_s > delta:
                        lo = mid
                    else:
                        hi = mid
                s_step = -np.linalg.solve(D + lo * np.eye(dim), g)

            # 4. Evaluate the proposed step
            x_new = x0 + s_step
            f_new = safe_call(x_new)
            
            # 5. Update trust region and center
            rho = (f0 - f_new) / (np.linalg.norm(s_step) * (np.linalg.norm(g) + 1e-12))
            
            if f_new < f0:
                # Success
                x0 = x_new
                f0 = f_new
                if rho > 0.75:
                    delta = min(2 * delta, 5.0) # Expand, but keep in box roughly
                elif rho > 0.25:
                    delta = 1.5 * delta
                else:
                    delta = delta # Keep
            else:
                # Failure
                delta = 0.5 * delta
                if delta < 1e-6:
                    # Stagnation, break or random restart
                    break

            # Clip x0 to box
            x0 = np.clip(x0, -5.0, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
