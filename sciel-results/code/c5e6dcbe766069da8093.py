"""
Optimizer for Bent Cigar (f12) combining Sobol global search with 
surrogate-assisted L-BFGS-B refinement for anisotropic convergence.
"""
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            
            calls_used += 1
            
            # Evaluate
            fx = func(x)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Use a Sobol sequence for low-discrepancy sampling
        # 32 points provides good coverage for 5D without wasting budget
        n_global = 32
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break
        
        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            wrapped_obj(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Surrogate-Assisted Refinement ---
        # Bent Cigar is highly anisotropic. We use a Radial Basis Function (RBF)
        # surrogate to model the landscape locally, then use L-BFGS-B on the
        # surrogate to propose new points. This leverages gradient information
        # (from the smooth surrogate) to navigate the valley efficiently.
        
        # We will collect points as we go.
        # Start with the Sobol points we already have? 
        # We don't store them in the wrapper easily without refactoring.
        # Let's just use the current best and a few new random points around it 
        # to build the initial surrogate, or just start with best_x and expand.
        
        # Actually, a simpler and more robust approach for the "surrogate" part
        # in this constrained budget is to use Powell's method (from inspiration)
        # which effectively adapts steps, OR implement a simple RBF if we store points.
        
        # Given the directive to combine: 
        # Current: Adaptive Coordinate Descent (slow, 2 calls/dim)
        # Inspiration: Powell (fast, adaptive, derivative-free)
        # Directive: Surrogate + L-BFGS-B
        
        # Let's implement a lightweight RBF surrogate approach.
        # 1. Collect a small set of points around the current best.
        # 2. Fit RBF.
        # 3. Minimize RBF using L-BFGS-B.
        # 4. Evaluate the proposed point.
        # 5. Add to dataset, refit, repeat.
        
        # Since we don't have the Sobol points stored in a list easily in the current structure,
        # we will restart the data collection for the surrogate phase.
        # We have used ~32 calls. We have ~468 left.
        
        # Initialize data structure for surrogate
        X_data = []
        F_data = []
        
        # Seed the surrogate data with the current best and some local perturbations
        # to ensure the RBF is well-defined.
        # We will generate a few random points in a local box around x0.
        # Box size: 2.0 (from -1 to 1 relative to x0, clipped to [-5,5])
        
        def add_point(x, f):
            X_data.append(x)
            F_data.append(f)
            
        # Add current best
        add_point(x0, best_fx)
        
        # Generate initial local samples for surrogate
        n_local_init = 10
        rng = np.random.default_rng(123)
        for _ in range(n_local_init):
            if time.time() >= deadline or calls_used >= budget:
                break
            # Sample in a hypercube of side 2 centered at x0
            dx = rng.uniform(-1.0, 1.0, dim)
            x_new = x0 + dx
            x_new = np.clip(x_new, -5, 5)
            try:
                f_new = wrapped_obj(x_new)
                add_point(x_new, f_new)
            except Exception:
                break
        
        if best_x is None:
            return np.zeros(dim), np.inf

        # Surrogate fit function
        def fit_rbf(X, F, x_query):
            """
            Fits a simple RBF (Gaussian) and predicts value and gradient at x_query.
            Using inverse multiquadric or Gaussian. Gaussian is smooth.
            To avoid singularities, we add a small regularization (identity matrix * eps).
            """
            n = len(X)
            if n == 0:
                return 0, np.zeros(dim)
            if n == 1:
                return F[0], np.zeros(dim)
                
            X = np.array(X)
            F = np.array(F)
            
            # Compute pairwise distances
            # D[i,j] = ||X_i - X_j||^2
            D = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
            
            # RBF kernel: exp(-beta * D)
            # Choose beta based on scale. Let's use a fixed beta or adapt.
            # For robustness, use a fixed beta that works for [-5,5] scale.
            # If points are close, beta should be large.
            # Let's use beta = 1.0 / (2 * sigma^2). If sigma ~ 0.5, beta ~ 2.
            # Let's try beta = 1.0.
            beta = 1.0
            K = np.exp(-beta * D)
            
            # Regularize
            reg = 1e-6
            K_reg = K + reg * np.eye(n)
            
            # Solve K * w = F
            try:
                w = np.linalg.solve(K_reg, F)
            except np.linalg.LinAlgError:
                # Fallback to least squares if singular
                w, _, _, _ = np.linalg.lstsq(K_reg, F, rcond=None)
                
            # Predict at x_query
            d_q = np.sum((X - x_query)**2, axis=1)
            f_pred = np.sum(w * np.exp(-beta * d_q))
            
            # Gradient
            # grad_j = sum_i w_i * d/dx_j (exp(-beta * ||x_i - x_q||^2))
            # d/dx_q exp(-beta * D) = exp(-beta * D) * (-beta) * (-2 * (x_q - x_i))
            #                        = 2 * beta * (x_q - x_i) * exp(-beta * D)
            # Note: x_i is fixed, x_q is variable.
            # dist_vec = x_q - X[i]
            # grad = sum_i w_i * 2 * beta * (x_q - X[i]) * exp(-beta * D[i])
            
            dist_vecs = x_query - X  # shape (n, dim)
            exp_term = np.exp(-beta * d_q)  # shape (n,)
            
            # grad = 2 * beta * sum_i w_i * exp_term_i * dist_vecs_i
            # w_i is shape (n,), exp_term_i is shape (n,), dist_vecs_i is shape (dim,)
            # Weighted sum: (w * exp_term) @ dist_vecs
            weights = w * exp_term
            grad = 2.0 * beta * (weights @ dist_vecs)
            
            return f_pred, grad

        # L-BFGS-B objective and gradient for the surrogate
        def surrogate_obj(x):
            f, g = fit_rbf(X_data, F_data, x)
            return f, g
        
        # Main refinement loop
        # We will perform several rounds of surrogate optimization.
        # In each round:
        # 1. Fit RBF on current data.
        # 2. Run L-BFGS-B from current best (or a few random starts) to find surrogate min.
        # 3. Evaluate the proposed point.
        # 4. If improved, update best and add to data.
        # 5. If not, add to data anyway (to inform the surrogate) or reject?
        #    Adding non-improving points helps the surrogate model the landscape,
        #    so we should add them.
        
        # We have limited budget. L-BFGS-B on a surrogate is cheap (no func calls),
        # but each evaluation of the proposed point costs 1 call.
        # So we can afford many surrogate steps.
        
        # Let's run until budget is exhausted or time limit.
        # We will do a "batch" of surrogate proposals.
        
        while calls_used < budget and time.time() < deadline:
            # Fit surrogate
            # Use L-BFGS-B to minimize the surrogate
            # Start from current best
            x_start = best_x.copy()
            
            # Define objective for scipy
            def obj(x):
                f, g = surrogate_obj(x)
                return f
            
            def grad(x):
                f, g = surrogate_obj(x)
                return g
            
            try:
                res = minimize(
                    obj, 
                    x_start, 
                    jac=grad, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 20, 'ftol': 1e-12}
                )
                
                x_proposed = res.x
                # Clip just in case
                x_proposed = np.clip(x_proposed, -5, 5)
                
                # Evaluate proposed point
                f_proposed = wrapped_obj(x_proposed)
                
                # Add to data
                add_point(x_proposed, f_proposed)
                
                # Update best
                # (wrapped_obj already updates best_x and best_fx)
                
            except Exception as e:
                # If L-BFGS-B fails, try a random local step
                if time.time() >= deadline or calls_used >= budget:
                    break
                dx = rng.normal(0, 0.1, dim)
                x_proposed = np.clip(best_x + dx, -5, 5)
                try:
                    f_proposed = wrapped_obj(x_proposed)
                    add_point(x_proposed, f_proposed)
                except Exception:
                    break
        
        return best_x, best_fx
