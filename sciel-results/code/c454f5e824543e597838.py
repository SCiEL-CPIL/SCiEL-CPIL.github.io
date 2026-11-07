"""Derivative-Free Trust-Region (DFTR) optimizer for BBOB f18 (Schaffers F7).

Unlike population-based CMA-ES, this method builds an explicit local quadratic
model of the objective function using a set of directional points around the
current center. This allows for more precise navigation of the ill-conditioned
basin by directly estimating the local geometry (gradient and Hessian) rather
than relying on statistical convergence of a population.
"""
import time
import numpy as np


class Optimizer:
    """DFTR optimizer with box repair and a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        WALL = 290.0  # Safety margin before 330s kill
        calls = [0]
        n = self.dim

        def time_ok():
            return (time.time() - t0) < WALL

        def safe(x):
            if calls[0] >= self.budget or not time_ok():
                return None
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                v = 1e18
            if not np.isfinite(v):
                v = 1e18
            calls[0] += 1
            return v

        # State
        best_x = np.zeros(n)
        best_f = np.inf
        rng = np.random.default_rng(42)

        # DFTR Parameters
        # We use a small number of points to build the local model.
        # For dimension n, we need at least n+1 points for a linear model,
        # but for a quadratic model we need ~n(n+1)/2 + 1 points.
        # To keep it efficient and robust within the budget, we use a "polyhedral"
        # or "directional" approximation which is cheaper.
        # Strategy: Use a set of 2n + 1 points (center + +/- directions) to build
        # a linear model, and adaptively add points if the model is good, or
        # simply use a fixed set of 2n+1 points for a robust linear/quadratic
        # hybrid step.
        
        # Actually, a full quadratic model in 5D requires 1+5+10 = 16 points.
        # We have 500 budget. 500/16 ~ 31 iterations. This is feasible.
        # Let's implement a proper Quadratic Model DFTR.

        # Generate a set of directions to span the space.
        # We need a set of points {x_k} such that the matrix of differences is full rank
        # for the quadratic terms.
        # Standard approach: Use a "standard simplex" or "orthogonal directions".
        # For n=5, we can use the 2n+1 points of a regular simplex centered at 0,
        # but scaled. However, for quadratic interpolation, we need points that
        # are distinct and not all on a hyperplane.
        
        # Let's use a set of 2n+1 points generated from a Hadamard matrix or similar
        # if available, or just random points that are well-spread.
        # Better: Use the vertices of a cross-polytope (2n points) + center?
        # No, for quadratic interpolation we need the points to determine the Hessian.
        # The number of parameters in a quadratic model in n dims is 1 + n + n(n+1)/2.
        # For n=5: 1 + 5 + 15 = 21 parameters.
        # So we need at least 21 points.
        # 500 / 21 ~ 23 iterations. Still feasible.

        # Let's generate 21 well-spread points in the unit ball.
        # We can use the first 21 points of a Sobol sequence or a random set.
        # To ensure stability, we'll generate random points on the surface of a sphere
        # and add the center.
        
        # Initialize model points relative to the current center m.
        # We will generate a fixed set of offsets D (21 x n) such that the
        # quadratic model can be uniquely determined.
        
        # A robust way to get these points is to use the vertices of a regular
        # simplex in n+1 dimensions projected to n, or just use random points
        # and check condition number. Given the small size, random points with
        # a fixed seed are sufficient.
        
        # Let's create the offsets.
        # We need 21 points.
        num_points = 1 + n + n*(n+1)//2
        
        # Generate random offsets
        offsets = rng.normal(0, 1, (num_points, n))
        # Normalize to unit length to keep them within a reasonable radius initially
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        offsets = offsets / norms
        # Add center
        offsets = np.vstack([np.zeros(n), offsets])
        # Now we have 22 points? No, num_points includes the center?
        # 1 (center) + n (linear) + n(n+1)/2 (quadratic) = 21.
        # So we need 20 non-center points + 1 center.
        # Let's adjust.
        num_non_center = num_points - 1
        offsets_nc = rng.normal(0, 1, (num_non_center, n))
        norms_nc = np.linalg.norm(offsets_nc, axis=1, keepdims=True)
        offsets_nc = offsets_nc / norms_nc
        offsets = np.vstack([np.zeros(n), offsets_nc])
        
        # Check if the design matrix for the quadratic model is well-conditioned.
        # The model is f(x) ~ c + g^T (x-m) + 0.5 (x-m)^T H (x-m)
        # We fit c, g, H.
        # We construct a matrix A where each row corresponds to a point x_k = m + delta_k
        # A_k = [1, delta_k, outer(delta_k, delta_k) vectorized]
        # We need A to be invertible (or well-conditioned).
        
        def build_design_matrix(deltas):
            # deltas: (K, n)
            K = deltas.shape[0]
            A = np.zeros((K, num_points))
            A[:, 0] = 1.0
            A[:, 1:1+n] = deltas
            # Vectorize symmetric matrix
            for i in range(K):
                d = deltas[i]
                # Fill the upper triangle of the quadratic terms
                idx = 1 + n
                for r in range(n):
                    for c in range(r, n):
                        A[i, idx] = d[r] * d[c]
                        idx += 1
            return A

        A_design = build_design_matrix(offsets)
        # If singular, regenerate (unlikely with random, but just in case)
        try:
            cond = np.linalg.cond(A_design)
            if cond > 1e10:
                # Regenerate
                offsets_nc = rng.normal(0, 1, (num_non_center, n))
                norms_nc = np.linalg.norm(offsets_nc, axis=1, keepdims=True)
                offsets_nc = offsets_nc / norms_nc
                offsets = np.vstack([np.zeros(n), offsets_nc])
                A_design = build_design_matrix(offsets)
        except np.linalg.LinAlgError:
            pass

        # Initial state
        m = rng.uniform(-3.0, 3.0, n)
        rho = 2.0  # Initial trust region radius
        g_since_improve = 0
        stagnation_limit = 15 # Iterations without improvement
        
        # Main Loop
        while calls[0] < self.budget and time_ok():
            # 1. Evaluate the model points
            # Points are m + rho * offsets
            X = m[None, :] + rho * offsets
            X = np.clip(X, -5.0, 5.0)
            
            f_vals = []
            for x in X:
                v = safe(x)
                if v is None:
                    f_vals.append(1e18)
                else:
                    f_vals.append(v)
            f_vals = np.array(f_vals)
            
            # Update best
            min_idx = np.argmin(f_vals)
            if f_vals[min_idx] < best_f:
                best_f = float(f_vals[min_idx])
                best_x = X[min_idx].copy()
                
            # Check for stagnation
            # If the best point in the model is not significantly better than the center
            # or if the model prediction doesn't match, we might need to adjust.
            
            # 2. Fit the quadratic model
            # We have f_vals and A_design.
            # Solve A_design * theta = f_vals for theta = [c, g, H_vec]
            try:
                theta, residuals, rank, s = np.linalg.lstsq(A_design, f_vals, rcond=None)
            except np.linalg.LinAlgError:
                theta = np.zeros(num_points)
                
            c_model = theta[0]
            g_model = theta[1:1+n]
            H_vec = theta[1+n:]
            
            # Reconstruct Hessian
            H_model = np.zeros((n, n))
            idx = 0
            for r in range(n):
                for c in range(r, n):
                    val = H_vec[idx]
                    H_model[r, c] = val
                    if r != c:
                        H_model[c, r] = val
                    idx += 1
            
            # 3. Find the minimizer of the quadratic model within the trust region
            # Minimize q(y) = c + g^T y + 0.5 y^T H y  subject to ||y|| <= 1
            # (We work in normalized coordinates where y = (x-m)/rho, so ||y||<=1 implies ||x-m||<=rho)
            
            # Use scipy or a simple iterative method? 
            # SciPy is available. We can use scipy.optimize.minimize with bounds.
            # But calling scipy inside the loop might be slow. 
            # Let's use a simple projected gradient or Newton step with line search.
            
            # Actually, for n=5, we can just use scipy.optimize.minimize on the model.
            # It's fast enough.
            
            def model_func(y):
                return c_model + g_model @ y + 0.5 * y @ H_model @ y
            
            def model_grad(y):
                return g_model + H_model @ y
            
            # Initial guess for the minimizer is 0 (center)
            y0 = np.zeros(n)
            
            try:
                from scipy.optimize import minimize
                res = minimize(model_func, y0, jac=model_grad, method='L-BFGS-B', 
                               bounds=[(-1.0, 1.0)] * n)
                y_opt = res.x
            except Exception:
                # Fallback: Simple gradient step
                y_opt = -g_model
                norm_y = np.linalg.norm(y_opt)
                if norm_y > 1.0:
                    y_opt = y_opt / norm_y

            # 4. Calculate predicted step and actual step
            x_proposed = m + rho * y_opt
            x_proposed = np.clip(x_proposed, -5.0, 5.0)
            
            # Evaluate proposed point
            f_proposed = safe(x_proposed)
            if f_proposed is None:
                break
                
            # 5. Acceptance criteria
            # Predicted reduction: c_model - model_func(y_opt)
            # Actual reduction: f_center - f_proposed
            # We need f_center. It's f_vals[0] (since offsets[0] is 0)
            f_center = f_vals[0]
            
            pred_red = c_model - model_func(y_opt)
            act_red = f_center - f_proposed
            
            # If the model is accurate, accept and expand rho
            # If the model is poor, reject and shrink rho
            if act_red > 0.25 * pred_red and act_red > 1e-8:
                # Accept
                m = x_proposed.copy()
                rho = min(2.0 * rho, 4.0) # Expand trust region
                g_since_improve = 0
            else:
                # Reject
                rho = 0.5 * rho
                g_since_improve += 1
                
                # If rho is too small, we might be stuck. 
                # If stagnation is high, restart with a random point?
                if rho < 1e-4:
                    # Restart: pick a new random center
                    m = best_x + rng.normal(0, 1.0, n)
                    m = np.clip(m, -5.0, 5.0)
                    rho = 1.0
                    g_since_improve = 0

            # If we have stagnated for a long time, do a global restart
            if g_since_improve > stagnation_limit:
                m = best_x + rng.normal(0, 0.5, n)
                m = np.clip(m, -5.0, 5.0)
                rho = 1.0
                g_since_improve = 0

        # Tail: (1+1)-ES refinement
        if calls[0] < self.budget and time_ok():
            m_tail = best_x.copy()
            sigma_tail = 0.5
            while calls[0] < self.budget and time_ok():
                y = m_tail + (sigma_tail * 0.5) * rng.standard_normal(n)
                y = np.clip(y, -5.0, 5.0)
                fy = safe(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f, best_x = fy, y.copy()
                    m_tail = best_x
                    sigma_tail *= 1.1
                else:
                    sigma_tail *= 0.9
                    sigma_tail = float(np.clip(sigma_tail, 1e-8, 10.0))

        return best_x, float(best_f)
