"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Quadratic Model Trust-Region Optimizer (2n+1 simplex design).
    
    Strategy:
    1. Maintain a local quadratic model of the function.
    2. Use a trust-region approach to propose the next step.
    3. Evaluate 2n+1 points around the current center to fit the model.
    4. This explicitly tests if a 'model-based' approach (fitting a surface)
       changes the chance of finding the minimum compared to blind (1+1)-ES steps.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s limit
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)
        
        def make_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Initial exploration: Sample a few diverse points to find a good region
        # Since the function is ill-conditioned and shifted, we need to find the basin first.
        # We will use a small random search phase, then switch to local model-based optimization.
        
        # Phase 1: Random Search / Basin Finding
        # Use 50% of the budget or until we find a point with low gradient (approximated by finite differences or just low value)
        # Actually, for f18 (Schaffers F7), the landscape is rugged. A robust strategy is to find the global minimum via multi-start or a robust global method, then refine.
        # Given the budget is 500, let's use 100 calls for random search to find a good candidate, then 400 for local refinement.
        
        # However, the "Quadratic Model" approach is most effective when we are close to the minimum.
        # Let's structure it as:
        # 1. Coarse Search: 50 random points in [-5, 5]^5.
        # 2. Select the best point.
        # 3. Local Model Optimization: Iterate fitting a quadratic model and taking a trust-region step.
        
        # Phase 1: Coarse Search
        n_coarse = 50
        coarse_points = []
        for i in range(n_coarse):
            if call_count >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = make_call(x)
            if fx is not None:
                coarse_points.append((fx, x))
        
        if not coarse_points:
            # Fallback if no calls were made (should not happen)
            x = np.zeros(self.dim)
            make_call(x)
            return best_x, best_f
        
        # Sort by function value
        coarse_points.sort(key=lambda t: t[0])
        best_fx, best_x = coarse_points[0]
        
        # Phase 2: Local Quadratic Model Optimization
        # We will maintain a set of points around the current best to fit a quadratic model.
        # The model is f(x) ≈ c + g^T (x - x0) + 0.5 (x - x0)^T H (x - x0)
        # We need at least (d+1)(d+2)/2 points to determine the coefficients, but for a trust-region step,
        # we can use a simpler approach: fit a linear model + quadratic term using a specific design.
        
        # A common derivative-free approach is the "Simplex" or "Coordinate" based model.
        # Let's use a 2n+1 design: Center + n positive axes + n negative axes.
        # This requires 2n+1 evaluations per step.
        # With n=5, that's 11 evaluations per step.
        # Budget remaining: 450. 450 / 11 ≈ 40 steps. This is reasonable.
        
        x_center = best_x.copy()
        f_center = best_fx
        delta = 0.5  # Initial step size for the model points
        
        # We will iteratively:
        # 1. Generate 2n+1 points around x_center.
        # 2. Evaluate them.
        # 3. Fit a quadratic model (or at least a linear model with a quadratic correction).
        # 4. Find the minimizer of the model within a trust region.
        # 5. Move x_center to this minimizer.
        # 6. Adapt delta.
        
        # To fit the model, we solve for g (gradient) and H (Hessian).
        # Let y_i = f(x_i) - f_center
        # x_i = x_center + delta * e_i (for i=1..n) -> y_i = delta * g_i + 0.5 * delta^2 * H_ii
        # x_i = x_center - delta * e_i (for i=n+1..2n) -> y_i = -delta * g_i + 0.5 * delta^2 * H_ii
        # Adding them: y_i + y_{i+n} = delta^2 * H_ii  => H_ii = (y_i + y_{i+n}) / delta^2
        # Subtracting them: y_i - y_{i+n} = 2 * delta * g_i => g_i = (y_i - y_{i+n}) / (2 * delta)
        
        # This gives us the diagonal of the Hessian and the gradient.
        # We assume the off-diagonal terms of H are zero for simplicity (separable approximation) or we can ignore them.
        # The model minimizer for f(x) ≈ f_center + g^T d + 0.5 d^T H d (with H diagonal) is:
        # d_i = -g_i / H_ii (if H_ii > 0)
        
        # If H_ii <= 0, the function is concave in that direction, so we should take a step in the direction that decreases f the most.
        
        while call_count < self.budget:
            if time.time() > deadline - 5.0:
                break
            
            # Check if we have enough budget for a full 2n+1 step
            if self.budget - call_count < 2 * self.dim + 1:
                # Use remaining budget for a simple (1+1)-ES refinement or just break
                # Let's do a few simple steps if possible
                while call_count < self.budget:
                    z = rng.standard_normal(self.dim)
                    x_child = x_center + 0.1 * z
                    x_child = np.clip(x_child, -5, 5)
                    fx = make_call(x_child)
                    if fx is None:
                        break
                    if fx < f_center:
                        x_center = x_child
                        f_center = fx
                    else:
                        # If no improvement, shrink delta and try again? 
                        # For now, just break to avoid wasting calls on bad directions
                        break
                break
            
            # Generate 2n+1 points
            points = [x_center]
            vals = [f_center]
            
            for i in range(self.dim):
                x_pos = x_center.copy()
                x_pos[i] += delta
                x_pos = np.clip(x_pos, -5, 5)
                
                x_neg = x_center.copy()
                x_neg[i] -= delta
                x_neg = np.clip(x_neg, -5, 5)
                
                points.append(x_pos)
                points.append(x_neg)
            
            # Evaluate all points
            for p in points[1:]:
                if call_count >= self.budget:
                    break
                fx = make_call(p)
                if fx is None:
                    break
                vals.append(fx)
            
            if len(vals) < 2 * self.dim + 1:
                break
            
            # Fit the model
            g = np.zeros(self.dim)
            H_diag = np.zeros(self.dim)
            
            for i in range(self.dim):
                y_pos = vals[1 + 2*i] - f_center
                y_neg = vals[2 + 2*i] - f_center
                
                g[i] = (y_pos - y_neg) / (2 * delta)
                H_diag[i] = (y_pos + y_neg) / (delta * delta)
                
                # If H_diag is negative or very small, it's not a convex direction.
                # We set a lower bound for H_diag to avoid division by zero or negative curvature issues.
                # If H_diag < 0, the quadratic model suggests moving away from the center in that direction? 
                # Actually, if H < 0, the function is concave. The minimizer of the quadratic model is at infinity.
                # In that case, we should take a step in the direction of the negative gradient.
            
            # Calculate the ideal step d
            d = np.zeros(self.dim)
            for i in range(self.dim):
                if H_diag[i] > 1e-8:
                    d[i] = -g[i] / H_diag[i]
                else:
                    # If curvature is non-positive, move in the direction of the negative gradient
                    # Limit the step size to delta to stay within the trust region
                    d[i] = -np.sign(g[i]) * delta if g[i] != 0 else 0
            
            # Clip the step to a trust region (e.g., 2 * delta)
            norm_d = np.linalg.norm(d)
            max_step = 2 * delta
            if norm_d > max_step:
                d = d * (max_step / norm_d)
            
            x_new = x_center + d
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate the new center
            f_new = make_call(x_new)
            if f_new is None:
                break
            
            if f_new < f_center:
                # Success
                x_center = x_new
                f_center = f_new
                # Increase delta slightly
                delta *= 1.5
                delta = min(delta, 5.0)
            else:
                # Failure
                # Decrease delta
                delta *= 0.5
                delta = max(delta, 1e-4)
                
                # If delta is too small, we might be stuck. 
                # We could try a random jump, but for now, let's just continue with smaller steps.
                # If we stagnate for many iterations, we might want to restart.
                # But with 500 budget, we should be fine.

        return best_x, best_f
# EVOLVE-BLOCK-END
