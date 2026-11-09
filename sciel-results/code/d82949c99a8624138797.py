import numpy as np
import time

class Optimizer:
    """DF-TR optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)
        
        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Global Search to find a good starting point
        # Use a small population CMA-ES or random search for the first 20% of budget
        global_budget = int(0.2 * self.budget)
        if global_budget < 10:
            global_budget = 10
            
        # Simple random search for initialization
        # We will evaluate 10 points randomly to pick the best
        # But to be more efficient, let's do a small CMA-ES run
        # Actually, let's just use random search for simplicity and robustness against bad CMA-ES init
        # 10 random points is cheap
        n_init = min(10, global_budget)
        for _ in range(n_init):
            x = np.random.uniform(-5, 5, n)
            f = eval_point(x)
            if f is None:
                break
        
        if calls >= self.budget:
            return best_x, best_f
            
        # Phase 2: Derivative-Free Trust-Region (DF-TR)
        # Start from the best point found
        x_current = best_x.copy()
        f_current = best_f
        
        # Trust region radius
        delta = 1.0
        delta_min = 1e-8
        delta_max = 10.0
        
        # Number of points to sample in the trust region for model building
        # For n=5, we need at least n+1 points for a linear model, 
        # but for quadratic we need more. 
        # A common choice is n + 1 for linear, or 2n for quadratic.
        # Let's use n + 1 points for a linear model first? 
        # No, the directive suggests quadratic. 
        # However, building a full quadratic model requires O(n^2) points.
        # For n=5, n^2 = 25 points. 
        # Let's use a set of 2n points (10 points) and fit a linear model + diagonal Hessian?
        # Or just use the "Polytope" method or similar.
        
        # Simpler DF-TR: 
        # 1. Sample n points along axes and n points along negative axes within delta.
        # 2. Fit a quadratic model.
        
        # Let's implement a simplified DF-TR:
        # We will maintain a set of points. 
        # Initially, we have 1 point (x_current).
        # We will generate 2n new points: x + delta*e_i and x - delta*e_i.
        # This allows fitting a quadratic model with diagonal Hessian and linear gradient.
        
        while calls < self.budget and time.time() < deadline:
            # Generate 2n points around x_current
            # We need to be careful not to exceed budget
            points_needed = 2 * n
            if calls + points_needed > self.budget:
                # If we can't afford a full cycle, do a final polish or break
                break
                
            # Generate points
            new_points = []
            new_fvals = []
            
            for i in range(n):
                x_plus = x_current.copy()
                x_plus[i] += delta
                f_plus = eval_point(x_plus)
                if f_plus is None: break
                new_points.append(x_plus)
                new_fvals.append(f_plus)
                
                x_minus = x_current.copy()
                x_minus[i] -= delta
                f_minus = eval_point(x_minus)
                if f_minus is None: break
                new_points.append(x_minus)
                new_fvals.append(f_minus)
            
            if len(new_points) < 2 * n:
                break
                
            # Fit a quadratic model: f(x) ~ f0 + g^T (x-x0) + 0.5 (x-x0)^T H (x-x0)
            # Using the 2n points, we can solve for g and diagonal H.
            # Let d_i = x_i - x_current
            # f(x_i) - f_current = g^T d_i + 0.5 d_i^T H d_i
            # For diagonal H, H = diag(h_1, ..., h_n)
            # For point i (axis +): d_i = delta * e_i
            # f_plus_i - f_current = g_i * delta + 0.5 * h_i * delta^2
            # For point i (axis -): d_i = -delta * e_i
            # f_minus_i - f_current = -g_i * delta + 0.5 * h_i * delta^2
            
            # We have 2 equations per axis i:
            # A = g_i * delta + 0.5 * h_i * delta^2
            # B = -g_i * delta + 0.5 * h_i * delta^2
            # A + B = h_i * delta^2  => h_i = (A+B)/delta^2
            # A - B = 2 * g_i * delta => g_i = (A-B)/(2*delta)
            
            g = np.zeros(n)
            h_diag = np.zeros(n)
            
            for i in range(n):
                f_plus = new_fvals[2*i]
                f_minus = new_fvals[2*i+1]
                A = f_plus - f_current
                B = f_minus - f_current
                
                h_diag[i] = (A + B) / (delta * delta)
                g[i] = (A - B) / (2 * delta)
                
            # Find the minimizer of the quadratic model
            # f_model(x) = f_current + g^T (x-x_current) + 0.5 (x-x_current)^T diag(h) (x-x_current)
            # Gradient of model: g + H (x-x_current) = 0
            # x_opt - x_current = - H^{-1} g
            # Since H is diagonal:
            step = np.zeros(n)
            for i in range(n):
                if abs(h_diag[i]) > 1e-10:
                    step[i] = -g[i] / h_diag[i]
                else:
                    # If curvature is zero, take a step in direction of -g
                    # Scale by delta to stay in trust region
                    step[i] = -g[i] * delta / (np.linalg.norm(g) + 1e-10)
                    
            # Ensure step is within trust region
            step_norm = np.linalg.norm(step)
            if step_norm > delta:
                step = step * (delta / step_norm)
                
            x_next = x_current + step
            f_next = eval_point(x_next)
            
            if f_next is None:
                break
                
            # Calculate actual improvement
            actual_improvement = f_current - f_next
            
            # Calculate predicted improvement
            # f_model(x_next) - f_current = g^T step + 0.5 step^T H step
            predicted_improvement = 0.0
            for i in range(n):
                predicted_improvement += g[i] * step[i] + 0.5 * h_diag[i] * step[i] * step[i]
            
            # Note: predicted_improvement should be positive if model says we go down
            # actual_improvement is positive if f_next < f_current
            
            if predicted_improvement > 1e-10:
                rho = actual_improvement / predicted_improvement
            else:
                # If predicted improvement is ~0, check if actual is better
                if actual_improvement > 1e-8:
                    rho = 1.0
                else:
                    rho = 0.0
                    
            if rho > 0.25:
                # Successful step
                x_current = x_next
                f_current = f_next
                # Expand trust region
                if rho > 0.75:
                    delta = min(2.0 * delta, delta_max)
                else:
                    delta = min(1.5 * delta, delta_max)
            else:
                # Unsuccessful step
                # Shrink trust region
                delta = max(0.5 * delta, delta_min)
                
            # If delta is very small, we are converged
            if delta < delta_min:
                break
                
        return best_x, best_f
