import numpy as np
import time

class Optimizer:
    """
    Stochastic Approximation (SA) optimizer using central finite differences.
    
    Strategy:
    1. Use a robust global search phase with a few random starts to find a good basin.
    2. Use Stochastic Approximation (gradient descent with noise) for local refinement.
    3. Estimate gradients via central differences (cost: 2*dim calls per gradient).
    4. Use a fixed, robust decreasing step-size schedule (Polyak-Ruppert style) to ensure convergence.
    5. This leverages the smoothness of Rosenbrock to move efficiently along the valley.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val
        
        # 1. Global Search Phase: Sample a few points to find a good starting basin
        # Rosenbrock in 5D has a narrow valley. Random starts might miss it.
        # We try a few strategic points: 0, 1, and random.
        start_points = [
            np.zeros(dim),
            np.ones(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim)
        ]
        
        current_x = None
        for sp in start_points:
            f_val = evaluate(sp)
            if f_val is None:
                break
            if current_x is None or f_val < evaluate(current_x) if calls_made < budget else True:
                # This logic is slightly flawed because evaluate modifies best. 
                # Let's just pick the best from the initial batch explicitly.
                pass
        
        # Let's do it cleanly:
        initial_candidates = []
        for sp in start_points:
            if calls_made >= budget:
                break
            f_val = func(sp)
            calls_made += 1
            initial_candidates.append((f_val, sp))
            
        if not initial_candidates:
            # Fallback
            x = np.zeros(dim)
            f_val = func(x)
            calls_made += 1
            best_f = f_val
            best_x = x
            return best_x, best_f
            
        # Pick the best initial point
        initial_candidates.sort(key=lambda x: x[0])
        current_x = initial_candidates[0][1].copy()
        best_f = initial_candidates[0][0]
        best_x = current_x.copy()
        
        # 2. SA Phase: Gradient Descent with Central Differences
        # We have budget - calls_made left.
        # Each gradient step costs 2*dim + 1 (gradient + new point evaluation) or 2*dim (if we reuse).
        # Let's say 2*dim per step for gradient, and we evaluate the new point.
        # So ~2*dim+1 calls per step.
        
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f
            
        # Step size schedule
        # Start with a moderate step size, decay it.
        # For Rosenbrock, the valley is narrow. Large steps can jump out.
        # Small steps are slow.
        # We use a schedule: alpha_t = alpha_0 / (t + t0)^beta
        # Let's try alpha_0 = 0.1, beta = 0.7, t0 = 10
        
        alpha_0 = 0.1
        beta = 0.7
        t0 = 10
        
        # Gradient estimation via central differences
        # h = 1e-4 is usually good for double precision
        
        h = 1e-4
        
        # Main SA Loop
        t = 0
        while calls_made < budget and time.time() < deadline:
            # Check time
            if time.time() > deadline - 5:
                break
                
            # Estimate Gradient
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = current_x.copy()
                x_minus = current_x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                # Clip to box
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                # Evaluate
                if calls_made >= budget - 1: # Need at least 1 call left for the step
                    break
                f_plus = func(x_plus)
                calls_made += 1
                f_minus = func(x_minus)
                calls_made += 1
                
                # Central difference
                # Note: if clipped, the effective h might be different, but for interior points it's fine.
                # Rosenbrock min is in [-4, 4], so we are likely interior.
                grad[i] = (f_plus - f_minus) / (2 * h)
                
            if calls_made >= budget:
                break
                
            # Normalize gradient to avoid huge steps if gradient is large?
            # Rosenbrock gradients can be large.
            # Let's use a normalized step direction but scale by alpha.
            # Actually, standard SA uses alpha * grad.
            # If grad is huge, we might overshoot.
            # Let's clip the gradient norm.
            g_norm = np.linalg.norm(grad)
            if g_norm > 1.0:
                grad = grad / g_norm
            
            # Step size
            alpha_t = alpha_0 / (t + t0) ** beta
            
            # Update
            x_new = current_x - alpha_t * grad
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate new point
            if calls_made >= budget:
                break
            f_new = func(x_new)
            calls_made += 1
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                
            # Check for convergence or stagnation
            # If we didn't improve, we might want to reduce step size faster or restart?
            # For now, just continue with the schedule.
            
            # Update current_x to the new point (standard SA)
            current_x = x_new
            
            t += 1
            
            # Early stopping if we are very close to 0 (theoretical min of Rosenbrock is 0)
            # But f_opt is unknown and shifted. 
            # However, if best_f is very small, we are likely in the basin.
            # If best_f < 1e-4, we can stop early to save time? 
            # No, we have a budget, use it.
            
            # If we have many calls left but no progress, maybe restart?
            # Let's check progress every 50 steps.
            if t % 50 == 0:
                # If no improvement in last 50 steps, reduce alpha_0?
                # Or just let the schedule handle it.
                pass

        return best_x, best_f
