"""Combined optimizer: (1+1)-ES global search + Finite-Difference Gradient Descent polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style Weierstrass f16.
    
    Strategy:
    1. Multi-start to find a promising basin.
    2. (1+1)-ES with adaptive step size for global convergence (70% of budget).
    3. Finite-Difference Gradient Descent for local precision (remaining budget).
    4. Deadline guard to ensure termination.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 0: Multi-Start ---
        # Evaluate origin and a few random points to pick a good starting point
        candidates = [np.zeros(dim)]
        for _ in range(5):
            candidates.append(rng.uniform(-4, 4, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        x = best_x
        fx = best_f
        calls_made = len(candidates)

        # --- Phase 1: (1+1)-ES Global Search ---
        # Use 70% of the budget for this phase
        es_budget = int(budget * 0.7)
        sigma = 1.0
        success_window = []
        window_size = 15
        target_success_rate = 0.25  # 1/5 rule variant

        while calls_made < es_budget:
            if time.time() > deadline:
                break

            # Generate offspring
            z = rng.standard_normal(dim)
            trial_x = x + sigma * z
            trial_x = np.clip(trial_x, -5, 5)
            
            trial_f = func(trial_x)
            calls_made += 1

            if trial_f < fx:
                x = trial_x
                fx = trial_f
                success_window.append(1)
            else:
                success_window.append(0)

            # Adapt sigma based on success rate in recent window
            if len(success_window) >= window_size:
                recent = success_window[-window_size:]
                success_rate = sum(recent) / window_size
                
                if success_rate > target_success_rate + 0.1:
                    sigma *= 1.15
                elif success_rate < target_success_rate - 0.1:
                    sigma *= 0.85
                else:
                    sigma *= 1.01  # Slight increase to keep exploring

                # Clamp sigma
                sigma = max(1e-6, min(sigma, 5.0))
                
                # Reset window occasionally to avoid bias, or keep sliding
                # Sliding is fine, but let's trim it to keep memory low
                if len(success_window) > window_size * 2:
                    success_window = success_window[-window_size:]

        # --- Phase 2: Finite-Difference Gradient Descent ---
        # Use the remaining budget for local precision
        # Central differences cost 2*dim calls per step
        # We have roughly 30% of budget left (~150 calls)
        # So we can do ~7-8 gradient steps with central diffs (10 calls each)
        
        # Estimate a good step size for FD
        # If we are close to the minimum, sigma from ES is small. 
        # Use a fixed small eps for FD, or adapt it.
        eps = 1e-4
        
        # Learning rate for gradient descent
        # Start with a moderate step size
        lr = 0.5 
        
        remaining_budget = budget - calls_made
        
        # We will run gradient steps as long as we have enough budget for a full gradient estimation
        # Cost per step: 2 * dim (for central diff)
        cost_per_grad_step = 2 * dim
        
        while calls_made + cost_per_grad_step <= budget:
            if time.time() > deadline:
                break
            
            # Estimate gradient using central differences
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls_made += 2
                
                grad[i] = (f_plus - f_minus) / (2 * eps)
                
                # Early exit if budget exceeded mid-gradient
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break

            # Normalize gradient to prevent huge steps if gradient is large
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break # Converged
            
            # Update x
            # Use a damped gradient step
            step = -lr * grad
            # Limit step size to avoid overshooting
            max_step = 0.5
            if np.linalg.norm(step) > max_step:
                step = step / np.linalg.norm(step) * max_step
                
            x_new = x + step
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls_made += 1
            
            if f_new < fx:
                x = x_new
                fx = f_new
                # Decay learning rate slightly
                lr *= 0.9
            else:
                # If step didn't improve, reduce learning rate significantly
                lr *= 0.5
                # Optional: try a smaller step in the opposite direction? 
                # For now, just reduce lr and try again.
                # To save budget, we might break if lr becomes too small
                if lr < 1e-6:
                    break

        # --- Phase 3: Final Polish ---
        # If we have budget left, do a few tiny random steps
        if calls_made < budget:
            polish_sigma = 1e-5
            while calls_made < budget:
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                trial_x = x + polish_sigma * z
                trial_x = np.clip(trial_x, -5, 5)
                trial_f = func(trial_x)
                calls_made += 1
                if trial_f < fx:
                    x = trial_x
                    fx = trial_f
                else:
                    polish_sigma *= 0.5
                    if polish_sigma < 1e-8:
                        break

        return x, fx
# EVOLVE-BLOCK-END
