"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with gradient estimation phase for Rosenbrock structure."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initial point: center of box
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # Phase 1: Gradient estimation to find descent direction
        # Use central differences to estimate gradient at best point
        grad_step = 0.1
        grad = np.zeros(n)
        for i in range(n):
            if calls_made >= budget or time.time() > deadline:
                break
            x_plus = best_x.copy()
            x_minus = best_x.copy()
            x_plus[i] += grad_step
            x_minus[i] -= grad_step
            x_plus = np.clip(x_plus, -5, 5)
            x_minus = np.clip(x_minus, -5, 5)
            fp = func(x_plus)
            fm = func(x_minus)
            calls_made += 2
            grad[i] = (fp - fm) / (2 * grad_step)
        
        # Check if gradient is informative
        grad_norm = np.linalg.norm(grad)
        use_gradient = grad_norm > 1e-4 and calls_made < budget - 100

        if use_gradient:
            # Phase 2: Gradient descent with adaptive step size
            step_size = 0.5
            no_improve_grad = 0
            max_no_improve_grad = 10
            
            while calls_made < budget - 50:
                if time.time() > deadline:
                    break
                
                # Take a step in negative gradient direction
                x_new = best_x - step_size * grad
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls_made += 1
                
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()
                    no_improve_grad = 0
                    
                    # Re-estimate gradient at new best point
                    grad = np.zeros(n)
                    for i in range(n):
                        if calls_made >= budget - 50 or time.time() > deadline:
                            break
                        x_plus = best_x.copy()
                        x_minus = best_x.copy()
                        x_plus[i] += grad_step
                        x_minus[i] -= grad_step
                        x_plus = np.clip(x_plus, -5, 5)
                        x_minus = np.clip(x_minus, -5, 5)
                        fp = func(x_plus)
                        fm = func(x_minus)
                        calls_made += 2
                        grad[i] = (fp - fm) / (2 * grad_step)
                    
                    # Decay step size slightly as we get closer
                    step_size *= 0.9
                else:
                    no_improve_grad += 1
                    step_size *= 0.7
                    
                    if no_improve_grad >= max_no_improve_grad:
                        break
        
        # Phase 3: (1+1)-ES refinement
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5
        no_improve = 0
        max_no_improve = 20
        success_window = []
        window_size = 8
        restart_count = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_made += 1

            success = fy < fx

            # Update success window
            if len(success_window) >= window_size:
                success_window.pop(0)
            success_window.append(1 if success else 0)

            if success:
                x, fx = y, fy
                no_improve = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                no_improve += 1

            # Adaptive sigma based on success ratio
            if len(success_window) == window_size:
                rate = sum(success_window) / window_size
                if rate > 0.55:
                    sigma *= 1.2
                elif rate < 0.25:
                    sigma *= 0.8

            # Clamp sigma
            sigma = max(0.005, min(sigma, 5.0))

            # Restart logic when stuck
            if no_improve >= max_no_improve:
                restart_count += 1

                # Strategy: try a few local perturbations around best, then global restart
                local_tries = min(4, budget - calls_made)
                found_better = False

                for _ in range(local_tries):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # Perturb around best with moderate step
                    candidate = np.clip(best_x + 0.5 * rng.standard_normal(n), -5, 5)
                    cf = func(candidate)
                    calls_made += 1
                    if cf < best_fx:
                        best_fx = cf
                        best_x = candidate.copy()
                        x = best_x.copy()
                        fx = best_fx
                        no_improve = 0
                        sigma = 0.3
                        found_better = True
                        break

                if not found_better:
                    # Global random restart, bias toward center for Rosenbrock
                    if restart_count % 2 == 0:
                        x_new = rng.uniform(-2, 2, n)
                    else:
                        x_new = rng.uniform(-5, 5, n)
                    fx_new = func(x_new)
                    calls_made += 1
                    if fx_new < best_fx:
                        best_fx = fx_new
                        best_x = x_new.copy()
                        x = x_new.copy()
                        fx = best_fx
                    else:
                        x = x_new.copy()
                        fx = fx_new
                    no_improve = 0
                    sigma = 0.8

                # Reset window on restart
                success_window = []

        # Nelder-Mead refinement with remaining budget
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline - 15:
            try:
                from scipy.optimize import minimize

                nm_calls = [0]

                def wrapper(z):
                    if nm_calls[0] >= remaining or time.time() > deadline:
                        return best_fx
                    nm_calls[0] += 1
                    return func(np.clip(z, -5, 5))

                result = minimize(
                    wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(1, remaining // 2),
                        'xatol': 1e-9,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )

                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = np.clip(result.x, -5, 5)
                    calls_made += nm_calls[0]

            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
