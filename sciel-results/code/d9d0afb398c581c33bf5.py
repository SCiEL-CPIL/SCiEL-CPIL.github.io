"""Stochastic gradient descent with finite-difference gradients for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """SGD with finite-difference gradient estimation for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Diverse starts to find a good basin early
        starts = [np.zeros(dim)]
        for scale in [1.0, 2.0, 3.0, 4.0]:
            starts.append(np.full(dim, scale))
            starts.append(np.full(dim, -scale))
        for _ in range(12):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        # Phase 2: SGD with finite-difference gradients
        # Use central differences for better gradient accuracy
        h = 1e-4  # Step size for finite differences
        
        def estimate_gradient(x):
            """Estimate gradient using central finite differences."""
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                if n_calls[0] >= budget or time.time() > deadline:
                    return None
                f_plus = safe_func(x_plus)
                if n_calls[0] >= budget or time.time() > deadline:
                    return None
                f_minus = safe_func(x_minus)
                
                if np.isinf(f_plus) or np.isinf(f_minus):
                    return None
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            return grad

        # Adaptive learning rate for SGD
        lr = 0.1
        momentum = 0.9
        velocity = np.zeros(dim)
        current_x = best_x.copy()
        current_f = best_f
        
        no_improve = 0
        stall_threshold = 20
        min_lr = 1e-6
        max_lr = 1.0

        while n_calls[0] < budget and time.time() < deadline and no_improve < stall_threshold:
            # Estimate gradient at current point
            grad = estimate_gradient(current_x)
            if grad is None:
                break
            
            # Check if gradient is too large or has NaN/Inf
            if not np.all(np.isfinite(grad)):
                break
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                no_improve += 1
                continue
            
            # Normalize gradient and apply learning rate
            grad_normed = grad / grad_norm
            
            # Update with momentum
            velocity = momentum * velocity - lr * grad_normed
            candidate = current_x + velocity
            candidate = np.clip(candidate, -5.0, 5.0)
            
            if n_calls[0] >= budget or time.time() > deadline:
                break
            
            f_candidate = safe_func(candidate)
            
            if np.isinf(f_candidate):
                # Step was too large, reduce learning rate
                lr *= 0.5
                velocity *= 0.5
                no_improve += 1
                continue
            
            if f_candidate < current_f:
                # Accept the step
                current_x = candidate.copy()
                current_f = f_candidate
                
                if current_f < best_f:
                    best_f = current_f
                    best_x = current_x.copy()
                    no_improve = 0
                    # Increase learning rate slightly if improving
                    lr = min(lr * 1.05, max_lr)
                else:
                    no_improve += 1
                    # Reduce learning rate if not making global progress
                    lr = max(lr * 0.95, min_lr)
            else:
                # Reject the step
                no_improve += 1
                velocity *= 0.5
                lr = max(lr * 0.8, min_lr)
            
            # Adapt step size h based on function scale
            if current_f > 10:
                h = max(h * 0.9, 1e-6)
            elif current_f < 1:
                h = min(h * 1.1, 1e-2)

        # Phase 3: Fine polish with smaller steps
        if n_calls[0] < budget and time.time() < deadline:
            polish_lr = 0.01
            polish_velocity = np.zeros(dim)
            polish_no_improve = 0
            polish_stall = 15
            
            while n_calls[0] < budget and time.time() < deadline and polish_no_improve < polish_stall:
                grad = estimate_gradient(best_x)
                if grad is None:
                    break
                
                if not np.all(np.isfinite(grad)):
                    break
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-14:
                    break
                
                grad_normed = grad / grad_norm
                polish_velocity = 0.9 * polish_velocity - polish_lr * grad_normed
                candidate = best_x + polish_velocity
                candidate = np.clip(candidate, -5.0, 5.0)
                
                if n_calls[0] >= budget or time.time() > deadline:
                    break
                
                f_candidate = safe_func(candidate)
                
                if np.isinf(f_candidate):
                    polish_lr *= 0.5
                    polish_velocity *= 0.5
                    polish_no_improve += 1
                    continue
                
                if f_candidate < best_f:
                    best_x = candidate.copy()
                    best_f = f_candidate
                    polish_no_improve = 0
                else:
                    polish_no_improve += 1
                    polish_velocity *= 0.5
                    polish_lr = max(polish_lr * 0.8, 1e-8)

        return best_x, best_f
# EVOLVE-BLOCK-END
