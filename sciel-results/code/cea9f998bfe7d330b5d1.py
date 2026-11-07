"""Gradient-based SGD optimizer for BBOB Weierstrass f16.

Uses finite-difference gradient estimation with adaptive step size and
momentum to exploit the smooth, differentiable structure of the Weierstrass
function. Combines multi-start exploration with gradient descent for rapid
local convergence.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """SGD with finite-difference gradients for BBOB Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0
        n_calls = [0]
        rng = np.random.default_rng(42)

        def safe(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float, copy=True)

        def estimate_gradient(x, f0, h=1e-5):
            """Estimate gradient using central finite differences."""
            if n_calls[0] >= budget or time.time() > deadline:
                return None
            grad = np.zeros(dim)
            for i in range(dim):
                if n_calls[0] >= budget or time.time() > deadline:
                    return None
                x_plus = x.copy()
                x_plus[i] += h
                x_plus = np.clip(x_plus, -5.0, 5.0)
                f_plus = safe(x_plus)
                if not np.isfinite(f_plus):
                    return None
                x_minus = x.copy()
                x_minus[i] -= h
                x_minus = np.clip(x_minus, -5.0, 5.0)
                f_minus = safe(x_minus)
                if not np.isfinite(f_minus):
                    return None
                grad[i] = (f_plus - f_minus) / (2.0 * h)
            return grad

        # Phase 1: Multi-start gradient descent
        n_starts = 8
        step_sizes = [2.0, 1.0, 0.5, 0.25]
        
        for s in range(n_starts):
            if n_calls[0] >= budget or time.time() > deadline:
                break
            
            # Choose starting point
            if s == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.0, 4.0, size=dim)
            
            f = safe(x)
            if np.isfinite(f):
                record(x, f)
            
            # Gradient descent with momentum
            momentum = np.zeros(dim)
            beta = 0.9
            lr = step_sizes[min(s, len(step_sizes) - 1)]
            lr_decay = 0.98
            grad_norm_threshold = 1e-8
            max_grad_steps = 40
            stall = 0
            max_stall = 15
            
            for step in range(max_grad_steps):
                if n_calls[0] >= budget or time.time() > deadline:
                    break
                
                # Estimate gradient
                grad = estimate_gradient(x, f)
                if grad is None:
                    break
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm < grad_norm_threshold:
                    break
                
                # Update with momentum
                momentum = beta * momentum + grad
                x_new = x - lr * momentum
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = safe(x_new)
                if not np.isfinite(f_new):
                    lr *= 0.5
                    continue
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    record(x, f)
                    stall = 0
                    # Adaptive learning rate: increase slightly if improving
                    lr = min(lr * 1.02, 2.0)
                else:
                    lr *= 0.7
                    stall += 1
                    if stall >= max_stall:
                        break
            
            # If we used up budget, stop
            if n_calls[0] >= budget:
                break
        
        # Phase 2: Fine-tuning with smaller steps around best found
        if n_calls[0] < budget and time.time() < deadline:
            h_fine = 1e-6
            for _ in range(30):
                if n_calls[0] >= budget or time.time() > deadline:
                    break
                
                grad = estimate_gradient(best_x, best_f, h=h_fine)
                if grad is None:
                    break
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    break
                
                # Small step
                step = 0.01 / (grad_norm + 1e-10)
                x_new = best_x - step * grad
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = safe(x_new)
                if np.isfinite(f_new) and f_new < best_f:
                    best_x = x_new
                    best_f = f_new
                    record(best_x, best_f)
                else:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
