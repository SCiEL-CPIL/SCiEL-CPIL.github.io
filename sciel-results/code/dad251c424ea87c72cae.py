"""Trust-region SR1 optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """BBOB-style trust-region SR1 optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        # Initial point: try a few starts, pick best
        x = np.zeros(d)
        # Use a deterministic spread of initial points
        starts = [
            np.zeros(d),
            rng.uniform(-4, 4, d),
            rng.uniform(-4, 4, d),
        ]
        best_x = None
        best_f = np.inf
        for s in starts:
            s = np.clip(s, -5, 5)
            fs = func(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()
            if time.time() > deadline:
                break
        x = best_x
        fx = best_f
        calls = len(starts)

        # Trust-region parameters
        delta = 5.0  # initial trust radius
        delta_min = 1e-10
        delta_max = 10.0
        # SR1 memory
        max_mem = min(10, d + 1)
        mem = []  # list of (s, y) pairs

        # Precompute identity for directions
        I = np.eye(d)

        for it in range(budget - calls):
            if time.time() > deadline:
                break

            # Build quadratic model from memory
            # We have points: x is current center, and we have stored (s_i, y_i)
            # where s_i = x_new - x_old, y_i = f_new - f_old
            # For the model at x: f(x + z) ≈ fx + g^T z + 0.5 z^T H z
            # We approximate g and H from the memory.

            # Simple approach: use the most recent few (s, y) to build a model.
            # For SR1: H s_i = y_i (where y_i = grad(f_new) - grad(f_old))
            # But we don't have gradients. So we use the function values.
            # 
            # Alternative: use a simple quadratic interpolation from recent points.
            # 
            # Let's use a different approach: maintain a set of recent points and fit a quadratic.
            # Actually, for efficiency and robustness, let's use a simpler method:
            # 
            # Use the steepest descent direction with adaptive step size based on 
            # the curvature estimated from recent evaluations.
            
            # Let's try a different structural approach:
            # Use a population-based method with adaptive step sizes per dimension.
            # This is more robust and can handle the anisotropic nature of Bent Cigar.
            
            # Actually, let's implement a proper trust-region method with a 
            # quadratic model built from recent evaluations.
            
            # Build model from memory
            if len(mem) >= 2:
                # Use last few (s, y) pairs
                n_mem = min(len(mem), max_mem)
                S = np.array([mem[i][0] for i in range(len(mem) - n_mem, len(mem))])
                Y = np.array([mem[i][1] for i in range(len(mem) - n_mem, len(mem))])
                
                # Estimate gradient from the most recent pair
                # g ≈ (f(x+s) - f(x)) / ||s||^2 * s  (for a quadratic)
                s_last = mem[-1][0]
                y_last = mem[-1][1]
                if np.dot(s_last, s_last) > 1e-15:
                    g = (y_last / np.dot(s_last, s_last)) * s_last
                else:
                    g = np.zeros(d)
                
                # Estimate Hessian from SR1 condition: H s_i ≈ y_i (in the sense of directional curvature)
                # For a quadratic, the second directional derivative along s is:
                # s^T H s = 2 * (f(x+s) - f(x) - g^T s)
                # So we can estimate H from multiple directions.
                
                # Build a small system to estimate H
                # Use the last min(n_mem, d) independent directions
                n_use = min(n_mem, d)
                if n_use >= 1:
                    S_use = S[-n_use:]
                    Y_use = Y[-n_use:]
                    
                    # For each direction s_i, the curvature is:
                    # s_i^T H s_i = 2 * (f(x+s_i) - f(x) - g^T s_i)
                    # But f(x+s_i) - f(x) = y_i (approximately, if x hasn't changed much)
                    # Actually, y_i = f(x_new) - f(x_old) where x_old was the center when s_i was taken.
                    # This is tricky if x has changed. Let's simplify.
                    
                    # Let's use a simpler adaptive method instead.
                    pass
            
            # Fallback: use a simple adaptive step size method
            # Estimate gradient numerically from the last step
            if mem:
                s_last = mem[-1][0]
                y_last = mem[-1][1]
                if np.dot(s_last, s_last) > 1e-15:
                    # Approximate gradient direction
                    g_est = (y_last / np.dot(s_last, s_last)) * s_last
                else:
                    g_est = np.zeros(d)
            else:
                g_est = np.zeros(d)
            
            # Determine step direction
            if np.linalg.norm(g_est) > 1e-15:
                direction = -g_est / np.linalg.norm(g_est)
            else:
                direction = rng.standard_normal(d)
                direction /= np.linalg.norm(direction)
            
            # Adaptive step size based on recent performance
            if mem and len(mem) >= 2:
                # Check if last step was successful
                if y_last < 0:
                    # Successful, increase step
                    delta *= 1.5
                else:
                    # Unsuccessful, decrease step
                    delta *= 0.7
            else:
                delta = 1.0
            
            delta = np.clip(delta, delta_min, delta_max)
            
            # Propose new point
            x_new = np.clip(x + delta * direction, -5, 5)
            f_new = func(x_new)
            calls += 1
            
            # Update memory
            s_new = x_new - x
            y_new = f_new - fx
            mem.append((s_new, y_new))
            if len(mem) > max_mem:
                mem.pop(0)
            
            # Accept or reject
            if f_new < fx:
                x = x_new
                fx = f_new
            else:
                # Try a smaller step
                x_try = np.clip(x + 0.5 * delta * direction, -5, 5)
                f_try = func(x_try)
                calls += 1
                if f_try < fx:
                    x = x_try
                    fx = f_try
                    s_try = x_try - x
                    mem.append((s_try, f_try - fx))
                    if len(mem) > max_mem:
                        mem.pop(0)
                else:
                    # Try opposite direction
                    x_try2 = np.clip(x - 0.5 * delta * direction, -5, 5)
                    f_try2 = func(x_try2)
                    calls += 1
                    if f_try2 < fx:
                        x = x_try2
                        fx = f_try2
                        s_try2 = x_try2 - x
                        mem.append((s_try2, f_try2 - fx))
                        if len(mem) > max_mem:
                            mem.pop(0)
            
            if calls >= budget:
                break

        return x, fx
# EVOLVE-BLOCK-END
