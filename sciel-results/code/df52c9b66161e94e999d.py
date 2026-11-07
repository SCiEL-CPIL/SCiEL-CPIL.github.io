"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Derivative-free quasi-Newton optimizer using Broyden's method.
    Designed for Rosenbrock-like functions with narrow valleys.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 290.0  # Safety margin for 300s limit

    def _numerical_gradient(self, func, x, h=1e-5):
        """Compute numerical gradient using central differences."""
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            # Clip to box [-5, 5]
            xp = np.clip(xp, -5, 5)
            xm = np.clip(xm, -5, 5)
            grad[i] = (func(xp) - func(xm)) / (2 * h)
        return grad

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Jacobian approximation (for gradient map, essentially Hessian for gradient)
        # We approximate the Jacobian of the gradient, i.e., the Hessian.
        # Broyden's method for minimization:
        # We want to solve H s = -g.
        # Update H approx using: H_new = H + (g_new - g_old - H s) s^T / (s^T s)
        # Wait, standard Broyden for root finding f(x)=0 is J.
        # For minimization, we are finding roots of grad(f)=0.
        # So we approximate J = grad(grad(f)) = Hessian.
        # Let's maintain H_inv (inverse Hessian) for efficiency? 
        # Broyden's update for H_inv is cheaper:
        # H_inv_new = H_inv + (s - H_inv g_diff) (s^T H_inv) / (s^T H_inv g_diff)
        # Let's stick to updating H (Hessian approx) as it's more stable to initialize.
        
        H = np.eye(self.dim)
        g = self._numerical_gradient(func, x)
        
        calls_used = 1 # initial eval
        # Gradient evals: 2*dim + 1? No, central diff is 2 per dim.
        # Actually, func is called 2*dim times for gradient + 1 for fx? 
        # My _numerical_gradient calls func 2*dim times.
        # Total calls for first gradient: 1 (fx) + 2*dim.
        # Let's be careful with budget.
        
        # Budget management
        # Each iteration:
        # 1. Compute gradient at x (2*dim calls)
        # 2. Line search: potentially multiple func calls
        # 3. Update H
        #
        # To save budget, we can reuse gradient if x hasn't moved much?
        # No, gradient changes.
        #
        # Let's optimize the gradient calculation.
        # We already have g at x.
        
        for i in range(self.budget):
            if time.time() > self.deadline:
                break
            
            # Check budget
            # We need at least 2*dim + 1 calls per full iteration roughly.
            if calls_used + 2 * self.dim + 5 > self.budget:
                break

            # Compute gradient
            g = self._numerical_gradient(func, x)
            calls_used += 2 * self.dim
            
            # Check convergence
            if np.linalg.norm(g) < 1e-8:
                break

            # Solve H s = -g
            try:
                s = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent step
                s = -g
                # Reset H if singular
                H = np.eye(self.dim)
            
            # Ensure step is not too large
            step_norm = np.linalg.norm(s)
            if step_norm > 10.0:
                s = s / step_norm * 10.0
            
            # Line Search
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            found_step = False
            max_line_search = 20
            
            for ls in range(max_line_search):
                calls_used += 1
                if calls_used > self.budget:
                    break
                
                x_new = x + alpha * s
                # Project to box
                x_new = np.clip(x_new, -5, 5)
                
                # If no movement, break
                if np.allclose(x_new, x):
                    alpha *= rho
                    continue
                
                fx_new = func(x_new)
                
                # Armijo condition
                if fx_new <= fx + c1 * alpha * np.dot(g, s):
                    found_step = True
                    break
                else:
                    alpha *= rho
            
            if not found_step:
                # If line search fails, try a small random step or reset
                # Reset Hessian approximation
                H = np.eye(self.dim)
                # Take a small random step to escape
                step_rand = rng.normal(0, 0.1, self.dim)
                x_new = np.clip(x + step_rand, -5, 5)
                fx_new = func(x_new)
                calls_used += 1
                if fx_new < fx:
                    x, fx = x_new, fx_new
                continue

            # Update Jacobian (Hessian approx)
            # s = x_new - x
            # g_diff = g_new - g_old
            # We need g_new. We can estimate it or just use the change in function?
            # Broyden's method requires the new gradient.
            # To save budget, we might skip the gradient update if we are tight on budget?
            # But we need g_new for the next iteration anyway.
            # Let's compute g_new now? That costs 2*dim.
            # Or we can update H using a simplified rank-1 update if we don't have g_new?
            # Standard Broyden: H_new = H + (g_new - g - H s) s^T / (s^T s)
            # We don't have g_new yet.
            # We can compute g_new for the next iteration's start.
            
            x_old = x
            g_old = g
            x = x_new
            fx = fx_new
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            # Compute new gradient for next iteration (and for H update)
            if calls_used + 2 * self.dim <= self.budget:
                g_new = self._numerical_gradient(func, x)
                calls_used += 2 * self.dim
                
                # Broyden update
                s_vec = x - x_old
                g_diff = g_new - g_old
                denom = np.dot(s_vec, s_vec)
                if denom > 1e-12:
                    # H = H + (g_diff - H @ s_vec) outer s_vec / denom
                    delta = (g_diff - H @ s_vec) / denom
                    H += np.outer(delta, s_vec)
                
                g = g_new
            else:
                # Out of budget for gradient, break
                break

        return best_x, best_fx

# EVOLVE-BLOCK-END
