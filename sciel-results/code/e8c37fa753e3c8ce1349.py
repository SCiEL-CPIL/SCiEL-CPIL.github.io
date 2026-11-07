"""DFTR optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Trust-Region with adaptive quadratic model and robust restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Initial point
        x = np.zeros(n)
        fx = func(x)
        f_best = fx
        x_best = x.copy()
        calls = 1

        # Trust region parameters
        rho = 1.0
        rho_min = 1e-4
        rho_max = 6.0
        gamma_reduce = 0.25
        gamma_accept = 0.75
        gamma_expand = 2.0

        stall_count = 0
        bad_count = 0
        last_improve = 0.0
        best_rho = rho
        H_prev = None

        def _solve_tr_subproblem(g, H, rho):
            """Solve trust-region subproblem: min g^T d + 0.5 d^T H d s.t. ||d|| <= rho."""
            # Steepest descent direction
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-15:
                return np.zeros(n)
            d_sd = -g / g_norm
            f_sd = 0.5 * (d_sd @ H @ d_sd) + g @ d_sd
            
            # Newton direction
            try:
                d_newton = np.linalg.solve(H + 1e-10 * np.eye(n), -g)
            except np.linalg.LinAlgError:
                d_newton = d_sd.copy()
            
            n_norm = np.linalg.norm(d_newton)
            if n_norm <= rho:
                f_newton = 0.5 * (d_newton @ H @ d_newton) + g @ d_newton
                if f_newton < f_sd:
                    return d_newton
                else:
                    return d_sd
            
            # Need to find alpha in [0,1] such that d = alpha * d_newton has ||d|| = rho
            # Solve: min phi(alpha) = 0.5*alpha^2*(d_n^T H d_n) + alpha*(g^T d_n) s.t. alpha*n_norm = rho
            # But we want to find the optimal alpha where the solution lies on the boundary
            # Use the standard approach: find alpha such that d = alpha * d_n is optimal
            # The optimal alpha satisfies: alpha = rho / n_norm if the unconstrained min is outside
            
            # Actually, the standard approach is:
            # If ||d_newton|| > rho, the solution is on the boundary
            # d = alpha * d_newton where alpha = rho / n_norm
            # But we need to check if this is actually optimal
            # The correct approach: solve the secular equation
            # (H + lambda I) d = -g, ||d|| = rho
            # This is equivalent to finding lambda such that ||(H + lambda I)^{-1} g|| = rho
            
            # Simpler: use the fact that for the boundary solution,
            # d = alpha * d_newton, and we need to find alpha
            # Actually, let's just use the boundary point and verify
            d_boundary = d_newton * (rho / n_norm)
            f_boundary = 0.5 * (d_boundary @ H @ d_boundary) + g @ d_boundary
            
            if f_boundary < f_sd:
                return d_boundary
            return d_sd

        while calls + 2 * n + 1 <= budget:
            # Build quadratic model using central differences
            h = max(rho * 0.5, 1e-4)
            f_plus = np.zeros(n)
            f_minus = np.zeros(n)
            
            for j in range(n):
                xp = x.copy()
                xp[j] += h
                xp = np.clip(xp, -5, 5)
                f_plus[j] = func(xp)
                calls += 1
                
                xm = x.copy()
                xm[j] -= h
                xm = np.clip(xm, -5, 5)
                f_minus[j] = func(xm)
                calls += 1

            if calls >= budget:
                break

            # Gradient estimate
            g = (f_plus - f_minus) / (2 * h)
            
            # Diagonal Hessian
            H_diag = (f_plus + f_minus - 2 * fx) / (2 * h * h)
            
            # Ensure positive semi-definite
            H = np.diag(np.maximum(H_diag, 1e-8))
            
            # Add small regularization for stability
            H += 1e-8 * np.eye(n)
            
            # Solve trust-region subproblem
            d = _solve_tr_subproblem(g, H, rho)
            
            # Evaluate new point
            x_new = x + d
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls += 1
            
            # Compute actual vs predicted improvement
            pred_improve = -(g @ d + 0.5 * d @ H @ d)
            act_improve = fx - f_new
            
            # Update trust region
            if act_improve > 1e-10:
                x = x_new
                fx = f_new
                
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
                    stall_count = 0
                    bad_count = 0
                    last_improve = act_improve
                    
                    ratio = act_improve / max(abs(pred_improve), 1e-15)
                    if ratio > gamma_accept:
                        rho = min(rho * gamma_expand, rho_max)
                        best_rho = max(best_rho, rho)
                    else:
                        rho = min(rho * 1.3, rho_max)
                else:
                    stall_count += 1
                    bad_count += 1
                    rho = max(rho * 0.7, rho_min)
            else:
                bad_count += 1
                rho = max(rho * gamma_reduce, rho_min)
                stall_count += 1
            
            # Restart strategy
            if (bad_count >= 3 or stall_count >= 6) and calls + 4 <= budget:
                jump_scale = max(0.3, min(3.0, rho * 0.5 + last_improve * 0.2))
                
                for _ in range(3):
                    if calls >= budget:
                        break
                    x_cand = x_best + rng.normal(0, jump_scale, n)
                    x_cand = np.clip(x_cand, -5, 5)
                    f_cand = func(x_cand)
                    calls += 1
                    
                    if f_cand < f_best:
                        x = x_cand
                        fx = f_cand
                        f_best = fx
                        x_best = x.copy()
                        break
                    elif f_cand < fx:
                        x = x_cand
                        fx = f_cand
                
                rho = max(0.5, best_rho * 0.6)
                stall_count = 0
                bad_count = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
