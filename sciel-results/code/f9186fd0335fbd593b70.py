"""Hybrid Nelder-Mead + Hessian Newton + (1+1)-ES optimizer for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex, Hessian Newton polish, then (1+1)-ES for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def estimate_hessian(x_center, step=0.01):
            """Estimate Hessian via central finite differences.
            Requires 2*dim+1 function evaluations.
            Returns (H, grad) or None if budget exhausted."""
            nonlocal call_count, best_f, best_x
            if call_count + 2 * dim + 1 > budget:
                return None, None
            
            fc = try_eval(x_center)
            if fc is None:
                return None, None
            
            grad = np.zeros(dim)
            H = np.zeros((dim, dim))
            
            for i in range(dim):
                xp = x_center.copy()
                xp[i] += step
                xmp = x_center.copy()
                xmp[i] -= step
                
                fp = try_eval(xp)
                if fp is None:
                    return None, None
                fmp = try_eval(xmp)
                if fmp is None:
                    return None, None
                
                grad[i] = (fp - fmp) / (2 * step)
                
                for j in range(i, dim):
                    xpp = x_center.copy()
                    xpp[i] += step
                    xpp[j] += step
                    xmm = x_center.copy()
                    xmm[i] -= step
                    xmm[j] -= step
                    xpm = x_center.copy()
                    xpm[i] += step
                    xpm[j] -= step
                    xmp_ = x_center.copy()
                    xmp_[i] -= step
                    xmp_[j] += step
                    
                    fpp = try_eval(xpp)
                    if fpp is None:
                        return None, None
                    fxmm = try_eval(xmm)
                    if fxmm is None:
                        return None, None
                    fxpm = try_eval(xpm)
                    if fxpm is None:
                        return None, None
                    fxmp = try_eval(xmp_)
                    if fxmp is None:
                        return None, None
                    
                    h_ij = (fpp - fxmm - fxpm + fxmp) / (4 * step * step)
                    H[i, j] = h_ij
                    H[j, i] = h_ij
            
            return H, grad

        def newton_step(x, H, grad, max_iter=5, step_scale=1.0):
            """Apply damped Newton steps to converge to the quadratic minimum.
            Returns (x_new, f_new) using the best point found."""
            nonlocal call_count, best_f, best_x
            x = x.copy()
            fx = best_f
            
            # Check if H is well-conditioned
            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) < 1e-10:
                    # Hessian not positive definite, use regularized version
                    H_reg = H + 1e-6 * np.eye(dim)
                else:
                    H_reg = H
                inv_H = np.linalg.inv(H_reg)
            except np.linalg.LinAlgError:
                return x, fx
            
            for _ in range(max_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                # Newton direction: x_new = x - H^{-1} * grad
                direction = -inv_H @ grad
                # Scale the step for safety
                x_new = x + step_scale * direction
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = try_eval(x_new)
                if f_new is None:
                    break
                
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    # Re-estimate gradient at new point using finite differences
                    # Use a smaller step for gradient estimation
                    grad_new = np.zeros(dim)
                    h = 0.005
                    for i in range(dim):
                        xp = x.copy()
                        xp[i] += h
                        xmp = x.copy()
                        xmp[i] -= h
                        fp = try_eval(xp)
                        if fp is None:
                            break
                        fmp = try_eval(xmp)
                        if fmp is None:
                            break
                        grad_new[i] = (fp - fmp) / (2 * h)
                    else:
                        grad = grad_new
                        # Check convergence
                        if np.linalg.norm(grad) < 1e-8:
                            break
                        step_scale *= 0.7  # Damp subsequent steps
                    break
                else:
                    # Step was rejected, reduce scale
                    step_scale *= 0.5
                    if step_scale < 0.01:
                        break
            
            return x, fx

        # Phase 1: Nelder-Mead Simplex Method
        # Initialize simplex
        x0 = np.zeros(dim)
        f0 = try_eval(x0)
        if f0 is None:
            return best_x, best_f

        # Re-initialize simplex to be safe
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            xi = x0.copy()
            xi[i] += 0.5 # Initial step size
            fi = try_eval(xi)
            if fi is None:
                break
            simplex.append(xi)
            simplex_f.append(fi)
        
        if len(simplex) < dim + 1:
            # Not enough calls to form a full simplex, fallback to random or just return
            pass
        else:
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            # Reserve budget for Hessian Newton phase
            nm_budget = max(50, budget - call_count - 120)  # Leave ~120 for Hessian + ES
            for _ in range(nm_budget):
                if call_count >= budget:
                    break
                if time.time() > deadline - 30:
                    break
                
                # Sort simplex by function value
                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = try_eval(xr)
                if fr is None:
                    break
                
                if fr < simplex_f[0]:
                    # Check for expansion
                    if fr < simplex_f[-2]: # Better than second worst
                        xe = centroid + gamma * (centroid - simplex[-1])
                        fe = try_eval(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Between best and second worst
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    # Outer contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = try_eval(xc)
                    if fc is None:
                        break
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrinkage
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fi = try_eval(simplex[i])
                            if fi is None:
                                break
                            simplex_f[i] = fi
                        if call_count >= budget:
                            break

        # Phase 2: Hessian Newton polish
        # After NM has converged to a local region, use Hessian to jump to the quadratic minimum
        if best_x is not None and call_count < budget - 30:
            # Estimate Hessian at current best point
            H, grad = estimate_hessian(best_x.copy(), step=0.01)
            
            if H is not None and grad is not None:
                # Apply Newton steps
                x_new, f_new = newton_step(best_x.copy(), H, grad, max_iter=8, step_scale=1.0)
                
                # If Newton improved, we're done with the heavy lifting
                # The (1+1)-ES will do final polish

        # Phase 3: (1+1)-ES polish
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            for _ in range(min(30, budget - call_count)):
                if time.time() > deadline - 5:
                    break
                if call_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
