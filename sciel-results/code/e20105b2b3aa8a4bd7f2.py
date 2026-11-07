"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive-epsilon gradient + CMA-ES + Nelder-Mead for rotated Rosenbrock (f9)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def estimate_gradient_adaptive(x, f_current, h_start=1e-3, h_min=1e-6):
            """Adaptive-epsilon gradient estimation: start coarse, refine if noisy."""
            nonlocal calls
            h = h_start
            for attempt in range(3):
                if calls >= budget or time.time() > deadline:
                    return None, h
                grad = np.zeros(n)
                valid = True
                for i in range(n):
                    if calls >= budget or time.time() > deadline:
                        valid = False
                        break
                    x_plus = x.copy()
                    x_plus[i] += h
                    x_minus = x.copy()
                    x_minus[i] -= h
                    f_plus = func(x_plus)
                    calls += 1
                    if calls >= budget or time.time() > deadline:
                        valid = False
                        break
                    f_minus = func(x_minus)
                    calls += 1
                    grad[i] = (f_plus - f_minus) / (2 * h)
                if not valid:
                    return None, h
                # Check consistency: if gradient norm is very small relative to f,
                # we might be in a flat region; try smaller h
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-12 and attempt < 2:
                    h *= 0.1
                    continue
                break
            return grad, h

        def bfgs(x0, f0, max_iter=100):
            """BFGS optimization with finite difference gradients."""
            nonlocal calls, best_x, best_f
            
            x = x0.copy()
            f = f0
            h = 1e-5
            H = np.eye(n)  # Inverse Hessian approximation
            g_prev = None
            x_prev = None
            
            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Estimate gradient
                grad = np.zeros(n)
                valid = True
                for i in range(n):
                    if calls >= budget or time.time() > deadline:
                        valid = False
                        break
                    x_plus = x.copy()
                    x_plus[i] += h
                    f_plus = func(x_plus)
                    calls += 1
                    if calls >= budget or time.time() > deadline:
                        valid = False
                        break
                    x_minus = x.copy()
                    x_minus[i] -= h
                    f_minus = func(x_minus)
                    calls += 1
                    grad[i] = (f_plus - f_minus) / (2 * h)
                
                if not valid:
                    break
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-8:
                    break
                
                # Search direction p = -H * grad
                p = -H @ grad
                
                # Line search (Backtracking)
                alpha = 1.0
                c1 = 1e-4
                rho = 0.5
                improved = False
                for ls in range(20):
                    if time.time() > deadline or calls >= budget:
                        break
                    x_new = x + alpha * p
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1
                    
                    # Armijo condition
                    if f_new <= f + c1 * alpha * (grad @ p):
                        improved = True
                        break
                    alpha *= rho
                
                if not improved:
                    break
                
                # Update BFGS matrix
                if g_prev is not None and x_prev is not None:
                    s = x - x_prev
                    y = grad - g_prev
                    sy = s @ y
                    if sy > 1e-10:
                        Hy = H @ y
                        H = H - (np.outer(s, s) * (s @ y) + np.outer(s, Hy) + np.outer(Hy, s)) / (y @ Hy) + (1 + (s @ y)/(y @ Hy)) * np.outer(y, y) / (y @ y)
                        # Simpler BFGS update: H_new = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
                        # Let's use the standard stable form
                        rho = 1.0 / sy
                        H = (1 - rho * (s @ y)) * H + rho * np.outer(s, s)
                        # Wait, the standard BFGS update for inverse Hessian H is:
                        # H_{k+1} = (I - rho_k s_k y_k^T) H_k (I - rho_k y_k s_k^T) + rho_k s_k s_k^T
                        # This is computationally intensive. Let's stick to the simpler approximation or just use gradient descent if BFGS is too complex to implement correctly in limited space/time.
                        # Actually, the simple update H = H - (H y y^T H)/(y^T H y) + (s s^T)/(y^T s) is for the Hessian H.
                        # For inverse Hessian H_inv:
                        # H_inv_new = (I - rho s y^T) H_inv (I - rho y s^T) + rho s s^T
                        # Let's re-initialize H if it becomes ill-conditioned or just use a simpler momentum-based approach if BFGS fails.
                        # Given the constraint, I will implement a simplified BFGS or just switch to a robust line-search gradient descent with momentum if BFGS is too risky.
                        # However, the prompt says BFGS worked well. I will implement the standard inverse Hessian update carefully.
                        
                        # Let's revert to a simpler, more robust approach if BFGS implementation is too prone to errors in this context.
                        # I will use a "L-BFGS" like memory or just standard BFGS.
                        # Let's try the standard BFGS update for H_inv:
                        # H = (I - rho*s*y.T) @ H @ (I - rho*y*s.T) + rho*s*s.T
                        # This is O(n^3) which is fine for n=5.
                        
                        # I will implement it in the next step. For now, let's just break and fall back to NM if this gets complicated.
                        # Actually, I can just use `scipy.optimize.fminbfgs`? No, it's a black box, but I can use the `func` inside.
                        # But I need to manage the budget.
                        # Let's stick to the code I wrote above but fix the update formula.
                        
                        # Correct BFGS update for H_inv:
                        # H_new = (I - rho * s @ y.T) @ H @ (I - rho * y @ s.T) + rho * np.outer(s, s)
                        # where rho = 1 / (y @ s)
                        
                        # I already computed rho above? No, I used rho for line search. Let's use tau.
                        tau = 1.0 / sy
                        I = np.eye(n)
                        H = (I - tau * np.outer(s, y)) @ H @ (I - tau * np.outer(y, s)) + tau * np.outer(s, s)
                        # Ensure symmetry
                        H = (H + H.T) / 2

                g_prev = grad.copy()
                x_prev = x.copy()
                x = x_new
                f = f_new
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            return x, f

        def cma_es(x0, f0, max_gen=100):
            """CMA-ES with small population for exploration."""
            nonlocal calls, best_x, best_f

            lam = 8
            mu = 4
            w = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(lam, mu, -1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w ** 2)

            sigma = 1.0
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            Csqrt = np.eye(n)
            Csqrt_inv = np.eye(n)

            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = x0.copy()
            f_best_local = f0
            x_best_local = x0.copy()

            # Initial population
            pop_x = [x0.copy()]
            pop_f = [f0]

            for i in range(1, lam):
                if calls >= budget or time.time() > deadline:
                    return x_best_local, f_best_local
                z = rng.standard_normal(n)
                xi = x_mean + sigma * (Csqrt @ z)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                if fi < f_best_local:
                    f_best_local = fi
                    x_best_local = xi.copy()
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                pop_x.append(xi)
                pop_f.append(fi)

            if calls >= budget:
                return x_best_local, f_best_local

            for gen in range(max_gen):
                if time.time() > deadline or calls >= budget:
                    break

                if sigma < 1e-8:
                    break

                order = np.argsort(pop_f)
                pop_x = [pop_x[i] for i in order]
                pop_f = [pop_f[i] for i in order]

                # Update mean
                m = np.zeros(n)
                for i in range(mu):
                    m += w[i] * pop_x[i]

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (sigma * chiN) * (Csqrt_inv @ (m - x_mean))

                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) > (1.4 + 2 / (n + 1)) * chiN:
                    sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1) / 2)
                else:
                    if np.linalg.norm(ps) / chiN < (1.4 + 2 / (n + 1)):
                        pc = (1 - cc) * pc + cc * (Csqrt_inv @ (m - x_mean) / sigma)
                    else:
                        pc = (1 - cc) * pc + (1 + cc) * (Csqrt_inv @ (m - x_mean) / sigma)

                    hsig = min(1, (np.linalg.norm(ps) / chiN) ** 2 / (1 - (1 - cs) ** (2 * (gen + 1))))
                    rank_one = np.outer(pc, pc)
                    rank_one[pc < 0] *= (1 + (2 / (n + 1) - 1) * hsig)
                    C = (1 + c1 * (hsig - 1)) * C + c1 * rank_one
                    rank_mu = np.zeros((n, n))
                    for i in range(mu):
                        z = Csqrt_inv @ (pop_x[i] - m) / sigma
                        rank_mu += w[i] * np.outer(z, z)
                    C += cmu * rank_mu

                    C = (C + C.T) / 2
                    C += 1e-12 * np.eye(n)

                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-15)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                        Csqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                        Csqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        Csqrt = np.eye(n)
                        Csqrt_inv = np.eye(n)

                x_mean = m

                # Generate new population
                new_pop_x = []
                new_pop_f = []
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        return x_best_local, f_best_local
                    z = rng.standard_normal(n)
                    xi = x_mean + sigma * (Csqrt @ z)
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls += 1
                    if fi < f_best_local:
                        f_best_local = fi
                        x_best_local = xi.copy()
                        if fi < best_f:
                            best_f = fi
                            best_x = xi.copy()
                    new_pop_x.append(xi)
                    new_pop_f.append(fi)

                pop_x = new_pop_x
                pop_f = new_pop_f

                # Convergence check
                if pop_f[-1] - pop_f[0] < 1e-10:
                    break

            return x_best_local, f_best_local

        def nelder_mead(x0, f0, max_iter=500):
            """Nelder-Mead simplex method for final polish."""
            nonlocal calls, best_x, best_f

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append(xi)
                fvals.append(fi)

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[-1] - fvals[0] < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget or time.time() > deadline:
                                return simplex[0], fvals[0]

                # Track best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            min_idx = np.argmin(fvals)
            return simplex[min_idx], fvals[min_idx]

        # Phase 1: Evaluate origin and a few random points
        f_origin = func(np.zeros(n))
        calls += 1
        best_x = np.zeros(n).copy()
        best_f = f_origin

        for i in range(5):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, n)
            f_try = func(x_try)
            calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES for exploration (use ~30% of remaining budget)
        remaining = budget - calls
        cma_budget = int(0.3 * remaining)
        max_gen = max(15, cma_budget // 8)
        x_cma, f_cma = cma_es(best_x.copy(), best_f, max_gen=max_gen)

        if f_cma < best_f:
            best_f = f_cma
            best_x = x_cma.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: BFGS for fast local convergence
        x_grad, f_grad = bfgs(best_x.copy(), best_f, max_iter=100)

        if f_grad < best_f:
            best_f = f_grad
            best_x = x_grad.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Nelder-Mead for final polish
        x_nm, f_nm = nelder_mead(best_x.copy(), best_f, max_iter=200)

        if f_nm < best_f:
            best_f = f_nm
            best_x = x_nm.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 5: Multiple restarts with remaining budget
        restarts_done = 0
        max_restarts = 3
        while restarts_done < max_restarts and budget - calls > 60:
            x_restart = best_x + 0.3 * rng.standard_normal(n)
            x_restart = np.clip(x_restart, -5, 5)
            f_restart = func(x_restart)
            calls += 1
            if f_restart < best_f:
                best_f = f_restart
                best_x = x_restart.copy()

            x_cma2, f_cma2 = cma_es(x_restart, f_restart, max_gen=15)
            if f_cma2 < best_f:
                best_f = f_cma2
                best_x = x_cma2.copy()

            if calls < budget and time.time() < deadline:
                x_grad2, f_grad2 = gradient_descent_adaptive(best_x.copy(), best_f, max_iter=300)
                if f_grad2 < best_f:
                    best_f = f_grad2
                    best_x = x_grad2.copy()

            restarts_done += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
