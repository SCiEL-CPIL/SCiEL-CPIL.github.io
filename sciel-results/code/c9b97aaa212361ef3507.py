"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Strategy:
  1. Fast (1+1)-ES with aggressive sigma adaptation for global convergence.
  2. Nelder-Mead local polish with adaptive simplex sizing.
  3. Final short (1+1)-ES refinement with very small sigma.

The unimodal, smooth nature of f14 (preserved under affine transforms) makes
(1+1)-ES highly effective: it converges quickly with minimal overhead per call,
freeing budget for local refinement.
"""
import time
import numpy as np


class Optimizer:
    """(1+1)-ES + Nelder-Mead polish optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 305.0
        rng = np.random.default_rng(12345)

        dim = self.dim
        budget = self.budget
        calls = 0

        def elapsed():
            return time.time() - start

        def still_ok():
            return elapsed() < deadline

        def make_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            fx = float(func(x))
            calls += 1
            return fx

        # Phase 0: Evaluate origin
        x0 = np.zeros(dim)
        fx0 = make_call(x0)
        if fx0 is None:
            return x0, np.inf

        best_x = x0.copy()
        best_f = fx0

        # ---- Phase 1: (1+1)-ES global search ----
        def run_11es(x_init, f_init, sigma_init, max_calls):
            nonlocal calls, best_x, best_f
            x = x_init.copy()
            f = f_init
            sigma = sigma_init
            c_sigma = 1.0 / (dim + 3)
            d_sigma = 1 + 2.0 / max(1, dim - 1) + c_sigma
            p_sigma = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            
            for _ in range(max_calls):
                if not still_ok() or calls >= max_calls:
                    break
                
                # Sample offspring
                y = rng.standard_normal(dim)
                x_child = x + sigma * y
                x_child = np.clip(x_child, -5.0, 5.0)
                f_child = make_call(x_child)
                if f_child is None:
                    break
                
                # Update evolution path
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * y
                
                # Accept or reject
                if f_child < f:
                    x = x_child
                    f = f_child
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                else:
                    sigma *= np.exp(-c_sigma / (2 * d_sigma))
                
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Track global best
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                
                # Convergence check
                if sigma < 1e-8:
                    break
            
            return x, f, sigma

        # Allocate ~45% to global ES
        es_budget = max(60, int(budget * 0.45))
        x, f, sigma = run_11es(best_x, best_f, 2.0, calls + es_budget)

        # ---- Phase 2: Nelder-Mead local polish ----
        if still_ok() and calls < budget:
            x_best = best_x.copy()
            f_best = best_f
            
            # Adaptive simplex size based on current sigma
            simplex_size = max(0.1, min(1.0, sigma * 0.5))
            simplex = [x_best.copy()]
            for i in range(dim):
                vertex = x_best.copy()
                vertex[i] += simplex_size
                simplex.append(np.clip(vertex, -5.0, 5.0))
            
            # Evaluate simplex vertices
            simplex_f = []
            for v in simplex:
                fv = make_call(v)
                if fv is None:
                    break
                simplex_f.append(fv)
                if fv < f_best:
                    f_best = fv
                    x_best = v.copy()
                    best_f = fv
                    best_x = v.copy()
            else:
                simplex_f = np.array(simplex_f)
                simplex = np.array(simplex)
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5
                
                nm_max_iter = 300
                nm_iter = 0
                
                while nm_iter < nm_max_iter and still_ok() and calls < budget:
                    nm_iter += 1
                    order = np.argsort(simplex_f)
                    simplex = simplex[order]
                    simplex_f = simplex_f[order]
                    
                    if np.max(simplex_f) - np.min(simplex_f) < 1e-12:
                        break
                    
                    x_worst = simplex[-1]
                    f_worst = simplex_f[-1]
                    x_best_nm = simplex[0]
                    f_best_nm = simplex_f[0]
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_ref = centroid + alpha * (centroid - x_worst)
                    x_ref = np.clip(x_ref, -5.0, 5.0)
                    f_ref = make_call(x_ref)
                    if f_ref is None:
                        break
                    
                    if f_ref < f_best_nm:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = make_call(x_exp)
                        if f_exp is None:
                            break
                        if f_exp < f_ref:
                            x_new, f_new = x_exp, f_exp
                        else:
                            x_new, f_new = x_ref, f_ref
                        simplex[-1] = x_new
                        simplex_f[-1] = f_new
                    elif f_ref < simplex_f[-2]:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + rho * (x_worst - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = make_call(x_con)
                        if f_con is None:
                            break
                        if f_con < f_worst:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best_nm + sigma_nm * (simplex[i] - x_best_nm)
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_sh = make_call(simplex[i])
                                if f_sh is None:
                                    break
                                simplex_f[i] = f_sh
                            else:
                                pass
                            break
                    
                    for i in range(dim + 1):
                        if simplex_f[i] < f_best:
                            f_best = simplex_f[i]
                            x_best = simplex[i].copy()
                            best_f = f_best
                            best_x = x_best.copy()

        # ---- Phase 3: Final (1+1)-ES refinement ----
        if still_ok() and calls < budget:
            x, f, sigma = run_11es(best_x, best_f, 0.05, budget)

        return best_x, float(best_f)
