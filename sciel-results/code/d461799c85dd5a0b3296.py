"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer: Random Sampling + Nelder-Mead Local Search + Iterated Local Search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin
        budget = self.budget
        dim = self.dim
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        rng = np.random.default_rng(42)

        def eval_point(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return None
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Broad Random Sampling
        # Sample 20 random points to find a promising basin
        n_samples = 20
        for _ in range(n_samples):
            if calls >= budget or time.time() > deadline:
                break
            x_rand = rng.uniform(-5.0, 5.0, size=dim)
            eval_point(x_rand)
        
        # Also evaluate origin and a few structured points
        eval_point(np.zeros(dim))
        for d in range(dim):
            x_pos = np.zeros(dim); x_pos[d] = 4.0
            x_neg = np.zeros(dim); x_neg[d] = -4.0
            eval_point(x_pos)
            eval_point(x_neg)
            if calls >= budget: break

        if best_x is None or np.isnan(best_f):
            best_x = np.zeros(dim)
            best_f = eval_point(best_x)
            if best_f is None:
                return best_x, 0.0

        # Phase 2: Nelder-Mead Local Search
        # Implement a simple Nelder-Mead to refine the best point found
        
        def nelder_mead_local(x0, max_iter=50):
            nonlocal calls, best_x, best_f
            
            # Initialize simplex
            # x0 is the best point
            simplex = [x0.copy()]
            f_vals = [best_f]
            
            # Generate other simplex points by perturbing x0
            step = 0.5
            for d in range(dim):
                x_pert = x0.copy()
                x_pert[d] += step
                x_pert = np.clip(x_pert, -5.0, 5.0)
                f_pert = eval_point(x_pert)
                if f_pert is None:
                    return
                simplex.append(x_pert)
                f_vals.append(f_pert)
            
            # Sort simplex by function value
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = eval_point(x_reflect)
                if f_reflect is None:
                    break
                
                if f_reflect < f_vals[0]:
                    # Try expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = eval_point(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_vals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_vals[-1] = f_reflect
                elif f_reflect < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    f_vals[-1] = f_reflect
                else:
                    # Try contraction
                    x_contract = centroid + rho * (simplex[-1] - centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = eval_point(x_contract)
                    if f_contract is None:
                        break
                    if f_contract < f_vals[-1]:
                        simplex[-1] = x_contract
                        f_vals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_shrink = eval_point(simplex[i])
                            if f_shrink is None:
                                return
                            f_vals[i] = f_shrink
                
                # Re-sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Check convergence
                if max(f_vals) - min(f_vals) < 1e-8:
                    break

        # Run Nelder-Mead from the current best
        nelder_mead_local(best_x.copy(), max_iter=100)

        # Phase 3: Iterated Local Search (ILS)
        # If we have budget left, try to escape local minima
        remaining_budget = budget - calls
        if remaining_budget > 50:
            # Perturb the best point and try again
            for i in range(5):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Perturb with Gaussian noise
                sigma_perturb = 1.0 * (0.8 ** i)
                x_perturb = best_x + sigma_perturb * rng.standard_normal(dim)
                x_perturb = np.clip(x_perturb, -5.0, 5.0)
                f_perturb = eval_point(x_perturb)
                if f_perturb is None:
                    break
                
                if f_perturb < best_f:
                    best_f = f_perturb
                    best_x = x_perturb.copy()
                
                # Run a short Nelder-Mead from this new point
                nelder_mead_local(best_x.copy(), max_iter=30)
                
                if time.time() > deadline:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
