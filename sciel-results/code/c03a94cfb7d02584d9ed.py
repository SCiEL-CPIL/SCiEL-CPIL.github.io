"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start optimizer: global exploration + Nelder-Mead local polish.
    
    Strategy:
    1. Phase 1: Global exploration using random and quasi-random points
       to identify promising regions in the shifted/rotated landscape.
    2. Phase 2: Local refinement using Nelder-Mead simplex method
       starting from the best points found in Phase 1.
    
    Time guard: returns best result before wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 300s kill

        def time_check():
            return (time.time() - start_time) > time_limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration
        # Use a combination of random and quasi-random (Halton-like) points
        n_explore = min(80, budget // 6)
        
        x_list = []
        f_list = []
        
        # Center point
        x0 = np.zeros(dim)
        f0 = func(x0)
        x_list.append(x0.copy())
        f_list.append(f0)
        
        # Random points
        for i in range(n_explore - 1):
            if time_check():
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            x_list.append(x)
            f_list.append(f)
        
        # Quasi-random points (simple Halton-like sequence)
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Add some quasi-random points
        n_quasi = min(20, budget - len(x_list) - 50)
        for i in range(max(0, n_quasi)):
            if time_check():
                break
            x = np.array([halton(i + 100, b) for b in range(2, dim + 2)])
            x = x * 10.0 - 5.0  # Map to [-5, 5]
            f = func(x)
            x_list.append(x)
            f_list.append(f)

        f_list = np.array(f_list)
        x_list = np.array(x_list)
        
        # Sort by fitness and take top candidates for local refinement
        order = np.argsort(f_list)
        top_k = min(10, len(x_list))
        best_starts = x_list[order[:top_k]]
        best_f = f_list[order[0]]
        best_x = x_list[order[0]].copy()

        # Phase 2: Local refinement with Nelder-Mead
        remaining_budget = budget - len(x_list)
        
        if remaining_budget > 20 and not time_check():
            # Try Nelder-Mead from the best few starting points
            for i in range(min(3, top_k)):
                if time_check():
                    break
                if budget - len(x_list) < 30:
                    break
                    
                x_start = best_starts[i].copy()
                
                # Use a simple Nelder-Mead with limited iterations
                # We'll implement a lightweight version to control budget
                def nm_step(x, f, step_size=0.5):
                    """Simple Nelder-Mead step in 5D."""
                    dim = len(x)
                    # Create simplex
                    simplex = [x.copy()]
                    simplex_f = [f]
                    for j in range(dim):
                        x_j = x.copy()
                        x_j[j] += step_size
                        f_j = func(x_j)
                        simplex.append(x_j)
                        simplex_f.append(f_j)
                    
                    # Iterate
                    max_iter = min(50, (budget - len(x_list)) // (dim + 1))
                    for _ in range(max_iter):
                        if time_check():
                            break
                        if len(x_list) >= budget - 5:
                            break
                            
                        # Sort simplex
                        idx = np.argsort(simplex_f)
                        simplex = [simplex[i] for i in idx]
                        simplex_f = [simplex_f[i] for i in idx]
                        
                        # Centroid of all but worst
                        centroid = np.mean(simplex[:-1], axis=0)
                        worst = simplex[-1]
                        worst_f = simplex_f[-1]
                        
                        # Reflection
                        x_ref = centroid + (centroid - worst)
                        x_ref = np.clip(x_ref, -5, 5)
                        f_ref = func(x_ref)
                        
                        if f_ref < simplex_f[0]:
                            # Expansion
                            x_exp = centroid + 2 * (centroid - worst)
                            x_exp = np.clip(x_exp, -5, 5)
                            f_exp = func(x_exp)
                            if f_exp < f_ref:
                                simplex[-1] = x_exp
                                simplex_f[-1] = f_exp
                            else:
                                simplex[-1] = x_ref
                                simplex_f[-1] = f_ref
                        elif f_ref < simplex_f[-2]:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                        else:
                            # Contraction
                            x_con = centroid + 0.5 * (worst - centroid)
                            x_con = np.clip(x_con, -5, 5)
                            f_con = func(x_con)
                            if f_con < worst_f:
                                simplex[-1] = x_con
                                simplex_f[-1] = f_con
                            else:
                                # Shrink
                                for j in range(1, dim + 1):
                                    simplex[j] = (simplex[0] + simplex[j]) / 2
                                    simplex_f[j] = func(simplex[j])
                        
                        # Update best
                        if simplex_f[0] < best_f:
                            best_f = simplex_f[0]
                            best_x = simplex[0].copy()
                        
                        x_list.append(simplex[-1].copy())
                        f_list.append(simplex_f[-1])
                    
                    # After Nelder-Mead, do a few more random restarts if budget allows
                    if len(x_list) < budget - 10 and not time_check():
                        # Fine-tune around best point
                        for _ in range(min(20, budget - len(x_list) - 5)):
                            if time_check():
                                break
                            if len(x_list) >= budget - 3:
                                break
                            x_try = best_x + rng.normal(0, 0.3, dim)
                            x_try = np.clip(x_try, -5, 5)
                            f_try = func(x_try)
                            if f_try < best_f:
                                best_f = f_try
                                best_x = x_try.copy()
                            x_list.append(x_try)
                            f_list.append(f_try)

        # Final: use remaining budget for local search around best point
        remaining = budget - len(x_list)
        if remaining > 0 and not time_check():
            # Fine-grained local search
            step = 0.1
            for _ in range(min(remaining, 50)):
                if time_check():
                    break
                if len(x_list) >= budget:
                    break
                # Try perturbations in each dimension
                improved = False
                for d in range(dim):
                    if len(x_list) >= budget:
                        break
                    for sign in [-1, 1]:
                        if len(x_list) >= budget:
                            break
                        x_try = best_x.copy()
                        x_try[d] += sign * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                            improved = True
                        x_list.append(x_try)
                        f_list.append(f_try)
                if improved:
                    step *= 0.7
                else:
                    step *= 0.5
                    if step < 0.01:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
