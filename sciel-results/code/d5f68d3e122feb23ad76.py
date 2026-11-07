"""Optimized program for BBOB f21 (Gallagher 101 peaks) with Basin Hopping and Nelder-Mead portfolio."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

class Optimizer:
    """
    Portfolio of Basin Hopping chains.
    Each chain uses Nelder-Mead for local minimization and random jumps for global exploration.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return 1e18
            n_calls += 1
            # Ensure x is a valid numpy array of correct shape
            x = np.asarray(x, dtype=np.float64)
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Nelder-Mead implementation (simplified, in-place)
        def nelder_mead(func, x0, max_iter=50, tol=1e-6):
            """
            Minimizes func(x) starting from x0.
            Returns (best_x, best_f, calls_made)
            """
            n = len(x0)
            # Initialize simplex
            simplex = []
            simplex.append(x0.copy())
            for i in range(n):
                xi = x0.copy()
                # Standard NM initialization
                if abs(xi[i]) > 1e-8:
                    xi[i] *= 1.05
                else:
                    xi[i] += 0.00025
                simplex.append(xi)
            
            # Evaluate initial simplex
            f_values = []
            calls = 0
            for i in range(n + 1):
                if n_calls >= budget or time.time() > deadline - 2:
                    break
                f_values.append(wrapped_func(simplex[i]))
                calls += 1
            
            if len(f_values) < n + 1:
                # Budget exhausted during init
                min_idx = np.argmin(f_values)
                return simplex[min_idx], f_values[min_idx], calls

            best_idx = np.argmin(f_values)
            best_f_val = f_values[best_idx]
            best_x_val = simplex[best_idx].copy()

            for _ in range(max_iter):
                if n_calls >= budget or time.time() > deadline - 2:
                    break
                
                # Sort simplex
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Check convergence
                if f_values[-1] - f_values[0] < tol:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                fr = wrapped_func(xr)
                calls += 1
                if n_calls >= budget: break

                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    fe = wrapped_func(xe)
                    calls += 1
                    if n_calls >= budget: break
                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                elif fr < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_values[-1] = fr
                else:
                    # Contraction
                    if fr < f_values[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        fc = wrapped_func(xc)
                        calls += 1
                        if n_calls >= budget: break
                        if fc <= fr:
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                if n_calls >= budget: break
                                f_values[i] = wrapped_func(simplex[i])
                                calls += 1
                            if n_calls >= budget: break
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        fc = wrapped_func(xc)
                        calls += 1
                        if n_calls >= budget: break
                        if fc < f_values[-1]:
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                if n_calls >= budget: break
                                f_values[i] = wrapped_func(simplex[i])
                                calls += 1
                            if n_calls >= budget: break

                # Update global best for this chain
                min_idx = np.argmin(f_values)
                if f_values[min_idx] < best_f_val:
                    best_f_val = f_values[min_idx]
                    best_x_val = simplex[min_idx].copy()

            return best_x_val, best_f_val, calls

        # Portfolio of Basin Hopping chains
        n_chains = 4
        chains = []
        
        # Initialize chains with diverse starting points
        for i in range(n_chains):
            # Spread starts
            x0 = rng.uniform(-4, 4, dim)
            chains.append({
                'x': x0,
                'active': True,
                'local_best_f': float('inf')
            })
        
        # Initial local optimization for each chain to find basins
        for chain in chains:
            if not chain['active'] or n_calls >= budget or time.time() > deadline - 10:
                continue
            x_opt, f_opt, _ = nelder_mead(wrapped_func, chain['x'], max_iter=20)
            chain['x'] = x_opt
            chain['local_best_f'] = f_opt
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Main loop: Basin Hopping
        # Strategy: Pick a chain, do a local step (NM), then a random jump.
        # We alternate between refining the current best and exploring new areas.
        
        jump_size = 1.5
        min_jump_size = 0.1
        
        while n_calls < budget and time.time() < deadline - 5:
            # Select chain: prioritize those with good local minima or random
            active_chains = [c for c in chains if c['active']]
            if not active_chains:
                break
            
            # Round-robin or best-first? Let's do a mix: 50% best, 50% random
            if rng.random() < 0.5:
                # Pick best chain
                chain = min(active_chains, key=lambda c: c['local_best_f'])
            else:
                chain = rng.choice(active_chains)
            
            # 1. Local Refinement (Nelder-Mead)
            # Limit NM iterations to save budget for jumps
            max_nm_iter = 30 if n_calls < budget * 0.5 else 10
            x_opt, f_opt, _ = nelder_mead(wrapped_func, chain['x'], max_iter=max_nm_iter)
            chain['x'] = x_opt
            chain['local_best_f'] = f_opt
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
            
            if n_calls >= budget or time.time() > deadline - 5:
                break

            # 2. Global Jump (Basin Hopping step)
            # Jump size adapts: if we found a good spot, jump smaller; if stuck, jump bigger
            # Simple adaptation: if local_best_f is close to global best_f, reduce jump
            progress = (best_f - chain['local_best_f']) / max(1e-8, abs(best_f))
            if progress < 0.1:
                # Chain is near global best, do small perturbations to escape
                current_jump = min_jump_size
            else:
                current_jump = jump_size
            
            # Random direction
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            
            x_new = chain['x'] + current_jump * direction
            # Clip to box
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate the jump
            f_new = wrapped_func(x_new)
            if f_new < chain['local_best_f']:
                chain['x'] = x_new
                chain['local_best_f'] = f_new
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            else:
                # Metropolis-like acceptance? 
                # For minimization, we usually only accept if better, 
                # but Basin Hopping often accepts worse to escape.
                # Given tight budget, let's only accept if significantly better or random restart.
                # Let's try: if worse, keep old x but mark for potential restart if stuck.
                # To keep it simple and effective: if jump fails, we just stay.
                # However, pure local search + failed jumps can get stuck.
                # Let's add a restart mechanism if a chain hasn't improved in a while.
                # For now, we just move to the next chain.
                pass

            # Check for timeout
            if time.time() > deadline - 5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
