"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Nelder-Mead Optimizer with Multi-Start Strategy.
    
    Strategy:
    1. Global Phase: Use a small set of deterministic Sobol points to identify potential basins.
    2. Local Phase: Run Nelder-Mead from the best point found.
    3. Restart Phase: If budget remains and Nelder-Mead stagnates, restart from the second-best 
       Sobol point with a fresh simplex, effectively implementing a multi-start strategy 
       without the overhead of a full population-based optimizer.
    
    Nelder-Mead is chosen for its efficiency in low-dimensional spaces (dim=5) where it can 
    converge to local minima in significantly fewer evaluations than CMA-ES, especially 
    when the landscape is smooth enough to follow simplex geometry.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, n, dim, seed=12345):
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
            points = sampler.random(n)
            return points * 10 - 5
        except ImportError:
            rng = np.random.default_rng(seed)
            return rng.uniform(-5, 5, (n, dim))

    def _nelder_mead(self, func, x0, calls_state, deadline, max_iter):
        """
        Runs Nelder-Mead optimization starting from x0.
        Returns (best_x, best_fx).
        """
        dim = self.dim
        # Initialize simplex
        simplex = []
        f_simplex = []
        
        # Start with x0
        x0 = np.clip(x0, -5, 5)
        f0 = func(x0)
        calls_state[0] += 1
        simplex.append(x0.copy())
        f_simplex.append(f0)
        
        # Create other vertices with a small perturbation
        # Standard Nelder-Mead uses alpha * delta in each dimension
        # We use a relative step size based on the domain
        delta = 0.5 
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += delta
            x_i = np.clip(x_i, -5, 5)
            f_i = func(x_i)
            calls_state[0] += 1
            simplex.append(x_i)
            f_simplex.append(f_i)
            
        # Sort simplex by fitness
        def sort_simplex():
            indices = np.argsort(f_simplex)
            simplex[:] = [simplex[i] for i in indices]
            f_simplex[:] = [f_simplex[i] for i in indices]
            
        sort_simplex()
        
        best_x = simplex[0].copy()
        best_fx = f_simplex[0]
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        for _ in range(max_iter):
            if time.time() > deadline or calls_state[0] >= self.budget:
                break
            
            # Check for convergence
            if max(f_simplex) - min(f_simplex) < 1e-8:
                break
                
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_state[0] += 1
            
            if f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_state[0] += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                if f_r < f_simplex[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_state[0] += 1
                    
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls_state[0] += 1
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_state[0] += 1
                    
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls_state[0] += 1
            
            sort_simplex()
            
            if f_simplex[0] < best_fx:
                best_fx = f_simplex[0]
                best_x = simplex[0].copy()
                
        return best_x, best_fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s safety margin
        dim = self.dim
        budget = self.budget
        
        calls_state = [0]
        
        # Phase 1: Global Exploration
        # Use 20 points to find good starting candidates
        n_global = 20
        sobol_points = self._sobol_sequence(n_global, dim, seed=42)
        
        candidates = []
        for i in range(n_global):
            if time.time() > deadline or calls_state[0] >= budget:
                break
            x = sobol_points[i]
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_state[0] += 1
            candidates.append((fx, x))
            
        if not candidates:
            x0 = np.zeros(dim)
            fx0 = func(x0)
            calls_state[0] += 1
            return x0, fx0

        # Sort candidates
        candidates.sort(key=lambda item: item[0])
        
        best_x = candidates[0][1].copy()
        best_fx = candidates[0][0]
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Try Nelder-Mead on the best candidate
        # Allocate remaining budget to Nelder-Mead
        # Nelder-Mead uses dim+1 evaluations per iteration roughly.
        # We run it until budget is nearly exhausted or it converges.
        
        # We will attempt Nelder-Mead on the top 2 candidates if budget allows,
        # but usually the first one is best.
        
        remaining_budget = budget - calls_state[0]
        if remaining_budget > 10:
            # Run Nelder-Mead
            # Max iterations is roughly remaining_budget / (dim+1) to be safe
            max_iter = remaining_budget // (dim + 2)
            x_opt, f_opt = self._nelder_mead(func, candidates[0][1], calls_state, deadline, max_iter)
            
            if f_opt < best_fx:
                best_fx = f_opt
                best_x = x_opt.copy()
            
            # If there is still significant budget left and we haven't converged perfectly,
            # try the second best candidate?
            # For 5D, Nelder-Mead is quite efficient. Let's stick to the best one to maximize depth.
            # However, if the first one stagnates early, we could restart.
            # For simplicity and robustness, we just run it as long as budget allows.
            
            # Check if we should try a second start if the first one didn't use much budget
            # and the gap to the second candidate is small?
            # Given the "21 peaks" nature, a multi-start is safer.
            # Let's implement a simple multi-start:
            # If calls_state[0] < budget - 50, try the next candidate.
            
            if calls_state[0] < budget - 50 and len(candidates) > 1:
                # Try second best candidate
                # Reset max_iter based on remaining
                remaining_budget = budget - calls_state[0]
                max_iter = remaining_budget // (dim + 2)
                if max_iter > 5:
                    x_opt2, f_opt2 = self._nelder_mead(func, candidates[1][1], calls_state, deadline, max_iter)
                    if f_opt2 < best_fx:
                        best_fx = f_opt2
                        best_x = x_opt2.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
