"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Start Nelder-Mead Optimizer for BBOB f14.
    Strategy:
    1. Global Diversification: Generate a set of diverse starting points (Halton sequence + random) 
       to cover the search space, aiming to find the basin of attraction of the global minimum 
       despite hidden shifts and rotations.
    2. Local Refinement: Run a robust Nelder-Mead simplex method from each promising starting point.
       Nelder-Mead is chosen over (1+1)-ES because it can better handle the anisotropic and 
       potentially ill-conditioned landscape of f14 (Different Powers) by adapting the shape 
       of the simplex, not just the step size.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        rng = np.random.default_rng(42)
        calls_used = 0
        dim = self.dim
        budget = self.budget
        
        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        def evaluate(x):
            nonlocal calls_used
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_used += 1
            return f

        # Phase 1: Generate diverse starting points
        # We allocate ~30% of the budget to global search / initial simplex evaluations.
        # We will create 5 different starting regions.
        
        start_points = []
        
        # 1. Origin
        start_points.append(np.zeros(dim))
        
        # 2. Halton sequence points for space-filling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        num_halton = 4
        for i in range(1, num_halton + 1):
            p = np.array([halton(i, b) for b in range(2, dim + 2)])
            # Map [0, 1)^dim to [-4.5, 4.5]^dim
            p = p * 9.0 - 4.5
            start_points.append(p)
            
        # 3. Random points
        num_random = 5
        for _ in range(num_random):
            p = rng.uniform(-4.5, 4.5, size=dim)
            start_points.append(p)

        # Phase 2: Nelder-Mead from each start point
        # We run NM from each start point. 
        # To manage budget, we give each NM instance a sub-budget.
        # Total budget: 500.
        # Start points: 10.
        # Let's give each NM ~40 calls. 10 * 40 = 400 calls. 
        # This leaves 100 calls for potential re-evaluation or just unused.
        # Actually, NM usually converges fast. Let's allow up to 50 calls per NM, but stop early if converged.
        
        x_best_global = None
        f_best_global = np.inf

        for sp in start_points:
            if check_time():
                break
                
            # Initialize simplex
            simplex = [sp.copy()]
            f_simplex = [evaluate(sp)]
            
            # Create initial simplex around sp
            # Standard deviation for initial simplex
            sigma_init = 0.5
            for i in range(dim):
                v = sp.copy()
                v[i] += sigma_init
                simplex.append(v)
                f_simplex.append(evaluate(v))
                
            # If we ran out of budget during simplex creation, break
            if check_time():
                break
            
            # Track best in this simplex
            f_best_nm = min(f_simplex)
            idx_best = np.argmin(f_simplex)
            x_best_nm = simplex[idx_best].copy()
            
            # NM Loop
            # Max iterations per NM instance
            max_iters = 100 
            # But we are limited by global budget and time
            # Let's set a hard limit on calls per NM to ensure we visit all starts
            calls_in_nm = dim + 1
            max_calls_per_nm = 45 # Roughly 450 total for 10 starts
            
            for _ in range(max_iters):
                if check_time():
                    break
                if calls_in_nm >= max_calls_per_nm:
                    break
                
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check convergence
                if f_simplex[-1] - f_simplex[0] < 1e-8:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)
                calls_in_nm += 1
                
                if f_simplex[0] <= f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                elif f_r < f_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    f_e = evaluate(x_e)
                    calls_in_nm += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    
                    f_c = evaluate(x_c)
                    calls_in_nm += 1
                    
                    if f_c < min(f_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = evaluate(simplex[i])
                            calls_in_nm += 1
                            if check_time():
                                break
                        if check_time():
                            break
                
                # Update local best
                if f_simplex[0] < f_best_nm:
                    f_best_nm = f_simplex[0]
                    x_best_nm = simplex[0].copy()

            # Update global best
            if f_best_nm < f_best_global:
                f_best_global = f_best_nm
                x_best_global = x_best_nm

        # If no points were evaluated (should not happen), fallback
        if x_best_global is None:
            x_best_global = np.zeros(dim)
            f_best_global = func(x_best_global)
            calls_used += 1

        return x_best_global, f_best_global
# EVOLVE-BLOCK-END
