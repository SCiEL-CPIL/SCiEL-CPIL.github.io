"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _sobol_point(dim, n, i):
    """
    Generate a deterministic point in [0, 1)^dim using a simplified Sobol-like 
    low-discrepancy sequence (van der Corput for each dimension with a twist).
    This ensures diverse, broad coverage without needing SciPy.
    """
    point = np.zeros(dim)
    for d in range(dim):
        # Use different bases or shifts to decorrelate dimensions
        # van der Corput sequence with base 2 is standard Sobol for 1D
        # To make it 2D/3D etc, we can use different primes or offsets.
        # Simple approach: use van der Corput with base (d+2)
        base = d + 2
        val = 0.0
        f = 1.0 / base
        n_val = i + 1 # 1-based index for sequence
        while n_val > 0:
            val += f * (n_val % base)
            n_val //= base
            f /= base
        point[d] = val
    return point

class Optimizer:
    """Deterministic Nelder-Mead with Sobol initialization and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        
        # 1. Generate a set of diverse starting points using low-discrepancy sequence
        # We want to use the budget efficiently. 
        # Nelder-Mead is fast for smooth functions. 
        # Let's pick a good starting point. 
        # The optimum is in [-4, 4]^5. 
        # A center point (0,0,0,0,0) is a good bet, but let's use a slightly off-center 
        # point from a Sobol sequence to avoid potential symmetries or flat spots, 
        # or just start at 0 if we suspect the hidden shift is small? 
        # The problem says "hidden seeded shift, rotation and offset". 
        # The minimizer is in [-4, 4]^5.
        # Starting at 0 is risky if the shift is large. 
        # Let's use the 10th point of a Sobol sequence mapped to [-5, 5]^5.
        
        # Actually, for AOCC, early progress is key.
        # Nelder-Mead converges very fast once in the basin of attraction.
        # If we miss the basin, we fail.
        # (1+1)-ES explores globally.
        # Nelder-Mead is local.
        # We should probably do a very short global search or a multi-start.
        # But the prompt suggests "deterministic multi-start local search... followed by sequential refinement".
        # With 500 budget, we can afford maybe 10-20 starts if NM is fast, or 1 start if NM is robust.
        # Given "Sharp ridge", NM might struggle if the ridge is sharp and narrow? 
        # Actually, NM handles ridges okay if initialized well.
        # Let's try a strategy: 
        # 1. Evaluate a few Sobol points (e.g., 5 points).
        # 2. Pick the best one.
        # 3. Run Nelder-Mead from there until budget/time runs out.
        
        # Let's generate 5 diverse points.
        candidates = []
        for i in range(5):
            p_unit = _sobol_point(dim, 5, i)
            p = p_unit * 10.0 - 5.0  # Map [0,1] to [-5,5]
            candidates.append(p)
            
        # Evaluate candidates
        best_x = None
        best_f = np.inf
        calls_made = 0
        for p in candidates:
            if time.time() > deadline:
                break
            f = func(p)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = p.copy()
                
        if best_x is None:
            # Fallback if we ran out of time immediately (unlikely)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # 2. Nelder-Mead Refinement
        # Initialize simplex around best_x
        # Standard NM initialization:
        # x0 is the best point.
        # Other vertices are x0 +/- delta * e_i
        delta = 0.5 # Initial step size
        simplex = [best_x.copy()]
        for i in range(dim):
            xi = best_x.copy()
            xi[i] += delta
            simplex.append(np.clip(xi, -5, 5))
            
        # Evaluate initial simplex
        simplex_f = [best_f]
        for i in range(1, dim + 1):
            if time.time() > deadline or calls_made >= self.budget:
                break
            f = func(simplex[i])
            calls_made += 1
            simplex_f.append(f)
            
        # Main NM loop
        # We have used 5 + dim calls so far. 
        # Remaining budget = 500 - (5 + dim)
        # NM typically takes O(N^2) to O(N^3) iterations. For N=5, it's fast.
        
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by fitness (ascending)
            # indices sorted by f
            sorted_indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in sorted_indices]
            simplex_f = [simplex_f[i] for i in sorted_indices]
            
            # x0 is best, xN is worst
            x_best = simplex[0]
            f_best = simplex_f[0]
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            
            # Centroid of all but worst
            x_c = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_r = x_c + alpha * (x_c - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < f_best:
                # 2. Expansion
                x_e = x_c + gamma * (x_r - x_worst) # Wait, expansion is from centroid towards reflection?
                # Standard: x_e = x_c + gamma * (x_r - x_c)
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]: # f_r is better than 2nd worst
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # 3. Contraction
                # Outside or Inside contraction?
                # If f_r >= f_worst, do outside contraction. Else inside.
                if f_r < f_worst:
                    # Inside contraction
                    x_c_i = x_c + rho * (x_worst - x_c)
                    x_c_i = np.clip(x_c_i, -5, 5)
                    f_c = func(x_c_i)
                    calls_made += 1
                else:
                    # Outside contraction
                    x_c_o = x_c - rho * (x_worst - x_c)
                    x_c_o = np.clip(x_c_o, -5, 5)
                    f_c = func(x_c_o)
                    calls_made += 1
                    
                if f_c < f_r:
                    simplex[-1] = x_c_i if f_r < f_worst else x_c_o
                    simplex_f[-1] = f_c
                else:
                    # 4. Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if time.time() > deadline or calls_made >= self.budget:
                            break
                        f_s = func(simplex[i])
                        calls_made += 1
                        simplex_f[i] = f_s
                        
            # Check for convergence (optional, but helps stop early if stuck)
            # If simplex is very small, we might be converged.
            # However, we want to use budget if possible? 
            # Actually, if we are converged, further calls don't help AOCC much if f_opt is known.
            # But we don't know f_opt.
            # If we stop early, we repeat the last n_i.
            # So stopping early is fine if we are close to optimum.
            
            # Convergence check: spread of simplex
            if np.max(simplex_f) - np.min(simplex_f) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
