"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step sizes and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        
        # We'll run multiple short runs with restarts if stagnation occurs
        # Each run uses a small population CMA-ES-like approach
        
        best_x = None
        best_f = np.inf
        
        # Initial exploration: try a few random points to get a good start
        # But we need to be careful with budget. Let's do 10 initial points.
        init_count = min(10, total_budget // 10)
        if init_count == 0:
            init_count = 1
            
        candidates = []
        for _ in range(init_count):
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        remaining_budget = total_budget - init_count
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Now run a local optimizer starting from the best point
        # Use a simple adaptive step size local search with momentum-like behavior
        # or a small population CMA-ES
        
        # Let's implement a simple (mu+lambda)-ES with adaptive sigma
        x = best_x.copy()
        fx = best_f
        
        # Set initial sigma based on problem scale
        sigma = 1.0
        
        # Population size
        # Keep it small to conserve budget
        # Let's use lambda = 4, mu = 2
        lam = 4
        mu = 2
        
        # Success rule: if best offspring is better than parent, increase sigma, else decrease
        # But with multiple offspring, we can be more sophisticated.
        
        # Actually, let's try a different approach: 
        # Use a random restart local search with multiple directions.
        # Or, better, use a simple CMA-ES with small population.
        
        # Let's stick to a robust (1+1)-ES but with better sigma adaptation
        # and add random restarts if no improvement for a while.
        
        # Actually, the current seed is (1+1)-ES. Let's try a different family:
        # Nelder-Mead simplex method? It's derivative-free and works well in low dim.
        # But Nelder-Mead can be sensitive to initial simplex.
        
        # Let's try Nelder-Mead with adaptive simplex resizing.
        
        # Initialize simplex around the best point found
        # Simplex has n+1 points
        # Start with the best point and n other points offset by small steps
        
        # But first, let's make sure we have a good starting point.
        # The initial 10 random points give us a decent start.
        
        # Nelder-Mead implementation
        # Parameters:
        # alpha = 1 (reflection)
        # gamma = 2 (expansion)
        # rho = 0.5 (contraction)
        # sigma = 0.5 (shrink)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5
        
        # Create initial simplex
        # Point 0 is the best point
        # Points 1..n are best point + delta * e_i
        # Choose delta based on the spread of the initial random points
        # or just use a fixed small value like 0.5
        
        # Let's compute a reasonable step size from the initial candidates
        # If we have diverse candidates, use their spread
        # Otherwise, use 1.0
        
        if init_count > 1:
            # Estimate scale from the spread of the initial points
            # Use the average distance from the best point to other points
            dists = []
            for f, x in candidates[1:]:
                d = np.linalg.norm(x - best_x)
                if d > 1e-6:
                    dists.append(d)
            if dists:
                delta = np.median(dists) * 0.5
            else:
                delta = 1.0
        else:
            delta = 1.0
            
        # Clamp delta to reasonable range
        delta = np.clip(delta, 0.1, 2.0)
        
        simplex = [best_x.copy()]
        for i in range(n):
            p = best_x.copy()
            p[i] += delta
            p = np.clip(p, -5, 5)
            simplex.append(p)
        
        # Evaluate all simplex points
        simplex_f = []
        for p in simplex:
            fp = func(p)
            simplex_f.append(fp)
        
        remaining_budget -= (n + 1)
        if remaining_budget <= 0:
            # Find best in simplex
            idx = np.argmin(simplex_f)
            best_x = simplex[idx].copy()
            best_f = simplex_f[idx]
            return best_x, best_f
        
        # Nelder-Mead iterations
        no_improve_count = 0
        best_simplex_f = min(simplex_f)
        
        while remaining_budget > 0:
            # Sort simplex by fitness
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Check if we've made progress
            if simplex_f[0] < best_simplex_f - 1e-10:
                best_simplex_f = simplex_f[0]
                best_x = simplex[0].copy()
                best_f = simplex_f[0]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # If no improvement for too long, restart
            if no_improve_count > 50:
                # Random restart: pick a new random point
                x_new = rng.uniform(-5, 5, n)
                fx_new = func(x_new)
                remaining_budget -= 1
                if fx_new < best_f:
                    best_x = x_new.copy()
                    best_f = fx_new
                    # Reinitialize simplex around new best
                    simplex = [best_x.copy()]
                    for i in range(n):
                        p = best_x.copy()
                        p[i] += delta
                        p = np.clip(p, -5, 5)
                        simplex.append(p)
                    simplex_f = []
                    for p in simplex:
                        fp = func(p)
                        simplex_f.append(fp)
                        remaining_budget -= 1
                    no_improve_count = 0
                    best_simplex_f = min(simplex_f)
                    if remaining_budget <= 0:
                        break
                    continue
            
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fxr = func(xr)
            remaining_budget -= 1
            
            if fxr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fxe = func(xe)
                remaining_budget -= 1
                if fxe < fxr:
                    simplex[-1] = xe
                    simplex_f[-1] = fxe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
            elif fxr < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_f[-1] = fxr
            else:
                # Contraction
                # Outside contraction if fxr < simplex_f[-1]
                if fxr < simplex_f[-1]:
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                else:
                    xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                fxc = func(xc)
                remaining_budget -= 1
                if fxc < min(fxr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fxc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        remaining_budget -= 1
                    if remaining_budget <= 0:
                        break
            
            # Check if simplex has collapsed
            if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
