"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid global-local optimizer for BBOB f24 (Lunacek bi-Rastrigin).
    
    Strategy:
    1. Deterministic Latin Hypercube sampling to rapidly identify basins.
    2. Fixed-step Nelder-Mead simplex method to exploit local smoothness.
    3. Deadline guard to ensure we return before wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: Latin Hypercube Sampling for global basin identification
        # Use a dense LHS to cover the space [-5, 5]^5
        # We'll allocate ~30% of budget to global search
        n_lhs = min(150, self.budget // 3)
        
        # Generate Latin Hypercube samples
        # For each dimension, divide into n_lhs equal intervals and pick one point per interval
        lhs_samples = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            # Random permutation of intervals
            perm = rng.permutation(n_lhs)
            # Pick a point within each interval [i/n_lhs, (i+1)/n_lhs] mapped to [-5, 5]
            points = (perm + rng.uniform(size=n_lhs)) / n_lhs
            lhs_samples[:, d] = -5 + 10 * points
        
        # Evaluate LHS samples
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_samples[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Nelder-Mead local optimization from best LHS point
        # Allocate remaining budget to local search
        remaining_budget = self.budget - calls_made
        
        # Nelder-Mead parameters
        # Initial simplex size
        alpha = 1.5  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Create initial simplex centered at best_x
        # Use small step size
        step = 0.5
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = best_x
        for i in range(1, self.dim + 1):
            simplex[i] = best_x.copy()
            simplex[i, i - 1] += step
        
        # Evaluate simplex vertices
        fvals = np.zeros(self.dim + 1)
        for i in range(self.dim + 1):
            if time.time() > deadline:
                break
            fvals[i] = func(simplex[i])
            calls_made += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()
        
        # Nelder-Mead iterations
        # Limit iterations to avoid infinite loops
        max_nm_iters = min(remaining_budget - (self.dim + 1), 200)
        
        for _ in range(max_nm_iters):
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fxr = func(xr)
            calls_made += 1
            if time.time() > deadline:
                break
            
            if fxr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fxe = func(xe)
                calls_made += 1
                if time.time() > deadline:
                    break
                if fxe < fxr:
                    simplex[-1] = xe
                    fvals[-1] = fxe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fxr
            elif fxr < fvals[-2]:
                # Simple reflection
                simplex[-1] = xr
                fvals[-1] = fxr
            else:
                # Contraction
                if fxr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    fxc = func(xc)
                    calls_made += 1
                    if time.time() > deadline:
                        break
                    if fxc <= fxr:
                        simplex[-1] = xc
                        fvals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if time.time() > deadline:
                                break
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    fxc = func(xc)
                    calls_made += 1
                    if time.time() > deadline:
                        break
                    if fxc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if time.time() > deadline:
                                break
            
            # Update best
            for i in range(self.dim + 1):
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()
            
            # Check for convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break
        
        # Phase 3: If we have budget left, do a few more random restarts with NM
        if calls_made < self.budget and time.time() < deadline:
            n_restarts = min(5, (self.budget - calls_made) // 50)
            for r in range(n_restarts):
                if time.time() > deadline:
                    break
                
                # Random restart point
                x0 = rng.uniform(-5, 5, self.dim)
                fx0 = func(x0)
                calls_made += 1
                if time.time() > deadline:
                    break
                
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()
                
                # Quick NM from this point
                step = 0.3
                simplex = np.zeros((self.dim + 1, self.dim))
                simplex[0] = x0
                for i in range(1, self.dim + 1):
                    simplex[i] = x0.copy()
                    simplex[i, i - 1] += step
                
                fvals = np.zeros(self.dim + 1)
                for i in range(self.dim + 1):
                    if time.time() > deadline:
                        break
                    fvals[i] = func(simplex[i])
                    calls_made += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = simplex[i].copy()
                
                # Few NM iterations
                for _ in range(30):
                    if time.time() > deadline or calls_made >= self.budget:
                        break
                    
                    idx = np.argsort(fvals)
                    simplex = simplex[idx]
                    fvals = fvals[idx]
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fxr = func(xr)
                    calls_made += 1
                    if time.time() > deadline:
                        break
                    
                    if fxr < fvals[0]:
                        simplex[-1] = xr
                        fvals[-1] = fxr
                    elif fxr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fxr
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                        fxc = func(xc)
                        calls_made += 1
                        if time.time() > deadline:
                            break
                        if fxc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fxc
                        else:
                            for i in range(1, self.dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                                calls_made += 1
                                if time.time() > deadline:
                                    break
                    
                    for i in range(self.dim + 1):
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
