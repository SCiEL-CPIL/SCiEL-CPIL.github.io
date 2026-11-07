"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Differential Evolution with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            try:
                return func(x)
            except Exception:
                return None

        # 1. Initial random sampling to find a good starting point
        # Sample 10 points
        rng = np.random.default_rng(42)
        init_samples = rng.uniform(-5, 5, (10, dim))
        for x in init_samples:
            if time.time() > deadline:
                break
            fx = safe_call(x)
            if fx is not None and fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            return None

        # 2. Differential Evolution Phase (Global Exploration)
        # Small population for fast iterations
        pop_size = 12
        # Initialize population around best_x and randomly
        pop = [best_x.copy()]
        for _ in range(pop_size - 1):
            # Mix of global random and local perturbations
            if rng.random() < 0.5:
                px = rng.uniform(-5, 5, dim)
            else:
                px = np.clip(best_x + rng.normal(0, 2.0, dim), -5, 5)
            pop.append(px)
        
        pop_f = [best_f]
        for i in range(1, pop_size):
            fx = safe_call(pop[i])
            if fx is None:
                fx = np.inf
            pop_f.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = pop[i].copy()

        F = 0.8
        CR = 0.9
        stagnation_counter = 0
        max_stagnation = 15

        while calls < budget and time.time() < deadline:
            # Check for stagnation to trigger restart or transition to local
            if stagnation_counter > max_stagnation:
                break
            
            improved = False
            new_pop = []
            new_pop_f = []
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover
                u = np.where(rng.random(dim) < CR, v, pop[i])
                # Ensure at least one gene from v
                if np.all(u == pop[i]):
                    j = rng.integers(dim)
                    u[j] = v[j]
                
                fu = safe_call(u)
                if fu is None:
                    fu = np.inf
                
                if fu < pop_f[i]:
                    new_pop.append(u)
                    new_pop_f.append(fu)
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        improved = True
                    # Update population in-place for next iterations? 
                    # Standard DE updates at end of generation, but for speed we can just keep track.
                    # Actually, let's just store and replace at end of loop for simplicity and correctness.
                    # But we need to update pop_f for subsequent comparisons in this generation? 
                    # Standard DE: compare against current pop[i].
                    # Let's just replace in the list we're building.
                else:
                    new_pop.append(pop[i])
                    new_pop_f.append(pop_f[i])
            
            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Update population
            pop = new_pop
            pop_f = new_pop_f

        # 3. Nelder-Mead Polish Phase (Local Exploitation)
        # Use remaining budget to polish best_x
        if best_x is not None and calls < budget and time.time() < deadline:
            # Initialize simplex
            simplex = [best_x.copy()]
            simplex_f = [best_f]
            delta = 0.5
            for i in range(dim):
                xi = best_x.copy()
                xi[i] += delta
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fx = safe_call(xi)
                if fx is None:
                    fx = np.inf
                simplex_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = xi.copy()

            # Nelder-Mead Steps
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            nm_calls = 0
            max_nm_calls = budget - calls
            min_nm_calls = 10
            if max_nm_calls < min_nm_calls:
                return best_x, best_f

            while nm_calls < max_nm_calls and time.time() < deadline:
                # Sort simplex by function value
                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]
                
                # Check convergence
                if len(set([f for f in simplex_f])) == 1:
                    break
                if simplex_f[0] - simplex_f[-1] < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_call(xr)
                nm_calls += 1
                if fxr is None:
                    fxr = np.inf

                if fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_call(xe)
                    nm_calls += 1
                    if fxe is None:
                        fxe = np.inf
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                    if fxe < best_f:
                        best_f = fxe
                        best_x = xe.copy()
                    elif fxr < best_f:
                        best_f = fxr
                        best_x = xr.copy()
                elif fxr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                    if fxr < best_f:
                        best_f = fxr
                        best_x = xr.copy()
                else:
                    # Contraction
                    if fxr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_call(xc)
                    nm_calls += 1
                    if fxc is None:
                        fxc = np.inf
                    
                    if fxc < min(fxr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                        if fxc < best_f:
                            best_f = fxc
                            best_x = xc.copy()
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fx = safe_call(simplex[i])
                            nm_calls += 1
                            if fx is None:
                                fx = np.inf
                            simplex_f[i] = fx
                            if fx < best_f:
                                best_f = fx
                                best_x = simplex[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
