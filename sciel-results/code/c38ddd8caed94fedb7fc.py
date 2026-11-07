"""BBOB f14 (Different powers) optimizer: Nelder-Mead with multi-start and polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead simplex method with multi-start and adaptive polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 325.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(42)
        
        best_x_global = None
        best_f_global = np.inf
        call_count = 0

        def nm_search(x_start, sigma_init, max_calls, deadline):
            """
            Perform Nelder-Mead optimization starting from x_start.
            Returns (best_x, best_f, calls_made).
            """
            nonlocal call_count
            nonlocal best_x_global
            nonlocal best_f_global

            # Initialize simplex
            # Vertices: x_start + sigma * basis vectors
            simplex = [x_start.copy()]
            for i in range(dim):
                v = x_start.copy()
                v[i] += sigma_init
                simplex.append(np.clip(v, -5, 5))
            
            # Evaluate initial simplex
            f_values = []
            for v in simplex:
                if call_count >= budget or time.time() >= deadline:
                    break
                f_v = func(v)
                call_count += 1
                if not np.isfinite(f_v):
                    f_v = np.inf
                f_values.append(f_v)
            
            # If we ran out of budget during initialization, return best found
            if len(f_values) < dim + 1:
                # Find best in what we have
                if f_values:
                    idx = np.argmin(f_values)
                    return simplex[idx], f_values[idx], call_count
                return x_start, np.inf, call_count

            # Sort simplex by f value
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Track best
            best_x = simplex[0].copy()
            best_f = f_values[0]
            if best_f < best_f_global:
                best_f_global = best_f
                best_x_global = best_x.copy()

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            while call_count < budget and time.time() < deadline:
                # 1. Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                call_count += 1
                if not np.isfinite(fr): fr = np.inf

                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    call_count += 1
                    if not np.isfinite(fe): fe = np.inf
                    
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
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        call_count += 1
                        if not np.isfinite(fc): fc = np.inf
                        
                        if fc <= fr:
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_values[i] = func(simplex[i])
                                call_count += 1
                                if not np.isfinite(f_values[i]): f_values[i] = np.inf
                            break # Restart logic usually needed here, but we just break to re-sort and continue
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - xr)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        call_count += 1
                        if not np.isfinite(fc): fc = np.inf
                        
                        if fc <= f_values[-1]:
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_values[i] = func(simplex[i])
                                call_count += 1
                                if not np.isfinite(f_values[i]): f_values[i] = np.inf
                            break

                # Sort simplex
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Update best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                    if best_f < best_f_global:
                        best_f_global = best_f
                        best_x_global = best_x.copy()

                # Check for convergence (simplex size)
                if np.linalg.norm(simplex[0] - simplex[-1]) < 1e-6:
                    break

            return best_x, best_f, call_count

        # Strategy:
        # 1. Do a few random evaluations to find a decent starting point.
        # 2. Run NM from the best random point with a moderate sigma.
        # 3. Run NM from the best point found in step 2 with a small sigma (polish).
        
        # Phase 1: Random sampling (10 points)
        random_starts = []
        for i in range(10):
            if call_count >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            call_count += 1
            if np.isfinite(f):
                random_starts.append((x, f))
        
        if not random_starts:
            # Fallback
            x = np.zeros(dim)
            f = func(x)
            call_count += 1
            if np.isfinite(f):
                best_f_global = f
                best_x_global = x
            return best_x_global, best_f_global

        # Sort random starts
        random_starts.sort(key=lambda t: t[1])
        
        # Phase 2: NM from top 3 random starts
        # We allocate roughly 40% of budget to each, but stop if time is up.
        # Let's just run NM until budget is low or time is up.
        # To ensure we try multiple, we limit each NM run to 100 calls initially.
        
        for i in range(min(3, len(random_starts))):
            if call_count >= budget - 50 or time.time() >= deadline:
                break
            x_start, f_start = random_starts[i]
            # Sigma: 1.0 is reasonable for box size 10
            nm_search(x_start, 1.0, budget - 50, deadline)
            
        # Phase 3: Polish
        # If we have budget left, polish the global best found so far.
        if call_count < budget - 10 and time.time() < deadline:
            if best_x_global is not None:
                # Use a very small sigma for polishing
                nm_search(best_x_global, 0.1, budget, deadline)
            
            # If still budget left, try even smaller
            if call_count < budget - 10 and time.time() < deadline:
                if best_x_global is not None:
                    nm_search(best_x_global, 0.01, budget, deadline)

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
