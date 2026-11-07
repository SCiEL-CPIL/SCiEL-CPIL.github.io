"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead and Random Search for f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, deadline, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.1):
        n = self.dim
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += 0.5  # Initial step size
        
        # Evaluate initial points
        f_values = np.zeros(n + 1)
        calls = 0
        
        for i in range(n + 1):
            if calls >= budget or time.time() > deadline:
                break
            f_values[i] = func(simplex[i])
            calls += 1
            
        if calls >= budget or time.time() > deadline:
            idx = np.argmin(f_values)
            return simplex[idx].copy(), f_values[idx], calls

        best_x = simplex[np.argmin(f_values)].copy()
        best_f = np.min(f_values)

        while calls < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
                
            # Check convergence
            if np.max(f_values) - np.min(f_values) < 1e-8:
                break
                
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            if calls >= budget or time.time() > deadline:
                break
            fr = func(xr)
            calls += 1
            
            if fr < f_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                if calls >= budget or time.time() > deadline:
                    break
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            elif fr < f_values[-2]:
                # Reflection accepted
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    # Outside contraction
                    xc = centroid + sigma * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - sigma * (xr - centroid)
                    
                if calls >= budget or time.time() > deadline:
                    break
                fc = func(xc)
                calls += 1
                
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        if calls >= budget or time.time() > deadline:
                            break
                        f_values[i] = func(simplex[i])
                        calls += 1
                    if calls >= budget or time.time() > deadline:
                        break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        budget = self.budget
        n = self.dim

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Global Random Search to find promising basins
        # Use 20% of budget for random search
        rand_budget = max(10, budget // 5)
        for _ in range(rand_budget):
            if total_calls >= budget or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, n)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            total_calls += 1

        # Phase 2: Multi-start Nelder-Mead from best points and random points
        # We have remaining budget for local refinement
        remaining_budget = budget - total_calls
        
        # Generate a few diverse starting points for Nelder-Mead
        # 1. The best point found in random search
        # 2. Origin
        # 3. A few random points near the best point
        starts = [best_x.copy(), np.zeros(n)]
        
        # Add points perturbed from the best
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            perturbation = np.random.randn(n) * 0.5
            x_new = np.clip(best_x + perturbation, -5, 5)
            starts.append(x_new)

        # Allocate remaining budget to these starts
        # Prioritize the best point
        nm_budgets = []
        # Give 50% to the best point, 25% to origin, 25% to others
        nm_budgets.append(int(remaining_budget * 0.5))
        nm_budgets.append(int(remaining_budget * 0.25))
        other_budget = remaining_budget - nm_budgets[0] - nm_budgets[1]
        per_other = other_budget // max(1, len(starts) - 2)
        for _ in range(len(starts) - 2):
            nm_budgets.append(per_other)

        for i, start in enumerate(starts):
            if total_calls >= budget or time.time() > deadline - 2:
                break
            b = nm_budgets[i]
            if b < 5: # Not enough budget for NM to be useful
                continue
            try:
                bx, bf, used = self._nelder_mead(func, start, b, deadline)
                total_calls += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx
            except Exception:
                pass

        # Phase 3: Final local refinement with (1+1)-ES or small NM if budget left
        if total_calls < budget and time.time() < deadline - 1 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while total_calls < budget and time.time() < deadline - 0.5:
                y = np.clip(x + sigma * np.random.randn(n), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
