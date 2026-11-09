"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Track best solution found
        best_x = None
        best_f = np.inf
        calls_made = 0

        # We will use a multi-start strategy with Nelder-Mead
        # Each local run gets a sub-budget
        # We need to be careful about the total number of function calls
        
        # Strategy:
        # 1. Generate several starting points.
        # 2. Run Nelder-Mead from each, limiting the number of iterations/calls.
        # 3. Keep track of the best point found across all runs.
        # 4. If budget remains, perform a final refinement around the best point.

        # Let's allocate budget dynamically.
        # First, do a quick random search to find a decent basin?
        # Or just start with a few Nelder-Mead runs.
        
        # f24 (Lunacek) has a global minimum and a local minimum.
        # The global minimum is at x* = x0 + s * (x0 - x0') roughly, but with rotation.
        # The local minimum is at x0 (shifted).
        # The function is multimodal.
        
        # Nelder-Mead is good for local optimization but can get stuck.
        # CMA-ES is better for multimodal but heavier.
        # Let's try a hybrid: Random restarts with Nelder-Mead.
        
        # To respect the budget strictly:
        # We wrap func to count calls.
        
        self._calls = 0
        def counted_func(x):
            nonlocal self
            self._calls += 1
            if self._calls > budget:
                # Return a large value to stop optimization, or raise?
                # The contract says "More than 500 calls ... makes the whole result invalid"
                # So we must stop before exceeding budget.
                # We'll just return the last known best or a large penalty.
                # Better to check before calling.
                pass
            return func(x)

        # Actually, simpler: just check self._calls before calling func.
        
        def safe_func(x):
            if self._calls >= budget:
                return 1e10
            self._calls += 1
            return func(x)

        # Initial random points
        # Let's try 5-10 different starting points.
        # Each Nelder-Mead run uses a simplex of size dim+1.
        # Iterations limit per run: say 50-100.
        
        num_starts = 8
        # Allocate budget:
        # 10% for initial random scan?
        # 90% for local optimizations.
        
        # Let's do a small random scan first to identify promising areas.
        # But random scan is inefficient for 5D.
        # Maybe just pick starts spread out.
        
        # Start points:
        # One at the center (0,0,0,0,0)
        # Others random.
        
        starts = [np.zeros(dim)]
        for _ in range(num_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))
            
        # Shuffle starts
        idx = rng.permutation(num_starts)
        starts = [starts[i] for i in idx]
        
        # Run Nelder-Mead from each start
        # Limit iterations to ensure we don't blow the budget
        # Nelder-Mead in scipy uses 'nfev' to track function evaluations.
        # We can pass a callback or just check self._calls.
        
        for start in starts:
            if self._calls >= budget - 10:
                break
                
            # Define options for Nelder-Mead
            # maxiter controls iterations, not function calls directly, but related.
            # We'll use a maxiter that is conservative.
            # In 5D, one iteration might take ~2-10 function calls.
            # Let's allow up to 200 function calls per local run?
            # But we have limited budget.
            
            # Let's estimate: if we have 500 calls, and 8 starts, that's ~60 calls per start.
            # That might be too low for NM to converge.
            # Maybe fewer starts, more budget per start?
            # 4 starts, 100 calls each?
            
            # Let's try 5 starts.
            # If we use 5 starts, and reserve 50 calls for final polish.
            # Remaining 450 calls / 5 = 90 calls per start.
            
            # We'll modify the loop to break early if budget is low.
            
            try:
                # Use scipy.optimize.minimize
                # Method: Nelder-Mead
                # We need to stop when self._calls reaches budget.
                # scipy doesn't have a direct "max function calls" stop for NM,
                # but we can use a callback to check.
                
                def callback(xk):
                    # This is called after each iteration
                    # We can't easily stop from here in older scipy, 
                    # but we can return a boolean in newer versions?
                    # Actually, scipy.optimize.minimize doesn't support stopping via callback return value for all methods.
                    # For Nelder-Mead, it stops on maxiter or tol.
                    # We'll just rely on maxiter and check self._calls before starting the next start.
                    # To be safe, we can set maxiter such that even if it goes over, we've already checked.
                    # But if it goes over, we violate the contract.
                    # So we must ensure the total calls don't exceed budget.
                    
                    # A safer approach: implement our own Nelder-Mead?
                    # Or use a wrapper that raises an exception when budget is exceeded,
                    # and catch it.
                    
                    if self._calls > budget:
                        raise Exception("Budget exceeded")
                    return False

                # The callback approach is risky if it doesn't stop immediately.
                # Let's use a try-except with a custom exception.
                
                # Actually, let's just limit maxiter conservatively.
                # If maxiter is too high, it might exceed budget.
                # We can check self._calls inside the loop of a custom NM implementation.
                
                # Given the constraints, a custom simple NM or using scipy with careful maxiter is needed.
                # Let's try scipy with maxiter=50.
                # In 5D, 50 iterations might use ~100-200 function calls.
                # That's too many for 5 starts.
                
                # Let's reduce starts to 3 and maxiter to 30.
                # 3 * 30 * ~3 calls/iter = 270 calls. Safe.
                
            except Exception:
                pass

        # This approach is getting complicated with scipy's black-box nature.
        # Let's implement a simple Nelder-Mead ourselves to have full control over function calls.
        
        # Reset calls for the custom implementation? No, self._calls is already tracking.
        # But we haven't started the custom NM yet. The code above didn't run anything because I commented out the actual call.
        # Let's restart the logic cleanly.

        # Clear the previous logic and implement custom NM.
        
        # Re-initialize state if we want to be clean, but self._calls is already 0.
        
        def nelder_mead(func, x0, max_calls=100):
            """
            Custom Nelder-Mead implementation.
            func: function to minimize
            x0: initial point
            max_calls: max number of function evaluations allowed in this call
            """
            nonlocal self, best_x, best_f
            
            dim = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            f_simplex = [func(x0)]
            self._calls += 1
            best_f = min(best_f, f_simplex[0])
            best_x = x0.copy()
            
            for i in range(1, dim + 1):
                xi = x0.copy()
                xi[i] += 0.5  # Small perturbation
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                self._calls += 1
                simplex.append(xi)
                f_simplex.append(fi)
                best_f = min(best_f, fi)
                if fi < best_f:
                    best_x = xi.copy()
            
            # Sort simplex by function value
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            while self._calls < max_calls:
                # Check convergence
                if max(f_simplex) - min(f_simplex) < 1e-8:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                self._calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if f_simplex[0] <= fr < f_simplex[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                elif fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    self._calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                else:
                    # Contraction
                    if fr < f_simplex[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    self._calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    
                    if fc < min(fr, f_simplex[-1]):
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            self._calls += 1
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            f_simplex[i] = fi
                        
                        # Re-sort
                        order = np.argsort(f_simplex)
                        simplex = [simplex[i] for i in order]
                        f_simplex = [f_simplex[i] for i in order]
                        
                # Re-sort after each step
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                if self._calls >= max_calls:
                    break
            
            return best_x, best_f

        # Multi-start strategy
        # We have 500 calls.
        # Let's do 5 starts.
        # Allocate 80 calls to each start (400 total).
        # Keep 100 calls for a final global refinement or more starts.
        
        num_starts = 5
        calls_per_start = 80
        
        starts = []
        # One start at 0
        starts.append(np.zeros(dim))
        # Other starts random
        for _ in range(num_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))
            
        # Shuffle
        idx = rng.permutation(num_starts)
        starts = [starts[i] for i in idx]
        
        for start in starts:
            if self._calls >= budget - 10:
                break
            # Determine how many calls we can afford for this run
            # We want to leave some budget for remaining starts.
            remaining_starts = num_starts - len(starts)
            # This is getting complex. Let's just use a fixed max_calls per run
            # and break if we're close to budget.
            
            # Adjust max_calls to not exceed budget
            available = budget - self._calls
            if available <= 0:
                break
            
            # Use a portion of the available budget
            # If this is the last start, use more.
            # Simple heuristic: use min(available, calls_per_start)
            # But if available is small, use all of it.
            current_max_calls = min(available, calls_per_start)
            
            if current_max_calls < 10:
                break
                
            nelder_mead(func, start, current_max_calls)
            
        # Final refinement: if we have budget left, run NM from the best point found
        if self._calls < budget - 10:
            if best_x is not None:
                # Perturb best_x slightly? Or just run NM again from best_x
                # NM from best_x might just converge immediately.
                # Let's try a few random perturbations of best_x
                for _ in range(3):
                    if self._calls >= budget - 5:
                        break
                    x_perturbed = best_x + rng.normal(0, 0.1, dim)
                    x_perturbed = np.clip(x_perturbed, -5, 5)
                    nelder_mead(func, x_perturbed, budget - self._calls - 5)
            
        return best_x, best_f

# EVOLVE-BLOCK-END
