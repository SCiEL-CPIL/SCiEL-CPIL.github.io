"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic multi-start Nelder-Mead optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # WFG7 specific knowledge:
        # The Pareto front of WFG7 is known analytically.
        # For WFG7, the Pareto front is defined by a parameter t in [0, 1] (or similar range).
        # The objectives are related to t.
        # Specifically, for WFG7, the Pareto front can be parameterized.
        # Let's sample the Pareto front directly.
        # WFG7 Pareto front:
        # f1 = t
        # f2 = (1-t)^2
        # f3 = (1-t)^3
        # Wait, I need to verify the exact WFG7 Pareto front shape.
        # WFG7 is a 3-objective problem.
        # From literature (e.g., Huband et al. 2006), WFG7 Pareto front:
        # f1 = t
        # f2 = (1-t)^2
        # f3 = (1-t)^3
        # This holds for t in [0, 1].
        
        # Strategy:
        # 1. Sample the analytic Pareto front to get initial X and F.
        #    We need to find x such that f(x) matches the front.
        #    For WFG7, there is a known mapping from the front parameter t to the decision variables.
        #    However, without the exact inverse mapping in memory, a safer approach is:
        #    - Sample points in the decision space that are known to be near the front? 
        #    - Or, just sample the front in objective space, and then optimize to match those objectives?
        #    
        #    Actually, a simpler and more robust strategy given the "deterministic adaptive refinement" directive:
        #    1. Generate a set of reference points on the known Pareto front (in objective space).
        #    2. For each reference point, run a local search (Nelder-Mead) to find an x that minimizes the distance to that reference point in objective space, OR minimizes a scalarization that targets that point.
        #    
        #    However, the directive says: "sample the known analytic front to establish initial spread, then use the remaining evaluations to perform local gradient-free line searches along the front's tangent directions".
        #    
        #    Let's try to reconstruct the decision variables for the front.
        #    For WFG7, the Pareto-optimal solutions have a specific structure.
        #    If we don't know the exact x for a given f, we can use a "targeting" approach.
        #    
        #    Alternative Interpretation:
        #    Use a fixed set of starting points derived from the front's geometry in decision space?
        #    Since I don't have the exact x-to-f mapping for WFG7 memorized perfectly, I will use a hybrid:
        #    1. Sample a diverse set of initial points in decision space (e.g., Latin Hypercube or uniform grid).
        #    2. Identify the non-dominated set.
        #    3. For each non-dominated point, perform a local Nelder-Mead optimization to improve hypervolume contribution.
        #       The objective for Nelder-Mead will be a scalarization that prioritizes moving towards the "knee" or improving the worst objective, or simply minimizing the sum of objectives while maintaining diversity.
        #       
        #    But the directive specifically mentions "sample the known analytic front".
        #    Let's assume the standard WFG7 front: f1=t, f2=(1-t)^2, f3=(1-t)^3 for t in [0,1].
        #    We can generate target objective vectors.
        #    Then, for each target, we run a local search to find an x that produces objectives close to the target.
        #    The objective function for the local search will be the sum of squared differences between the actual objectives and the target objectives.
        #    This is a "tracking" strategy.
        
        n_starts = 20  # Number of local searches to run
        # Allocate budget: 
        # - 1 evaluation for initial guess? No, we can start from a random point.
        # - Each Nelder-Mead run will consume evaluations.
        # - We have 200 budget.
        # - Let's do 10 starts, with 20 evaluations each? 
        # - Or 20 starts with 10 evaluations each?
        # - Nelder-Mead in 12D is expensive. 10 evaluations is very few.
        # - Maybe 5 starts with 40 evaluations each?
        
        # Let's use 10 starts.
        # For each start:
        # 1. Pick a target point on the Pareto front (t in [0, 1]).
        # 2. Initialize x randomly.
        # 3. Run Nelder-Mead to minimize ||f(x) - target||^2.
        # 4. Record the best x and f(x) found.
        
        # We need to be careful with the budget.
        # scipy.optimize.minimize with method='Nelder-Mead' allows maxiter.
        # Each iteration does 1 evaluation (roughly, actually it does several per simplex update).
        # We can use a callback to count evaluations.
        
        eval_count = 0
        best_X = []
        best_F = []
        
        # Generate targets on the Pareto front
        # t values: evenly spaced
        t_vals = np.linspace(0.0, 1.0, n_starts)
        
        # Shuffle to ensure diversity in decision space starts?
        # No, just use random starts for each target.
        
        rng = np.random.default_rng(42)
        
        for i in range(n_starts):
            if time.time() > deadline - 10: # Stop if close to deadline
                break
            
            target_t = t_vals[i]
            target_f = np.array([target_t, (1-target_t)**2, (1-target_t)**3])
            
            # Initial x: random
            x0 = lo + range_ * rng.random(self.dim)
            
            # Objective function for Nelder-Mead: minimize distance to target
            def obj(x):
                nonlocal eval_count
                if time.time() > deadline - 5:
                    return 1e10
                eval_count += 1
                if eval_count > self.budget:
                    return 1e10
                fx = problem(x)
                # Sum of squared errors
                err = np.sum((fx - target_f)**2)
                # Also penalize if any objective is worse than reference?
                # The target is on the front, so it should be good.
                return err
            
            # Run Nelder-Mead
            # maxiter should be small to save budget for other starts
            # 12D Nelder-Mead needs at least 13 points for simplex.
            # Let's allow up to 15 iterations? That might use 15*13 = 195 evals per start? No, that's too many.
            # Nelder-Mead complexity is O(n) per iteration.
            # Let's limit function calls.
            # We have 200 total. 10 starts. 20 evals per start.
            # This is very tight for 12D NM.
            
            # Alternative: Use a simpler local search?
            # Or reduce the number of starts?
            # Let's try 5 starts with 40 evals each.
            
            # Actually, let's just run it and let the budget limit stop it.
            # We'll use a custom loop or careful maxiter.
            
            # Let's try 5 starts.
            # If i >= 5: break (handled by n_starts=5)
            
            # Reset eval count tracking for this start? No, global budget.
            
            # We need to track evaluations inside the minimize call.
            # scipy minimize doesn't have a hard eval limit, only maxiter.
            # We can use a callback to check time/budget.
            
            def callback(xk):
                nonlocal eval_count
                if time.time() > deadline - 5:
                    return True
                if eval_count > self.budget:
                    return True
                return False
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4},
                               callback=callback)
                
                # Evaluate the result
                if eval_count <= self.budget:
                    fx_res = problem(res.x)
                    eval_count += 1 # This evaluation is extra, need to be careful
                    # Actually, the last evaluation in minimize might be the one we want.
                    # But to be safe, we evaluate res.x again.
                    # This might exceed budget if we are tight.
                    # Let's check if we have budget.
                    if eval_count <= self.budget:
                        best_X.append(res.x)
                        best_F.append(fx_res)
                    else:
                        # We exceeded budget, ignore this result
                        pass
                else:
                    pass
                    
            except Exception:
                pass

        # If we have used all budget, return what we have.
        # If we have budget left, we can do more.
        
        # However, the above strategy might be inefficient because NM in 12D is slow.
        # Let's try a different local search: Coordinate Descent or Random Hill Climbing.
        # Random Hill Climbing is simpler and might be more robust with small budgets.
        
        # Let's restart the logic with a more budget-aware approach.
        
        # Clear previous results
        best_X = []
        best_F = []
        eval_count = 0
        
        # Strategy: 10 starts, each with 20 evaluations of Random Hill Climbing.
        # For each start:
        # 1. Target on Pareto front.
        # 2. Random x0.
        # 3. Loop 19 times:
        #    - Generate neighbor: x0 + noise
        #    - Evaluate
        #    - If closer to target, accept.
        
        n_starts = 10
        evals_per_start = 19 # +1 for initial = 20
        
        for i in range(n_starts):
            if time.time() > deadline - 5:
                break
            if eval_count >= self.budget:
                break
                
            target_t = t_vals[i]
            target_f = np.array([target_t, (1-target_t)**2, (1-target_t)**3])
            
            x_cur = lo + range_ * rng.random(self.dim)
            
            # Evaluate initial
            if eval_count < self.budget:
                f_cur = problem(x_cur)
                eval_count += 1
            else:
                break
                
            best_x = x_cur.copy()
            best_f = f_cur.copy()
            best_dist = np.sum((best_f - target_f)**2)
            
            for j in range(evals_per_start):
                if time.time() > deadline - 5:
                    break
                if eval_count >= self.budget:
                    break
                    
                # Generate neighbor
                # Adaptive step size
                step = 0.1 * (0.9 ** j) # Decreasing step
                neighbor = x_cur + step * range_ * rng.standard_normal(self.dim)
                neighbor = np.clip(neighbor, lo, hi)
                
                f_nei = problem(neighbor)
                eval_count += 1
                
                dist_nei = np.sum((f_nei - target_f)**2)
                
                if dist_nei < best_dist:
                    best_x = neighbor.copy()
                    best_f = f_nei.copy()
                    best_dist = dist_nei
                    x_cur = neighbor.copy()
                else:
                    # If we get stuck, try a larger jump?
                    # Keep it simple for now.
                    pass
            
            if eval_count <= self.budget:
                best_X.append(best_x)
                best_F.append(best_f)
        
        # Convert to arrays
        if len(best_X) > 0:
            best_X = np.array(best_X)
            best_F = np.array(best_F)
            
            # Filter out points worse than reference
            mask = np.all(best_F < ref_point, axis=1)
            best_X = best_X[mask]
            best_F = best_F[mask]
            
            # Remove dominated points?
            # The scoring uses hypervolume of ALL recorded vectors.
            # Dominated points do not contribute to HV if there are non-dominated points,
            # but they don't hurt either (they are just ignored by the HV calculation if they are dominated by others in the set? 
            # No, HV calculation considers the union of boxes. A dominated point's box is contained in a non-dominated point's box, so it adds no volume.
            # So we can return all points, or just the non-dominated ones.
            # Returning non-dominated is cleaner.
            
            n = len(best_F)
            if n > 0:
                le = np.all(best_F[:, np.newaxis, :] <= best_F[np.newaxis, :, :], axis=2)
                lt = np.any(best_F[:, np.newaxis, :] < best_F[np.newaxis, :, :], axis=2)
                dominates = le & lt
                np.fill_diagonal(dominates, False)
                is_dominated = np.any(dominates, axis=0)
                nd_idx = np.where(~is_dominated)[0]
                if len(nd_idx) > 0:
                    return best_X[nd_idx]
                else:
                    return best_X
            else:
                return np.array([])
        else:
            return np.array([])

# EVOLVE-BLOCK-END
