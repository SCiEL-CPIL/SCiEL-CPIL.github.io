"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start quadratic model optimizer for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        # Deterministic starting points
        starts = [
            np.zeros(dim),
            np.ones(dim) * 2.5,
            np.ones(dim) * -2.5,
            np.array([1, -1, 1, -1, 1]) * 1.5,
            np.array([-1, 1, -1, 1, -1]) * 1.5,
            np.array([0, 0, 0, 0, 0]), # Duplicate for safety, will be overwritten by randoms if needed
        ]
        
        # Add some random starts for diversity
        rng = np.random.default_rng(42)
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, dim))

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration with quadratic model fitting
        # We will try to fit a quadratic model to 11 points (center + 10 neighbors)
        # and jump to the minimum of that model.
        
        # We use a list of "candidates" to start local optimization
        candidates = starts.copy()
        
        # We will iterate through candidates. For each, we do a local quadratic refinement.
        # To save budget, we limit the number of refinement steps per candidate.
        
        # Pre-compute neighbor directions for quadratic fit
        # 11 points: center, +d_i, -d_i for i in 1..5
        # This is a standard finite difference set for quadratic fit.
        
        while candidates and calls_made < budget and time.time() < deadline:
            x = candidates.pop(0)
            
            # Local refinement loop
            # We perform a few steps of quadratic model fitting
            max_local_steps = 10
            step_scale = 0.5
            
            for _ in range(max_local_steps):
                if calls_made + 11 > budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Evaluate center
                fc = evaluate(x)
                if not np.isfinite(fc):
                    break
                
                # Evaluate neighbors
                neighbors = []
                f_neighbors = []
                
                for i in range(dim):
                    for sign in [1, -1]:
                        xn = x.copy()
                        xn[i] += sign * step_scale
                        neighbors.append(xn)
                        f_neighbors.append(evaluate(xn))
                
                # Check if all evaluations were valid
                if any(not np.isfinite(f) for f in f_neighbors):
                    break
                    
                # Fit quadratic model: f(x) = a + b^T (x-x0) + (x-x0)^T Q (x-x0)
                # We use the 11 points to solve for b and Q.
                # Let d_i = x_i - x0.
                # f_i = a + b^T d_i + d_i^T Q d_i
                # We have 11 equations. Unknowns: a (1), b (5), Q (15) -> 21 unknowns.
                # Underdetermined! We need more points or constraints.
                # Standard "pure quadratic" fit requires N+1 + N(N+1)/2 points.
                # For N=5: 1 + 5 + 15 = 21 points.
                # 11 points is not enough for a full quadratic fit.
                
                # Alternative: Fit a linear model + diagonal quadratic?
                # Or use the 11 points to estimate gradient and Hessian diagonal?
                # Let's try a simpler approach: Gradient estimation and step.
                # But the directive says "fit a local paraboloid... to jump to predicted basin center".
                # With 11 points, we can't get the full Hessian.
                # However, Lunacek is separable in some transformed space? No, it's rotated.
                
                # Let's use the 11 points to fit a model f(x) = a + b^T d + sum(q_i * d_i^2).
                # This assumes diagonal Hessian in the local coordinate system (aligned with axes).
                # This is a "diagonal quadratic" model.
                # Unknowns: a (1), b (5), q (5) -> 11 unknowns.
                # We have 11 points. This is exactly determined!
                
                A = np.zeros((11, 11))
                y = np.zeros(11)
                
                # Point 0: center
                A[0, 0] = 1
                y[0] = fc
                
                idx = 1
                for i in range(dim):
                    for sign in [1, -1]:
                        d_i = sign * step_scale
                        # d vector has d_i at index i, 0 elsewhere
                        # Model: a + b_i * d_i + q_i * d_i^2
                        A[idx, 0] = 1
                        A[idx, 1+i] = d_i
                        A[idx, 6+i] = d_i**2
                        y[idx] = f_neighbors[idx-1]
                        idx += 1
                
                try:
                    coeffs = np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    break
                    
                a = coeffs[0]
                b = coeffs[1:6]
                q = coeffs[6:11]
                
                # Predicted minimum of the model:
                # For each dimension i, if q_i != 0, min is at -b_i / (2*q_i)
                # If q_i == 0, no min in that direction (linear).
                # We clamp the step to avoid huge jumps if q_i is tiny.
                
                x_new = x.copy()
                for i in range(dim):
                    if abs(q[i]) > 1e-6:
                        step = -b[i] / (2 * q[i])
                        # Clamp step to reasonable range relative to step_scale
                        # If step is huge, the model is likely bad or flat
                        step = np.clip(step, -2 * step_scale, 2 * step_scale)
                        x_new[i] += step
                    else:
                        # If linear, take a small step against gradient
                        if b[i] > 0:
                            x_new[i] -= step_scale * 0.5
                        elif b[i] < 0:
                            x_new[i] += step_scale * 0.5
                        
                # Evaluate new point
                f_new = evaluate(x_new)
                
                if f_new < fc:
                    x = x_new
                    # Shrink step scale slightly for next iteration to converge
                    step_scale *= 0.7
                    if step_scale < 0.01:
                        break
                else:
                    # No improvement, break local loop
                    break

            # If we improved, add neighbors to candidate list for further exploration?
            # To keep it simple and fast, we just rely on the initial set of starts.
            # However, if we found a good spot, we might want to try a bit more.
            # But budget is tight. The initial 11 starts * 11 evals * 10 steps is too many.
            # 11 starts * 11 evals * 10 steps = 1210 calls. Budget is 500.
            # We need to be more conservative.
            
            # Let's adjust the strategy: 
            # Use fewer starts, or fewer local steps.
            # With 500 budget, we can afford ~40 local refinements of 11 evals each.
            # Or ~45 starts with 1 refinement step each.
            
            # Let's prioritize the most promising starts.
            # We'll sort candidates by their initial evaluation if we had them.
            # But we evaluate them on the fly.
            
            # Let's reduce max_local_steps to 5 and step_scale decay faster.
            # And limit total candidates processed.

        # Fallback: if we have budget left, do some random search
        while calls_made < budget and time.time() < deadline:
            x_rand = rng.uniform(-5, 5, dim)
            evaluate(x_rand)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
