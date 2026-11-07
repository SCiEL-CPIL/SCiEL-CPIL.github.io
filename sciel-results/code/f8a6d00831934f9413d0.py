import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We target finishing well before.
        # 200 evaluations is small, so overhead is the main risk.
        limit_time = 320

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # WFG9 specific knowledge:
        # The Pareto front of WFG9 (3 objectives) is a triangular surface.
        # The decision variable x1 directly controls the trade-off.
        # Specifically, f1 = x1, and f2, f3 depend on (1-x1) and some position-dependent terms.
        # In standard WFG9, the Pareto front lies on the plane f1+f2+f3 = 1 (approximately, depending on scaling).
        # Actually, for WFG9 with 3 objectives:
        # f1 = x1
        # f2 = (1-x1) * (1 + 0.1 * (f1 + f3)) ... this is recursive.
        # Let's look up WFG9 definition.
        # WFG9 is a 3-objective problem.
        # f1 = x1
        # f2 = (1 - x1) * (1 + 0.1 * (f1 + f3))  -- No, this is not standard.
        # Standard WFG9:
        # f1 = x1
        # f2 = (1 - x1) * (1 + 0.1 * (f1 + f3)) is incorrect.
        # Correct WFG9:
        # f1 = x1
        # f2 = (1 - x1) * (1 + 0.1 * (f1 + f3)) -- wait, the definition involves t1, t2, etc.
        # Let's assume the standard structure where the Pareto front is a triangle in the objective space.
        # The key insight is that the Pareto front is a 2D manifold in the 3D objective space.
        # We can sample points on this manifold and map them back to the decision space.
        # However, the mapping is not trivial.
        
        # Alternative approach:
        # Use a specialized optimizer for WFG9.
        # Since the Pareto front is known to be a triangle, we can:
        # 1. Generate random points on the triangle f1+f2+f3 = 1 (or similar) in the objective space.
        # 2. For each point, solve for x that minimizes the distance to that target objective vector.
        # This is a constrained optimization problem.
        
        # But we have a budget of 200. Solving 200 constrained problems is expensive.
        
        # Better approach:
        # Use a multi-objective evolutionary algorithm (MOEA) with a small population, but with a twist:
        # Initialize the population with points that are likely on or near the Pareto front.
        # For WFG9, the Pareto front is characterized by x1 being the only "position" variable.
        # The other variables x2...x12 are "shape" variables that should be set to specific values to lie on the Pareto front.
        # Specifically, for WFG9, the Pareto front is reached when x2...x12 are set to values that make the shape functions zero or minimal.
        # In many WFG problems, the Pareto front is reached when the "position" variable is varied and the "shape" variables are set to specific constants.
        # For WFG9, the Pareto front is a triangle.
        # Let's try to initialize with points where x2...x12 are set to the midpoint of their bounds, and x1 is varied.
        # Then, use local search to refine.

        # Strategy:
        # 1. Initialize a population of 20 points.
        #    - 10 points with x1 uniform in [lo[0], hi[0]] and x2...x12 at midpoints.
        #    - 10 points fully random.
        # 2. Use a fast non-dominated sort to find the Pareto front.
        # 3. Use local search (CMA-ES or simple gradient-free) on the best points to refine.
        #    Since the budget is small, we can afford a few local searches.

        # Let's implement a simple version:
        # - Initial population: 20 points.
        # - Loop:
        #   - Non-dominated sort.
        #   - Select parents from the first front.
        #   - Generate offspring via crossover and mutation.
        #   - Evaluate offspring.
        #   - Update population (keep best 20).

        # To speed up, we will vectorize where possible.
        # But problem(x) is a black-box, so we must call it sequentially.

        # Let's try a different approach:
        # Since the Pareto front is a triangle, we can parameterize it.
        # Let u, v be parameters in [0,1] with u+v <= 1.
        # f1 = u, f2 = v, f3 = 1 - u - v.
        # We can sample u, v and then use a local search to find x that minimizes ||f(x) - [u, v, 1-u-v]||^2.
        # This is a single-objective optimization problem for each sample.
        # With 200 budget, we can do, say, 20 samples and 10 evaluations each for local search.
        # This might be more effective than a generic EA.

        # Let's implement this "target-based" approach.
        # 1. Sample 20 points on the triangle.
        # 2. For each target, run a local search (e.g., Nelder-Mead) with a budget of 10 evaluations.
        # 3. Collect all evaluated points.
        # 4. Return the non-dominated set.

        # Nelder-Mead is available in scipy.optimize.
        # But we must be careful with the budget.
        # Total budget = 200.
        # If we use 20 targets * 10 evals = 200, we are exactly at the budget.
        # But we need to evaluate the initial points too.
        # So, let's use 10 targets * 10 evals = 100 evals for local search.
        # And 100 evals for initial random sampling.
        # Total 200.

        # Let's refine:
        # - 50 random points to get a rough idea of the front.
        # - 15 targets on the triangle.
        # - For each target, run Nelder-Mead with maxiter=10 (approx 10 evals).
        # - Total: 50 + 15*10 = 200.

        # This is a solid plan.

        # Step 1: Generate initial random points.
        n_init = 50
        X_init = lo + np.random.rand(n_init, dim) * span
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Step 2: Generate targets on the triangle.
        # The Pareto front of WFG9 is approximately f1+f2+f3 = 1.
        # But we don't know the exact scaling. Let's assume the reference point [3,5,7] suggests the objectives are in [0,1] range.
        # So, f1+f2+f3 = 1 is a good guess.
        n_targets = 15
        targets = []
        for _ in range(n_targets):
            u = np.random.rand()
            v = np.random.rand() * (1 - u)
            targets.append(np.array([u, v, 1 - u - v]))

        # Step 3: For each target, run Nelder-Mead.
        # We need to define the objective function for Nelder-Mead.
        # It should minimize the distance between f(x) and the target.
        
        # We will use scipy.optimize.minimize with method='Nelder-Mead'.
        # But we must limit the number of function evaluations.
        # Nelder-Mead does not have a direct maxfev parameter in all versions, but we can use options={'maxiter': 10}.
        
        # Let's implement a simple local search instead of relying on scipy, to have full control over the budget.
        # A simple coordinate descent or random perturbation might be sufficient.
        
        # Let's use a simple random perturbation local search:
        # For each target:
        #   1. Start from the best point in F_init that is closest to the target.
        #   2. For 10 iterations:
        #      - Perturb the point.
        #      - If the new point is closer to the target, accept it.
        #      - Otherwise, try a smaller perturbation.

        # This is simpler and more controllable.

        # First, find the best initial point for each target.
        # We can compute the distance from each initial point to each target and pick the closest.
        
        # Let's do that.
        
        # Compute distances between F_init and targets.
        # F_init shape: (50, 3)
        # targets shape: (15, 3)
        # Distances shape: (50, 15)
        
        # We will use this to initialize the local search.
        
        # But we must be careful with the time limit.
        # Let's add a time check in the loop.

        # Let's implement the local search.
        
        # We will keep a list of all evaluated points to return the non-dominated set.
        all_X = list(X_init)
        all_F = list(F_init)

        for target in targets:
            if time.time() > limit_time:
                break
            if used >= budget:
                break

            # Find the best initial point for this target.
            # We can use the last evaluated point or the closest initial point.
            # Let's use the closest initial point.
            # But we can also use the result of the previous local search to warm-start.
            # Let's use the closest initial point for simplicity.
            
            # Compute distances from F_init to target.
            # We can do this efficiently.
            dists = np.linalg.norm(F_init - target, axis=1)
            best_init_idx = np.argmin(dists)
            x_current = X_init[best_init_idx].copy()
            f_current = F_init[best_init_idx].copy()
            
            # Local search
            step_size = 0.1 * span
            for _ in range(10):
                if time.time() > limit_time:
                    break
                if used >= budget:
                    break

                x_trial = x_current + np.random.normal(0, step_size)
                x_trial = np.clip(x_trial, lo, hi)
                f_trial = problem(x_trial)
                used += 1

                # Check if the trial point is closer to the target.
                dist_current = np.linalg.norm(f_current - target)
                dist_trial = np.linalg.norm(f_trial - target)

                if dist_trial < dist_current:
                    x_current = x_trial
                    f_current = f_trial
                    step_size *= 1.1  # Increase step size if we are improving.
                else:
                    step_size *= 0.8  # Decrease step size if we are not improving.

                all_X.append(x_current)
                all_F.append(f_current)

        # Step 4: Compute the non-dominated set from all_X and all_F.
        # We will use a fast non-dominated sort.
        
        all_X = np.array(all_X)
        all_F = np.array(all_F)
        
        # Non-dominated sort
        n = len(all_F)
        S = [[] for _ in range(n)]
        n_dom = [0] * n
        rank = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(all_F[i] <= all_F[j]) and np.any(all_F[i] < all_F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        if current_front:
            fronts.append(current_front)
        
        k = 0
        while k < len(fronts) and fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            if next_front:
                k += 1
                fronts.append(next_front)

        # Return the first front (Pareto front).
        # But we must return a list of points, not indices.
        # The task says "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # So we can return anything, but let's return the Pareto front for clarity.
        
        if fronts:
            pareto_front = all_X[fronts[0]]
            return pareto_front
        else:
            return all_X

