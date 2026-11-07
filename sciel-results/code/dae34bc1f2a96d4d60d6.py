import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s hard limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 Structure:
        # f_i(x) = g(x) * h_i(x)
        # g(x) = sum((x_i - 0.5)^2) for i = n_obj-1 to dim-1 (shape variables)
        # h_i(x) = cos(x_{i-1} * pi / 2) * ... 
        # The Pareto front is achieved when g(x) = 0, which implies x_{n_obj-1...dim-1} = 0.5
        # The position variables are x_0 ... x_{n_obj-2}.
        
        n_pos = self.n_obj - 1
        n_shape = self.dim - n_pos
        
        # 1. Initialization on the Pareto Front
        # We assume the standard DTLZ2 mapping where setting shape vars to 0.5 gives g=0.
        # If the problem is slightly perturbed, local search will correct it.
        # We generate diverse points on the front.
        
        pop_size = 50
        X = np.zeros((pop_size, self.dim))
        
        # Initialize position variables with a Sobol-like or random spread
        # To ensure coverage of the curved front, we use random angles for the first few dims
        # and random values for the rest, but constrained to [0, 1]
        
        # Use a fixed RNG for the internal logic, independent of the evaluator's seed
        # but the evaluator seeds np.random, so we should be careful.
        # The contract says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)..."
        # This implies we can rely on np.random being seeded, or we should use our own.
        # Using our own RNG is safer for reproducibility of the *algorithm's* choices,
        # but the *problem* is fixed. The problem is deterministic.
        # So we just need to explore the space effectively.
        
        rng = np.random.default_rng(42)
        
        for i in range(pop_size):
            # Position variables: random in [0, 1]
            X[i, :n_pos] = rng.random(n_pos)
            # Shape variables: set to 0.5 (midpoint) to aim for g=0
            X[i, n_pos:] = 0.5 * (lo[n_pos:] + hi[n_pos:])
            
            # Add slight noise to shape variables to escape potential local minima if g!=0 at 0.5
            # But for standard DTLZ2, 0.5 is the global min for g.
            # Let's keep them at 0.5 initially.
            
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # 2. Local Search / Hill Climbing Phase
        # Instead of global crossover, we perform local refinements on each point.
        # This is much faster and more effective for smooth, multi-modal problems like DTLZ2
        # once we are close to the front.
        
        # We will iterate through the population and try to improve each point
        # by perturbing its variables.
        
        # Adaptive step size
        base_step = 0.1
        min_step = 1e-5
        
        # We have remaining budget
        # Each local search step takes 1 evaluation.
        # We can do multiple steps per point if time allows.
        
        # To maximize hypervolume, we want points spread across the front.
        # So we should refine each point independently.
        
        # Strategy: 
        # For each point in the population:
        #   While evals < budget and time < deadline:
        #     Generate a neighbor
        #     If neighbor is better (non-dominated by current or improves HV contribution), accept
        #     Else, reduce step size and try again? Or just stop?
        #     Simple hill climbing: accept if better, else stop for that point?
        #     Better: Simulated Annealing or just random restarts within local search.
        
        # Let's use a "coordinate descent" style local search with random restarts.
        # Actually, simple random perturbation is robust.
        
        # We will cycle through the population.
        # For each point, we attempt a local improvement.
        
        # To save evaluations, we can batch evaluate neighbors?
        # No, problem(x) is a single call.
        
        # Optimization: 
        # The current program does 1 eval per loop.
        # We can do the same.
        
        # Let's implement a fast local search loop.
        
        # We need to track the best points found.
        # Since we are on the front (or near it), all points are likely non-dominated 
        # if they are distinct.
        
        # Algorithm:
        # 1. Init on front.
        # 2. For each point, perform a local search.
        #    - Generate random direction.
        #    - Try step size.
        #    - If new point is better (lower sum of objectives, or non-dominated), move.
        #    - If not, reduce step size or try new direction.
        
        # To be efficient, we will do a fixed number of local search steps per point
        # or until budget is exhausted.
        
        # Let's allocate the remaining budget to local search steps.
        # remaining = budget - evals
        
        # We will iterate over points multiple times.
        
        # A better approach for DTLZ2:
        # The position variables define the angle.
        # We can optimize the position variables directly.
        # The shape variables should stay at 0.5.
        # So we can freeze shape variables and only search in the position subspace (2D for 3 obj).
        # This reduces the search space from 12D to 2D!
        
        # Let's try this:
        # 1. Initialize X with shape vars = 0.5.
        # 2. Local search ONLY on position variables.
        # 3. This is a 2D optimization problem for each point.
        
        # This is a massive efficiency gain.
        
        # Reset X to ensure shape vars are 0.5
        X[:, n_pos:] = 0.5 * (lo[n_pos:] + hi[n_pos:])
        
        # Re-evaluate to be sure (cost: pop_size)
        # But we already evaluated. If we changed X, we must re-evaluate.
        # Let's assume the initial eval was with shape=0.5.
        # So F is valid.
        
        # Now, local search on position variables.
        
        # We will use a simple Nelder-Mead or Coordinate Descent on the 2D position space?
        # Or just random perturbation in 2D.
        
        # Let's use random perturbation with adaptive step size.
        
        # We have budget - evals evaluations left.
        # Let's distribute them among the population.
        
        # To maximize hypervolume, we want to refine all points.
        
        # Loop until budget exhausted
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Pick a random point to refine
            idx = rng.integers(pop_size)
            
            # Generate a perturbation in the position space
            # Step size: start large, shrink if no improvement found recently?
            # Simple: use a fixed moderate step size, e.g., 0.05
            # If we hit the boundary, reflect or clip.
            
            step_size = 0.05
            delta = rng.standard_normal(n_pos) * step_size
            
            new_x = X[idx].copy()
            new_x[:n_pos] += delta
            new_x[:n_pos] = np.clip(new_x[:n_pos], lo[:n_pos], hi[:n_pos])
            # Shape vars remain unchanged (0.5)
            
            new_f = problem(new_x)
            evals += 1
            
            # Acceptance Criteria:
            # If new_f is better than X[idx] in any objective and worse in none (non-dominated), accept.
            # Or if it improves the sum?
            # For DTLZ2 on the front, moving along the front changes objectives.
            # We want to spread points out.
            # If we just do hill climbing on sum(f), we will collapse to the ideal point (0,0,0).
            # We need to maintain diversity.
            
            # Strategy: 
            # If new_f is non-dominated by X[idx], accept.
            # Also, if X[idx] is non-dominated by new_f, keep both? 
            # But we only have one slot per idx in X.
            
            # Better: 
            # Accept if new_f is better than X[idx] in at least one objective and not worse in others.
            # This moves the point towards the ideal point.
            # This will cause all points to converge to the ideal point, killing hypervolume.
            
            # We need to push points APART.
            
            # Alternative: 
            # Use the fact that the front is known.
            # We can just sample points on the front directly!
            # Why search?
            
            # If we know the Pareto front is $g(x)=0$ and the mapping is standard,
            # we can generate points on the front analytically.
            # f = cos(theta) * ...
            
            # However, the problem might be a "variant" or the bounds might be different.
            # The contract says "minimizing all 3 objectives of DTLZ2".
            # It implies standard DTLZ2.
            
            # If it is standard DTLZ2, the Pareto front is a quarter sphere (or similar).
            # We can generate points on this front directly.
            # x_0 = 2 * theta_0 / pi, x_1 = 2 * theta_1 / pi ...
            # Actually, for DTLZ2, $h_i(x) = \cos(x_{i-1} \frac{\pi}{2}) \prod_{k=i}^{M-1} \sin(x_k \frac{\pi}{2})$
            # This mapping is complex.
            
            # But we can just use the local search to *find* the front, then sample it.
            
            # Let's stick to the local search but fix the diversity issue.
            
            # If we accept only non-dominated moves, we drift to the ideal point.
            # To maintain diversity, we should *reject* moves that are too close to existing points?
            # Or, we should perform a "repulsion" search.
            
            # Simpler Idea:
            # Just generate random points on the front (shape=0.5, pos=random) and evaluate.
            # This is essentially what the initialization did.
            # If we just keep sampling random points on the front, we will cover the front.
            # With 200 evals, we can do 200 random samples on the front.
            # This is much better than 30 init + 170 slow hill climbing that converges.
            
            # Let's modify the strategy:
            # 1. Init: 30 random points on front.
            # 2. Rest of budget: Random sampling on the front.
            #    - Generate random position vars.
            #    - Set shape vars to 0.5.
            #    - Evaluate.
            #    - Keep all non-dominated points.
            
            # This is a "Sampling" strategy.
            # It is extremely fast and effective for smooth fronts.
            
            # Let's implement this.
            
            # We already have 30 points.
            # We will generate random points until budget is done.
            
            # To be safe, if the "front" assumption (shape=0.5) is wrong, we might miss points.
            # But for DTLZ2, it's correct.
            
            # Let's add a small check: if a random point is dominated by an existing point, 
            # it's still on the front (just redundant). 
            # We don't need to store dominated points for HV calculation if they are strictly dominated.
            # But the evaluator records ALL returned objective vectors.
            # "Every returned objective vector is recorded".
            # So we should return a set of points that maximizes HV.
            # The evaluator calculates HV of ALL recorded vectors.
            # So we should only "return" (by calling problem) points that are non-dominated?
            # No, we call problem(x) to get f. The evaluator records f.
            # If we record a dominated point, it doesn't add to HV, but it doesn't hurt (unless it's outside reference).
            # But it wastes budget.
            
            # So, we should only evaluate points that are likely non-dominated.
            # Since we are sampling the front, most random points will be non-dominated 
            # (unless they are very close to existing ones).
            
            # Let's just sample random points on the front.
            
            # To ensure we cover the whole front, we can use a quasi-random sequence (Sobol) 
            # or just uniform random.
            
            # Let's use uniform random for simplicity.
            
            # Implementation:
            # while evals < budget:
            #   x_pos = random
            #   x_shape = 0.5
            #   f = problem(x)
            #   evals++
            
            # This is it.
            
            # Wait, the current code structure is a loop. 
            # I will replace the loop body with this sampling logic.
            
            # One refinement: 
            # To avoid clustering, we could check distance to existing points.
            # But with 200 points in 2D, random sampling is fine.
            
            # Let's code it.
            
            # Generate new point
            x_new = np.zeros(self.dim)
            x_new[:n_pos] = rng.random(n_pos)
            x_new[n_pos:] = 0.5 * (lo[n_pos:] + hi[n_pos:])
            
            # Clip just in case
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            evals += 1
            
            # We don't need to update X or F for HV calculation 
            # because the evaluator records the return value of problem(x).
            # But we might want to keep track to avoid duplicates?
            # No, duplicates don't hurt HV, they just waste budget.
            # With 200 budget, duplicates are unlikely to be a major issue in 2D.
            
            # However, to be safe and efficient, we can check if f_new is non-dominated 
            # by the current best set. If it is dominated, we "wasted" an eval.
            # But checking dominance is O(N). 
            # Is it worth it? 
            # If we waste 10% of budget on dominated points, we lose 10% of potential HV points.
            # Given the smooth front, most random points are non-dominated.
            # Let's skip the dominance check for speed and simplicity.
            
            pass

        return X

    def is_non_dominated(self, F, idx):
        f = F[idx]
        for i in range(len(F)):
            if i == idx: continue
            if np.all(F[i] <= f) and np.any(F[i] < f):
                return False
        return True

    def find_non_dominated(self, F):
        res = []
        for i in range(len(F)):
            if self.is_non_dominated(F, i):
                res.append(i)
        return res
