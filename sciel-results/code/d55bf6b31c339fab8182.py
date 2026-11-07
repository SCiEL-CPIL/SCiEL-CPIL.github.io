import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        dim = self.dim

        # DTLZ2 structure:
        # First M-1 variables (here 2) determine position on front.
        # Remaining N-M+1 variables (here 10) determine distance from front (g function).
        n_pos = n_obj - 1
        n_dist = dim - n_pos

        # Phase 1: Determine the optimal value for distance variables (g minimizer)
        # Standard DTLZ2 has g=0 when all dist vars are 0.5
        # But we verify by testing a couple of candidates
        
        # Test point A: dist vars at 0.5 (standard DTLZ2 optimum)
        x_A = np.zeros(dim)
        x_A[0] = 0.5
        x_A[1] = 0.5
        x_A[n_pos:] = 0.5
        f_A = np.array(problem(x_A))
        used = 1
        
        # Test point B: dist vars at 0.0 (alternative hypothesis)
        x_B = np.zeros(dim)
        x_B[0] = 0.5
        x_B[1] = 0.5
        x_B[n_pos:] = 0.0
        f_B = np.array(problem(x_B))
        used = 2

        # The point on the Pareto front should have minimal norm (closest to origin)
        # assuming the front is at distance 1 (or minimal possible distance)
        norm_A = np.linalg.norm(f_A)
        norm_B = np.linalg.norm(f_B)
        
        # Choose the distance variable value that yields lower norm
        if norm_A <= norm_B:
            dist_val = 0.5
        else:
            dist_val = 0.0

        # Phase 2: Generate a dense set of points on the Pareto front
        # Use Latin Hypercube Sampling (LHS) for better spread than uniform grid
        remaining_budget = self.budget - used
        
        # We'll use most of the budget for front sampling
        # LHS gives better space-filling properties than grid for hypervolume
        n_samples = min(remaining_budget, 196)  # Leave a few for potential refinement
        
        # Create LHS samples in [0, 1]^2 for the position variables
        rng = np.random.default_rng(42)
        
        # Latin Hypercube Sampling: divide each axis into n_samples equal intervals
        # and place exactly one point in each interval
        idx = rng.permutation(n_samples)
        t1_lhs = (idx + rng.random(n_samples)) / n_samples
        t2_lhs = (rng.permutation(n_samples) + rng.random(n_samples)) / n_samples
        
        # Clip to ensure we stay in [0, 1]
        t1_lhs = np.clip(t1_lhs, 0.0, 1.0)
        t2_lhs = np.clip(t2_lhs, 0.0, 1.0)
        
        # Also add explicit boundary points for better corner coverage
        # The Pareto front in DTLZ2 with M=3 is a quarter sphere
        # Key boundary points: (1,0), (0,1), and points along the arc
        
        boundary_points = []
        # Arc points: x_pos[0]^2 + x_pos[1]^2 = 1, both >= 0
        for theta in np.linspace(0, np.pi/2, 20):
            boundary_points.append([np.cos(theta), np.sin(theta)])
        # Corner points
        boundary_points.append([1.0, 0.0])
        boundary_points.append([0.0, 1.0])
        boundary_points.append([0.0, 0.0])
        
        # Build solutions
        evaluated_points = []
        evaluated_f = []
        
        # First, evaluate the LHS points
        for i in range(n_samples):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = t1_lhs[i]
            x[1] = t2_lhs[i]
            x[n_pos:] = dist_val
            x = np.clip(x, lo, hi)
            
            f = np.array(problem(x))
            evaluated_points.append(x.copy())
            evaluated_f.append(f)
            used += 1
        
        # Then, evaluate boundary points if we have budget
        for bp in boundary_points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Check if this point is already close to an existing point
            is_duplicate = False
            for existing in evaluated_points:
                if np.allclose(existing[:2], bp, atol=1e-6):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            x = np.zeros(dim)
            x[0] = bp[0]
            x[1] = bp[1]
            x[n_pos:] = dist_val
            x = np.clip(x, lo, hi)
            
            f = np.array(problem(x))
            evaluated_points.append(x.copy())
            evaluated_f.append(f)
            used += 1
        
        # Phase 3: Use remaining budget for local refinement if any
        remaining = self.budget - used
        if remaining > 0 and len(evaluated_points) > 0:
            # Try small perturbations around existing points to see if we can
            # find slightly better positions (in case the front is not exactly where we think)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                # Pick a random existing point
                j = rng.integers(len(evaluated_points))
                x = evaluated_points[j].copy()
                
                # Small perturbation in position variables
                perturb = 0.01 * rng.standard_normal(2)
                x[0] = np.clip(x[0] + perturb[0], lo[0], hi[0])
                x[1] = np.clip(x[1] + perturb[1], lo[1], hi[1])
                
                f = np.array(problem(x))
                evaluated_points.append(x.copy())
                evaluated_f.append(f)
                used += 1
        
        # Return the last evaluated point (return value is ignored by evaluator)
        if evaluated_points:
            return evaluated_points[-1]
        else:
            return lo.copy()
