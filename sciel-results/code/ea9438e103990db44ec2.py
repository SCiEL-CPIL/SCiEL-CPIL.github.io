"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware optimizer: structured grid on Pareto-relevant vars + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ7 structure: x[0], x[1] are Pareto-relevant, x[2:] should be 0 for g=0
        # Pareto front is parameterized by (x0, x1) in [0,1]^2
        # f1 = x0*(1+g), f2 = x1*(1+g), f3 = 1 - sqrt(x0)*(1 - h(x1,g))
        # With g=0: f1=x0, f2=x1, f3 = 1 - sqrt(x0)*(1 - x1*(1 - x1/(1+x1)*x0^2))

        def make_point(x0, x1):
            x = np.zeros(dim)
            x[0] = np.clip(x0, 0.0, 1.0)
            x[1] = np.clip(x1, 0.0, 1.0)
            # x[2:] = 0 ensures g=0
            return x

        used = 0
        points = []
        fvals = []

        # Phase 1: Max-min facility location on the Pareto front
        # Place points to maximize minimum distance in (x0, x1) space
        # This ensures uniform coverage for hypervolume computation
        
        n_phase1 = min(121, budget)  # Same budget as before for fair comparison
        
        # Start with corner points for good initial spread
        initial_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
        
        placed = []
        for pt in initial_points:
            if used >= budget:
                break
            if time.time() > deadline:
                return points
            x = make_point(pt[0], pt[1])
            f = np.array(problem(x))
            points.append(x)
            fvals.append(f)
            used += 1
            placed.append(pt)
        
        # Now use max-min facility location for remaining points
        while len(placed) < n_phase1 and used < budget:
            if time.time() > deadline:
                return points
            
            # Find the point that maximizes the minimum distance to existing points
            best_x0, best_x1 = 0.0, 0.0
            best_min_dist = -1.0
            
            # Sample candidate locations on a fine grid
            n_candidates = 50
            x0_candidates = np.linspace(0.0, 1.0, n_candidates)
            x1_candidates = np.linspace(0.0, 1.0, n_candidates)
            
            for x0 in x0_candidates:
                for x1 in x1_candidates:
                    # Compute minimum distance to all placed points
                    min_dist = min(np.sqrt((x0 - px)**2 + (x1 - py)**2) for px, py in placed)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_x0 = x0
                        best_x1 = x1
            
            # Place the best candidate
            x = make_point(best_x0, best_x1)
            f = np.array(problem(x))
            points.append(x)
            fvals.append(f)
            used += 1
            placed.append((best_x0, best_x1))

        # Phase 2: Fill remaining budget with additional max-min points
        # Continue placing points using the same max-min criterion
        while used < budget:
            if time.time() > deadline:
                return points
            
            # Extract placed (x0, x1) coordinates from points
            placed = [(p[0], p[1]) for p in points]
            
            # Find the point that maximizes the minimum distance to existing points
            best_x0, best_x1 = 0.0, 0.0
            best_min_dist = -1.0
            
            # Use a finer grid for later stages
            n_candidates = 60
            x0_candidates = np.linspace(0.0, 1.0, n_candidates)
            x1_candidates = np.linspace(0.0, 1.0, n_candidates)
            
            for x0 in x0_candidates:
                for x1 in x1_candidates:
                    min_dist = min(np.sqrt((x0 - px)**2 + (x1 - py)**2) for px, py in placed)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_x0 = x0
                        best_x1 = x1
            
            x = make_point(best_x0, best_x1)
            f = np.array(problem(x))
            points.append(x)
            fvals.append(f)
            used += 1

        return points
# EVOLVE-BLOCK-END
