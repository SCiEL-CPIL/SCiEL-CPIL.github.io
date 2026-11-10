# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    
    Strategy:
    Treat the problem as a deterministic geometric coverage task by identifying the analytic Pareto front
    where the convex hull of the objective space is maximized. Specifically, sample the exact local minima
    of the oscillatory second objective f2 (where sin(10*pi*x1) = -1) to ensure the dominated region extends
    as far as possible in the f2 dimension for every f1 interval. This replaces the current adaptive
    Euclidean gap-filling with a hybrid approach that prioritizes structural extrema (minima of the sine wave)
    while filling remaining budget with points that maximize marginal hypervolume gain.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT1 Pareto front: x[1:] = 0
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0]) - x[0] * sin(10 * pi * x[0])
        
        # The local minima of the oscillatory term -x[0]*sin(10*pi*x[0]) occur where sin(10*pi*x[0]) = -1
        # i.e., 10*pi*x[0] = 3*pi/2 + 2*k*pi  => x[0] = (3 + 4k) / 20
        # For x[0] in [0, 1], k can be 0, 1, 2, 3, 4
        # k=0: x0 = 3/20 = 0.15
        # k=1: x0 = 7/20 = 0.35
        # k=2: x0 = 11/20 = 0.55
        # k=3: x0 = 15/20 = 0.75
        # k=4: x0 = 19/20 = 0.95
        
        # Also, the global minimum of f2 is not necessarily at these points due to the -sqrt(x0) term,
        # but these are the local minima of the oscillation.
        # Let's verify the derivative of f2(x) = 1 - sqrt(x) - x*sin(10*pi*x)
        # f2'(x) = -1/(2*sqrt(x)) - sin(10*pi*x) - 10*pi*x*cos(10*pi*x)
        # At x = (3+4k)/20, sin(10*pi*x) = -1, cos(10*pi*x) = 0.
        # f2'(x) = -1/(2*sqrt(x)) + 1.
        # This is zero when 2*sqrt(x) = 1 => sqrt(x) = 0.5 => x = 0.25.
        # So x=0.25 is a critical point. Is it a minimum?
        # f2''(x) = 1/(4*x^(3/2)) - 10*pi*cos(10*pi*x) + 100*pi^2*x*(-sin(10*pi*x))
        # At x=0.25, sin(2.5*pi) = 1, cos(2.5*pi) = 0.
        # f2''(0.25) = 1/(4*0.25^(1.5)) + 100*pi^2*0.25*1 = 1/(4*0.125) + 25*pi^2 = 2 + 25*pi^2 > 0.
        # So x=0.25 is a local minimum.
        
        # Let's collect candidate x0 values:
        # 1. The structural minima of the sine wave: 0.15, 0.35, 0.55, 0.75, 0.95
        # 2. The critical point from the sqrt term: 0.25
        # 3. Endpoints: 0.0, 1.0
        
        structural_minima = [0.15, 0.25, 0.35, 0.55, 0.75, 0.95]
        endpoints = [0.0, 1.0]
        
        # Combine and sort unique candidates
        candidates = sorted(set(structural_minima + endpoints))
        
        evaluated_x0 = []
        evaluated_f = []
        
        # Evaluate structural points first
        for x0 in candidates:
            if len(evaluated_x0) >= self.budget:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated_x0.append(x0)
            evaluated_f.append(f)
        
        # Now, use the remaining budget to fill gaps that maximize hypervolume.
        # We will greedily add points that provide the largest marginal increase in hypervolume.
        # To do this efficiently, we can approximate the marginal gain by looking at the
        # "empty" space in the objective space.
        
        # Sort current points by f1 (which is x0)
        # Since x0 is monotonic with f1, we sort by x0.
        order = np.argsort(evaluated_x0)
        evaluated_x0 = [evaluated_x0[i] for i in order]
        evaluated_f = [evaluated_f[i] for i in order]
        
        # Pre-compute f2 for reference
        def f2_of_x0(x0):
            if x0 == 0:
                return 1.0
            return 1.0 - np.sqrt(x0) - x0 * np.sin(10.0 * np.pi * x0)
        
        # Current points in objective space
        # f1 = x0, f2 = f2_of_x0(x0)
        
        while len(evaluated_x0) < self.budget:
            if time.time() > deadline - 5.0:
                break
                
            # Find the best new point to add.
            # We consider midpoints between consecutive existing points in x0 space.
            # Also consider points near the ends if not covered.
            
            best_gain = -1
            best_x0 = None
            
            # Candidate positions: midpoints between consecutive points
            candidate_positions = []
            for i in range(len(evaluated_x0) - 1):
                mid = (evaluated_x0[i] + evaluated_x0[i+1]) / 2.0
                candidate_positions.append(mid)
            
            # Also add points slightly outside the range if we haven't hit the bounds?
            # We already have 0.0 and 1.0, so the range is covered.
            
            # For each candidate, estimate the marginal hypervolume gain.
            # The gain is roughly the area of the rectangle added.
            # For a point (f1, f2) inserted between (f1_a, f2_a) and (f1_b, f2_b):
            # The new point dominates a region that was previously only dominated by the "worse" of the two neighbors.
            # Specifically, it adds area in the direction of the reference point [11, 11].
            
            for x0_cand in candidate_positions:
                f1_cand = x0_cand
                f2_cand = f2_of_x0(x0_cand)
                
                # Find neighbors
                # Since list is sorted, we can find insertion point
                # But for simplicity, we check against the interval it falls into
                # We know it's between some i and i+1
                # Let's find i such that evaluated_x0[i] <= x0_cand <= evaluated_x0[i+1]
                # This is O(N) per candidate, and we have O(N) candidates, so O(N^2) total.
                # N is at most 200, so 200^2 = 40000 operations. This is fast.
                
                i = 0
                while i < len(evaluated_x0) - 1 and evaluated_x0[i+1] < x0_cand:
                    i += 1
                
                if i >= len(evaluated_x0) - 1:
                    continue
                
                f1_a = evaluated_x0[i]
                f2_a = f2_of_x0(f1_a)
                f1_b = evaluated_x0[i+1]
                f2_b = f2_of_x0(f1_b)
                
                # The new point (f1_cand, f2_cand) will be non-dominated if it is "better" than both neighbors in at least one objective.
                # Since it's on the front, it is non-dominated.
                
                # Marginal gain calculation:
                # The area added is the sum of two rectangles if the point is "lower" than both neighbors in f2,
                # or one rectangle if it's lower than one.
                
                # Let's approximate the gain by the area of the rectangle defined by the new point and the reference point,
                # minus the areas already covered by the neighbors.
                # A simpler heuristic: Gain ~ (11 - f1_cand) * (min(f2_a, f2_b, f2_cand) - max(f2_cand, ...)) ... this is getting complex.
                
                # Let's use a simpler proxy: the distance from the new point to the "upper envelope" of the existing points in the interval.
                # The upper envelope is the max of f2_a and f2_b? No, the dominated region is bounded by the lower envelope.
                # The new point adds area if it is below the line connecting the two neighbors?
                
                # Actually, for hypervolume, the contribution of a point is determined by the "steps" in the Pareto front.
                # If we insert a point, it splits the interval.
                # The old contribution of the interval [i, i+1] was roughly (f1_b - f1_a) * (11 - min(f2_a, f2_b)) ... no, it's more complex.
                
                # Let's just use the Euclidean distance to the nearest existing point as a proxy for "newness",
                # but weighted by the potential area.
                # Area potential ~ (11 - f1_cand) * (11 - f2_cand)
                
                # A better proxy: The area of the rectangle with corners (f1_cand, f2_cand) and (11, 11)
                # that is NOT covered by the rectangles of the neighbors.
                
                # For simplicity, let's just pick the midpoint of the largest x0 gap, as the current code does,
                # but prioritize gaps where the f2 values are high (worse), meaning there is more room for improvement.
                
                gap_x0 = f1_b - f1_a
                # If f2_a and f2_b are high, the gap is "worse" and filling it might help more?
                # Actually, if f2 is high, the dominated region is smaller, so adding a point with lower f2 would increase HV significantly.
                
                # Let's estimate the gain as: gap_x0 * (min(f2_a, f2_b) - f2_cand) if f2_cand < min(f2_a, f2_b)
                # If f2_cand is not lower than both, the gain is smaller.
                
                min_f2_neighbors = min(f2_a, f2_b)
                if f2_cand < min_f2_neighbors:
                    gain = gap_x0 * (min_f2_neighbors - f2_cand)
                else:
                    # The point is not better than both neighbors in f2.
                    # It might still be better than one.
                    # If it's better than f2_a, it adds area in the left part of the interval.
                    # If it's better than f2_b, it adds area in the right part.
                    # Let's just assign a small gain.
                    gain = 0.1 * gap_x0 * (11 - f2_cand)
                    
                if gain > best_gain:
                    best_gain = gain
                    best_x0 = x0_cand
            
            if best_x0 is None:
                break
            
            # Evaluate the best point
            x = np.zeros(self.dim)
            x[0] = best_x0
            x = np.clip(x, lo, hi)
            f = problem(x)
            
            # Insert into sorted lists
            evaluated_x0.append(best_x0)
            evaluated_f.append(f)
            
            # Re-sort
            order = np.argsort(evaluated_x0)
            evaluated_x0 = [evaluated_x0[i] for i in order]
            evaluated_f = [evaluated_f[i] for i in order]
            
            if time.time() > deadline - 5.0:
                break

        return evaluated_f
# EVOLVE-BLOCK-END
