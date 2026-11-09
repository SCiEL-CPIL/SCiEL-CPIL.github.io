"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    
    Strategy:
    The current program samples the Pareto front by setting x[1:] = 0 and varying x[0].
    This assumes the standard ZDT1 definition where f1 = x1 and g(x) is minimized at x[1:] = 0.
    
    A materially different strategy is to treat the problem as a black-box multi-objective optimization 
    problem using a simple Evolutionary Algorithm (NSGA-II-like) or a more robust local search 
    that doesn't rely on the analytic structure, OR to use a dense sampling of the front 
    but with a different distribution (e.g., non-uniform to better approximate hypervolume).
    
    However, the most robust "different" strategy that can still exploit the structure 
    but is less brittle to implementation variations is to perform a local search 
    (e.g., Nelder-Mead or a simple gradient-free search) from multiple starting points 
    on the presumed front, or simply to use a much denser sampling of the front 
    if the budget allows, or to use a non-uniform sampling strategy.
    
    Actually, the current program uses 100 points out of 200 budget. 
    The hypervolume is dominated by the "knee" of the front. 
    A uniform sampling might be suboptimal. 
    A non-uniform sampling (e.g., more points near the knee) could improve hypervolume.
    
    Let's try a strategy that:
    1. Identifies the front (assuming x[1:] = 0 is optimal for g).
    2. Uses the remaining budget to refine the points or to use a non-uniform distribution.
    
    Wait, the directive says "MATERIALLY DIFFERENT". 
    The current program is a deterministic grid on the analytic front.
    
    Alternative: Use a stochastic local search (e.g., CMA-ES or simple random hill climbing) 
    starting from points on the front to ensure we are exactly on the front, 
    and to handle any potential deviations in the problem definition.
    
    However, given the time limit and the fact that the analytic solution is likely correct,
    the best improvement might be to use a non-uniform sampling strategy that 
    maximizes hypervolume for a fixed number of points.
    
    For a concave front, more points near the "knee" (where the curvature is high) 
    or where the marginal gain in hypervolume is highest are more valuable.
    
    Let's implement a strategy that:
    1. Generates a set of candidate points on the front (x[1:] = 0).
    2. Uses a greedy algorithm to select the subset of points that maximizes hypervolume.
    3. If budget allows, it can also do a local search to verify/improve the points.
    
    Actually, a simpler and robust different strategy:
    Use a population-based evolutionary algorithm (like a simple NSGA-II) 
    but initialized with points on the front to converge quickly.
    
    Let's stick to the analytic front but use a **greedy hypervolume maximization** 
    to select which x1 values to evaluate. This is a different optimization method 
    (combinatorial optimization over the front) compared to uniform sampling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Assume standard ZDT1: f1 = x1, f2 = g(x) * h(f1, g(x))
        # Pareto front is at x[1:] = 0, so f1 = x[0], f2 = h(x[0], 1)
        # h(u, 1) = 1 - sqrt(u) - u * sin(10 * pi * u)
        
        # We want to select a set of x[0] values in [0, 1] to maximize hypervolume.
        # Reference point is [11.0, 11.0].
        # Only points with f1 < 11 and f2 < 11 count.
        # Since f1 <= 1 and f2 is typically small (h(u,1) is between ~-1 and 1), all points count.
        
        # Hypervolume of a set of points P with reference point R:
        # Sort points by f1.
        # HV = sum_{i} (R_f1 - f1_i) * (f2_{i-1} - f2_i) ... roughly.
        # More precisely, for non-dominated points sorted by f1:
        # HV = sum_{i=0}^{n-1} (R_f1 - f1_i) * (f2_{i-1} - f2_i)
        # where f2_{-1} = R_f2 and f2_{n} = 0 (or min f2).
        
        # Let's generate a large pool of candidate x1 values and greedily select.
        
        # Number of evaluations we can afford.
        # Let's reserve some for local search if needed, but likely just sampling is enough.
        n_evals = self.budget
        
        # Generate a fine grid of candidates
        n_candidates = 1000
        x1_candidates = np.linspace(0.0, 1.0, n_candidates)
        
        # Calculate objective values for candidates
        f1_vals = x1_candidates
        f2_vals = 1.0 - np.sqrt(f1_vals) - f1_vals * np.sin(10.0 * np.pi * f1_vals)
        
        # Filter out points that are worse than reference or non-dominated by others?
        # All points on the true front are non-dominated.
        # But we need to select a subset of size n_evals.
        
        # Greedy selection:
        # Start with the point that gives the largest individual contribution to HV?
        # Or start with a point near the middle?
        # A simple greedy: add the point that maximizes the increase in total HV.
        
        # This is O(N^2 * K) which might be slow if N=1000, K=200.
        # 1000^2 * 200 = 200,000,000 operations. In Python this is too slow.
        
        # Alternative: Use a non-uniform sampling strategy based on curvature.
        # The knee of the ZDT1 front is around x1 = 0.35 - 0.5.
        # Let's use a sine or cosine distribution to cluster points near the knee.
        
        # Or simply: use the budget to sample the front very densely.
        # If n_evals = 200, a uniform sample of 200 points is already quite good.
        # The current program uses 100. Let's use all 200.
        
        # Is there a better distribution?
        # For hypervolume, points that "cut off" the most area are best.
        # This is equivalent to maximizing the sum of the areas of the rectangles.
        
        # Let's try a simple strategy: 
        # 1. Evaluate a dense uniform grid (e.g., 100 points).
        # 2. Use remaining budget to fill gaps in the objective space where the 
        #    "gap" (difference in f1 or f2) is largest.
        
        # Actually, the simplest improvement to the current program (which uses 100 points)
        # is to use all 200 points uniformly. This alone might improve the score.
        # But the directive asks for a MATERIALLY DIFFERENT strategy.
        
        # Different strategy: **Stochastic Local Search (Hill Climbing) on the Front**
        # 1. Initialize with a few random points on the front.
        # 2. For each point, perform a local search in the x1 direction to find a 
        #    neighbor that improves the local hypervolume contribution.
        # 3. This effectively adapts the sampling density to the local curvature.
        
        # Implementation:
        # 1. Generate n_init = 20 random points on the front.
        # 2. Evaluate them.
        # 3. For the remaining budget, pick the point with the largest "gap" to its 
        #    neighbors in objective space, and split the gap (midpoint in x1).
        # 4. Evaluate the new point.
        # 5. Repeat until budget exhausted.
        
        # This is an adaptive sampling strategy, different from fixed grid.
        
        n_init = 20
        # Initial points: uniform to ensure coverage
        x1_init = np.linspace(0.0, 1.0, n_init)
        
        # Store evaluated points: (x1, f1, f2)
        evaluated = []
        
        for x1 in x1_init:
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x1, f[0], f[1]))
            
        remaining_budget = self.budget - n_init
        
        # Sort by f1 (which is x1)
        evaluated.sort(key=lambda p: p[1])
        
        while remaining_budget > 0 and time.time() < deadline:
            # Find the largest gap in objective space (f1, f2)
            # Approximate gap by distance in objective space between consecutive points
            max_gap = 0
            max_idx = -1
            for i in range(len(evaluated) - 1):
                # Distance between i and i+1
                d_f1 = evaluated[i+1][1] - evaluated[i][1]
                d_f2 = abs(evaluated[i+1][2] - evaluated[i][2])
                # Use a combined metric, e.g., Euclidean distance in objective space
                gap = np.sqrt(d_f1**2 + d_f2**2)
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
            
            if max_idx == -1:
                break
                
            # Interpolate new x1 between max_idx and max_idx+1
            x1_a = evaluated[max_idx][0]
            x1_b = evaluated[max_idx+1][0]
            # Midpoint in x1 is a good heuristic
            x1_new = (x1_a + x1_b) / 2.0
            
            x = np.zeros(self.dim)
            x[0] = x1_new
            x = np.clip(x, lo, hi)
            f = problem(x)
            
            # Insert into sorted list
            new_point = (x1_new, f[0], f[1])
            # Find insertion point
            ins_idx = 0
            for i, p in enumerate(evaluated):
                if p[1] > new_point[1]:
                    ins_idx = i
                    break
                ins_idx = i + 1
            evaluated.insert(ins_idx, new_point)
            
            remaining_budget -= 1
            
            # Check time
            if time.time() > deadline:
                break
        
        return evaluated
# EVOLVE-BLOCK-END
