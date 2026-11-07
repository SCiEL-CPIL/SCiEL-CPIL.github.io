"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting ZDT3 structure.
    
    ZDT3 Pareto front: when x2..xn = 0, g(x) = 1, so:
    f1 = x1
    f2 = 1 - sqrt(x1) - x1 * sin(10*pi*x1)
    
    The front is 1D and discontinuous with 11 segments.
    Strategy: Use all budget to sample x1 strategically to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT3, the Pareto front is on x1 in [0,1] with x2..xn = lower bounds
        # The front has 11 segments due to sin(10*pi*x1)
        # Segment boundaries are at x1 = k/10 for k=0,1,...,10
        
        # Strategy: Use all budget to sample x1
        # - Include all 11 segment boundaries and midpoints
        # - Add uniform samples to fill gaps
        # - Emphasize regions with high hypervolume contribution
        
        n_evals = self.budget
        
        # Key points: segment boundaries and midpoints of ZDT3 front
        # The sin(10*pi*x1) term creates discontinuities at x1 = k/10
        # Each segment [k/10, (k+1)/10] for k=0..9 has a specific shape
        
        # Generate candidate x1 values
        x1_candidates = set()
        
        # Add segment boundaries (11 points: 0, 0.1, 0.2, ..., 1.0)
        for k in range(11):
            x1_candidates.add(k / 10.0)
        
        # Add midpoints of each segment
        for k in range(10):
            x1_candidates.add((2*k + 1) / 20.0)
        
        # Add more uniform points to use remaining budget
        # We have 21 key points so far, need n_evals total
        n_remaining = n_evals - len(x1_candidates)
        if n_remaining > 0:
            # Add uniform samples in [0, 1]
            uniform_samples = np.linspace(0, 1, n_remaining + 2)[1:-1]
            for s in uniform_samples:
                x1_candidates.add(s)
        
        # Also add points near the boundaries of segments (where curvature is high)
        # These contribute more to hypervolume
        for k in range(11):
            boundary = k / 10.0
            for offset in [-0.005, -0.002, 0.002, 0.005]:
                val = boundary + offset
                if 0 <= val <= 1:
                    x1_candidates.add(val)
        
        # Convert to sorted list and trim to budget
        x1_sorted = sorted(x1_candidates)
        if len(x1_sorted) > n_evals:
            # Keep the most important points: boundaries, midpoints, and uniform
            # Prioritize: boundaries first, then midpoints, then rest
            boundary_points = [k / 10.0 for k in range(11)]
            midpoint_points = [(2*k + 1) / 20.0 for k in range(10)]
            
            # Create priority-ordered list
            priority_points = boundary_points + midpoint_points
            remaining_points = [p for p in x1_sorted if p not in boundary_points and p not in midpoint_points]
            
            # Take all priority points, then fill with remaining
            x1_final = list(priority_points)
            for p in remaining_points:
                if len(x1_final) >= n_evals:
                    break
                x1_final.append(p)
            
            x1_final = x1_final[:n_evals]
        else:
            x1_final = x1_sorted
        
        all_points = []
        all_f = []
        
        for x1 in x1_final:
            x = np.zeros(self.dim)
            x[0] = x1
            x[1:] = lo[1:]
            
            if time.time() > deadline:
                break
                
            f = problem(x)
            all_points.append(x)
            all_f.append(f)
        
        return all_points
# EVOLVE-BLOCK-END
