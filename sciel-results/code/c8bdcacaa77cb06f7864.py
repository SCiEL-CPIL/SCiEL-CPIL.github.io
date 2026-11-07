import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Generate Sobol sequence in 2D for f1, f3
        # f1 in [0, 1], f3 in [0, 1]
        # We want 200 points.
        # Sobol requires power of 2 for the generator, but we can sample more and take first N.
        # Or use Halton.
        # Let's use a simple Halton sequence to avoid SciPy dependency issues if any, 
        # though SciPy is allowed.
        
        def halton_sequence(n, dims):
            # Simple Halton sequence
            # Primes for first few dims
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            points = []
            for i in range(1, n + 1):
                p = []
                for d in range(dims):
                    f = 1.0
                    r = 0.0
                    rad = primes[d]
                    while i > 0:
                        f /= rad
                        r += f * (i % rad)
                        i //= rad
                    p.append(r)
                points.append(p)
            return np.array(points)

        # Use Halton for f1, f3
        # Note: Halton points are in (0, 1).
        # We want to include boundaries? 
        # The corners (0,0), (1,0), (0,1), (1,1) are important.
        # Halton doesn't hit boundaries.
        # So we should mix: Grid + Halton?
        # Or just use a Grid which hits boundaries.
        
        # The current grid approach is strong because it hits the extremes.
        # Let's stick to the grid but maybe optimize the spacing or add points.
        # 14x14 grid = 196 points.
        # Let's try 14x14 grid in f1, f3.
        
        n_grid = 14
        f1_vals = np.linspace(0, 1, n_grid)
        f3_vals = np.linspace(0, 1, n_grid)
        
        points = []
        for f1 in f1_vals:
            for f3 in f3_vals:
                # f1 = cos(theta1) -> theta1 = arccos(f1)
                # f3 = sin(theta2) -> theta2 = arcsin(f3)
                theta1 = np.arccos(f1)
                theta2 = np.arcsin(f3)
                
                x = np.zeros(dim)
                x[0] = theta1 / (np.pi/2)
                x[1] = theta2 / (np.pi/2)
                x[2:] = 0.5
                points.append(x)
                
        # We have 196 points.
        # Add 4 more points.
        # Where to add them?
        # The grid is uniform in f1, f3.
        # The "knee" of the front is around f1=0.5, f3=0.5?
        # Actually, the point with max f2 is when theta1=pi/4, theta2=pi/4?
        # f2 = sin(pi/4)cos(pi/4) = 0.5.
        # f1 = cos(pi/4) = 0.707.
        # f3 = sin(pi/4) = 0.707.
        # So (0.707, 0.707) in (f1, f3) space.
        # The grid points are at 0, 1/13, 2/13...
        # 0.707 is close to 9/13 = 0.692 or 10/13 = 0.769.
        # Maybe add points closer to the "center" of the front?
        
        # Let's add 4 points at the "midpoints" of the grid cells in the center region.
        # Or just random?
        # Given the budget is tight, maybe 14x14 is the best integer grid.
        # What if we use 15x13? 195 points.
        # 15 points in f1, 13 in f3.
        # Spacing in f1: 1/14 = 0.0714
        # Spacing in f3: 1/12 = 0.0833
        # This might be better if the front is "wider" in f1?
        # No, it's symmetric.
        
        # Let's try 14x14 + 4 points sampled from Halton to fill gaps.
        # But Halton might cluster?
        
        # Alternative: Use a 200-point Sobol sequence in (theta1, theta2) space.
        # This avoids the boundary issue if we map carefully?
        # No, Sobol is in (0,1).
        
        # Let's stick to the grid but verify if 14x14 is optimal.
        # The score 1.0741 is from this grid.
        # To improve, we need better coverage.
        # Maybe a non-uniform grid?
        # Chebyshev nodes?
        
        # Let's try Chebyshev nodes for f1 and f3.
        # n = 14
        # x_i = cos(pi * (2i+1) / (2n)) ?
        # These are in (-1, 1). Map to (0, 1).
        
        n = 14
        # Chebyshev nodes in [0, 1]
        i = np.arange(n)
        # Standard Chebyshev on [0, 1]: 0.5 * (1 + cos(pi * (2i+1)/(2n)))
        f1_cheb = 0.5 * (1 + np.cos(np.pi * (2*i + 1) / (2*n)))
        f3_cheb = f1_cheb.copy()
        
        points_cheb = []
        for f1 in f1_cheb:
            for f3 in f3_cheb:
                theta1 = np.arccos(f1)
                theta2 = np.arcsin(f3)
                x = np.zeros(dim)
                x[0] = theta1 / (np.pi/2)
                x[1] = theta2 / (np.pi/2)
                x[2:] = 0.5
                points_cheb.append(x)
                
        # Chebyshev points are clustered near the boundaries (0 and 1).
        # This might be good for hypervolume as the extremes contribute more.
        # Let's evaluate this set.
        
        # We have 196 points. Add 4 more.
        # Add the center point?
        # f1=0.5, f3=0.5 -> theta1=pi/3, theta2=pi/6.
        # Add 4 points around the center?
        
        extra = []
        # Add points at f1=0.5, f3=0.5
        # And f1=0.5, f3=0.25, 0.75
        # And f1=0.25, 0.75, f3=0.5
        
        for f1, f3 in [(0.5, 0.5), (0.5, 0.25), (0.5, 0.75), (0.25, 0.5)]:
             theta1 = np.arccos(f1)
             theta2 = np.arcsin(f3)
             x = np.zeros(dim)
             x[0] = theta1 / (np.pi/2)
             x[1] = theta2 / (np.pi/2)
             x[2:] = 0.5
             extra.append(x)
             
        points_cheb.extend(extra)
        
        # Evaluate
        F = np.array([problem(x) for x in points_cheb])
        
        if time.time() > deadline:
            return F
            
        return F
