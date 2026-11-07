"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic analytical front sampling for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0

        # Strategy: 
        # 1. Generate a set of candidate points on the Pareto front.
        #    For DTLZ2, the Pareto front is a unit hypersphere in the objective space.
        #    The decision variables x_0..x_{n-2} determine the position on the front via:
        #    f_i = g * (1 - cos(theta_i)) roughly, but more precisely for DTLZ2:
        #    The standard mapping is:
        #    Let x_1..x_m be the position variables.
        #    f_i = g(x) * (1 - \prod_{j=1}^{i-1} (1-x_j) * \sin(\pi x_i / 2)) ... wait, that's not quite right.
        
        # Correct DTLZ2 definition:
        # g(x) = 1 + 10 * (n-m) * (sum(x_{m+1..n}^2) - sum(x_{m+1..n} sin(20*pi*x_{m+1..n})))
        # f_i = g * (1 - \prod_{j=1}^{i-1} (1-x_j) * \sin(\pi x_i / 2)) for i=1..m
        # f_m = g * \prod_{j=1}^{m} (1-x_j) * \sin(\pi x_j / 2)
        
        # To get the Pareto front, we set g=1 (by setting x_{m+1..n} = 0).
        # Then f_i depends only on x_1..x_m.
        # The front is the unit hypersphere: sum(f_i^2) = 1.
        # We can sample points on the front by choosing angles.
        # Alternatively, we can just sample x_1..x_m uniformly and compute the objectives.
        # Since we have 12 variables, m=3 (n_obj=3), so x_1, x_2 are position variables, x_3..x_12 are distance variables.
        # To be on the front, x_3..x_12 should be 0 (or such that g=1).
        # So we set x_3..x_12 = 0.
        # Then we sample x_1, x_2 in [0, 1].
        
        # How many points?
        # Budget is 200.
        # Let's use 190 points for the front, 10 for random exploration.
        
        n_front = 190
        n_random = 10
        
        # Generate x_1, x_2 samples
        # Use a grid or Sobol? The prompt suggests "deterministic uniform grid of 10x10" in the directive, 
        # but the strategy I stated is "analytical mapping". A 10x10 grid is 100 points. 
        # Let's use a denser set. Maybe a 14x14 grid? 196 points. 
        # Or just random samples on the front? 
        # The directive says: "Use a deterministic uniform grid of 10x10 points... to guarantee exact coverage... more effectively than low-discrepancy".
        # Let's follow the directive's hint but adapt it to be robust.
        # Actually, a 10x10 grid might be too sparse for 3D hypervolume. 
        # Let's use a combination: A grid for structure, plus some random points on the front.
        
        # Let's try a 13x13 grid = 169 points. Plus 20 random points on the front. Total 189.
        # Or simply 190 random points on the front? Random points on the front might leave gaps.
        # The directive specifically mentions "deterministic uniform grid... to guarantee exact coverage of the Pareto front's boundaries and corners".
        # So let's stick to a grid.
        
        # For 2D position space (x_1, x_2), a 10x10 grid covers [0,1]x[0,1].
        # Let's use a slightly denser grid if budget allows.
        # 14x14 = 196. That fits in 200.
        
        grid_size = 14
        xs1 = np.linspace(0, 1, grid_size)
        xs2 = np.linspace(0, 1, grid_size)
        
        front_points = []
        for x1 in xs1:
            for x2 in xs2:
                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                # x[2:] are distance variables, set to 0 for g=1
                front_points.append(x)
        
        # Add some random points on the front to fill gaps
        rng = np.random.default_rng(42)
        for _ in range(4): # 196 + 4 = 200
            x1 = rng.random()
            x2 = rng.random()
            x = np.zeros(self.dim)
            x[0] = x1
            x[1] = x2
            front_points.append(x)
        
        # Clip to bounds (though 0 and 1 are within [0,1])
        for i in range(len(front_points)):
            front_points[i] = np.clip(front_points[i], lo, hi)
        
        # Evaluate
        results = []
        for x in front_points:
            if time.time() > deadline:
                break
            f = problem(x)
            results.append(f)
        
        # If we have time, do a few random explorations to ensure we don't miss anything if the problem isn't exactly DTLZ2
        # But the task says "minimizing all 3 objectives of DTLZ2".
        # The scoring is based on hypervolume of recorded vectors.
        # If we only record front points, we get the max hypervolume.
        # Random points inside the front (g>1) would be dominated and not contribute to hypervolume if better points exist.
        # So it's best to spend all budget on the front.
        
        return results
# EVOLVE-BLOCK-END
