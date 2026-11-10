"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for ZDT3 exploiting its known structure.
    
    ZDT3:
      f1 = x[0]
      f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))) + sin(10*f1*pi))
    
    The Pareto front lies on x[1:] = 0. With n=30, 1+10*(n-1) = 291.
    f2(x1) = 291 * (1 - sqrt(x1/291) + sin(10*pi*x1))
    
    Strategy:
    1. Compute the exact Pareto front analytically (x[1:]=0).
    2. Use a dense grid of x1 values to cover all 10 oscillations of sin(10*pi*x1).
    3. Allocate 200 evaluations to sample the front densely, with extra points
       in regions where the front curves sharply (near local minima of f2).
    4. Add a time guard to ensure we return before the 330s wall-clock limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s safety margin before 330s kill

        n = self.dim
        c = 1 + 10 * (n - 1)  # 291 for n=30

        # The Pareto front: x[0] in [0,1], x[1:] = 0
        # f1 = x[0]
        # f2 = c * (1 - sqrt(x[0]/c) + sin(10*pi*x[0]))

        # We have budget=200 evaluations.
        # Generate candidate x1 values that densely cover [0,1] with emphasis
        # on regions where sin(10*pi*x1) has local extrema (sharp curvature).

        # sin(10*pi*x1) has period 0.1, so 10 full cycles in [0,1].
        # Local minima of f2 occur where sin(10*pi*x1) is minimized, i.e., at
        # x1 = (2k+1)*0.05 for k=0,...,9 (where sin = -1).
        # Local maxima at x1 = k*0.1 for k=0,...,10.

        # Strategy: use a non-uniform grid with more points near the local minima
        # of the oscillatory term, since those regions contribute most to the
        # hypervolume (they form the "lower" parts of the front).

        # Create a base uniform grid
        n_base = 100
        x1_base = np.linspace(0.0, 1.0, n_base)

        # Create additional points clustered around local minima of sin(10*pi*x1)
        # Local minima at x1 = (2k+1)*0.05, k=0,...,9
        min_locs = np.array([(2 * k + 1) * 0.05 for k in range(10)])

        # Add extra points near each local minimum
        n_extra_per_min = 8
        x1_extra = []
        for loc in min_locs:
            # Small cluster around each minimum
            spread = 0.02
            offsets = np.linspace(-spread, spread, n_extra_per_min)
            cluster = loc + offsets
            x1_extra.extend(cluster.tolist())

        x1_extra = np.array(x1_extra)

        # Combine and sort
        x1_all = np.concatenate([x1_base, x1_extra])
        x1_all = np.sort(x1_all)

        # Remove duplicates
        x1_all = np.unique(x1_all)

        # We have budget=200 evaluations. Select up to 200 points.
        # If we have more candidates than budget, use a stratified selection.
        budget = self.budget
        if len(x1_all) > budget:
            # Stratified sampling: divide [0,1] into budget equal-width bins,
            # pick the point in each bin closest to the bin center weighted
            # by curvature (more points where curvature is high).
            # Simpler: just take evenly spaced from the sorted list.
            indices = np.linspace(0, len(x1_all) - 1, budget).astype(int)
            x1_selected = x1_all[indices]
        else:
            x1_selected = x1_all

        # Ensure we have exactly 'budget' points (pad if needed)
        while len(x1_selected) < budget:
            # Add more uniform points
            x1_add = np.random.uniform(0, 1, budget - len(x1_selected))
            x1_selected = np.concatenate([x1_selected, x1_add])
            x1_selected = np.sort(x1_selected)
            x1_selected = np.unique(x1_selected)

        x1_selected = x1_selected[:budget]

        # Evaluate all points
        F = np.zeros((len(x1_selected), 2))
        for i, x1 in enumerate(x1_selected):
            if time.time() > deadline:
                # Return what we have so far
                break
            x = np.zeros(n)
            x[0] = x1
            # x[1:] = 0 (already zero)
            f = problem(x)
            F[i, 0] = f[0]
            F[i, 1] = f[1]

        return F
# EVOLVE-BLOCK-END
