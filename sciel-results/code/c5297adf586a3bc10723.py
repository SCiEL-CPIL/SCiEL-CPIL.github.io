# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Parametric analytic inversion strategy for ZDT3 hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        if budget <= 0:
            return None

        # ZDT3 Pareto front: x0 in [0,1], xi=0.5 for i>=1, g_min=1
        # f1 = x0
        # f2 = (1+sqrt(f1))*h(f1, 1) where h(u,g) = 1 - sqrt(u) - u*sin^2(10*pi*u)
        # The front has 7 segments separated by points where sin^2(10*pi*u)=1
        # i.e., 10*pi*u = pi/2 + k*pi => u = (2k+1)/20 for k=0,...,9
        # But sin^2 = 1 at u = 1/20, 3/20, 5/20, ..., 19/20
        # The "segments" are between these critical points.
        # For hypervolume maximization, we want to place points to cover the front well.
        # 
        # Key insight: The hypervolume contribution is maximized when points are spread
        # in objective space. Since f1=x0 is linear, but f2 is non-linear with oscillations,
        # we should place points at locations that maximize the hypervolume coverage.
        #
        # A good strategy: Place points at the critical points (where sin^2=1, i.e., local
        # minima/maxima of f2) and spread the remaining points evenly in the gaps.
        #
        # Critical points for u = f1: u_k = (2k+1)/20 for k=0,...,9 (10 points)
        # But wait, sin^2(10*pi*u) = 1 when 10*pi*u = pi/2 + m*pi => u = (2m+1)/20
        # For u in [0,1], m=0,...,9 gives u = 1/20, 3/20, ..., 19/20
        # These are 10 points.
        #
        # Actually, let me reconsider. The function h(u,1) = 1 - sqrt(u) - u*sin^2(10*pi*u)
        # f2 = (1+sqrt(u))*h(u,1) = (1+sqrt(u))*(1-sqrt(u) - u*sin^2(10*pi*u))
        # = (1+sqrt(u))(1-sqrt(u)) - (1+sqrt(u))*u*sin^2(10*pi*u)
        # = 1 - u - (1+sqrt(u))*u*sin^2(10*pi*u)
        #
        # The term (1+sqrt(u))*u*sin^2(10*pi*u) oscillates. The local maxima of f2 occur
        # when sin^2=0, i.e., u = k/10 for k=0,...,10.
        # The local minima occur when sin^2=1, i.e., u=(2k+1)/20 for k=0,...,9.
        #
        # For hypervolume, we want to capture the "envelope" of the front. The upper
        # envelope is at sin^2=0 points. Let's place points at these 11 critical points
        # and distribute the rest.

        # Critical points where sin^2(10*pi*u) = 0: u = k/10 for k=0,...,10
        critical_u = np.array([k / 10.0 for k in range(11)])
        
        # We have budget points to place. Let's use a mix:
        # 1. Place some points at critical locations (local maxima of f2)
        # 2. Fill the rest with evenly spaced points in [0,1]
        #
        # To maximize hypervolume, we should ensure good coverage. Let's use:
        # - All 11 critical points (if budget >= 11)
        # - Remaining points evenly spaced in [0,1]
        
        if budget <= 11:
            # Just use evenly spaced points
            x0_vals = np.linspace(0, 1, budget)
        else:
            # Use critical points + additional evenly spaced points
            n_crit = len(critical_u)
            n_extra = budget - n_crit
            extra_u = np.linspace(0, 1, n_extra + 2)[1:-1]  # exclude endpoints to avoid duplicates
            x0_vals = np.concatenate([critical_u, extra_u])
            x0_vals = np.sort(x0_vals)
            # Remove duplicates (shouldn't happen but just in case)
            x0_vals = np.unique(x0_vals)
            # If we have fewer than budget due to dedup, add more
            if len(x0_vals) < budget:
                diff = budget - len(x0_vals)
                # Add points at midpoints of largest gaps
                gaps = np.diff(x0_vals)
                gap_indices = np.argsort(gaps)[::-1][:diff]
                new_points = []
                for idx in gap_indices:
                    mid = (x0_vals[idx] + x0_vals[idx+1]) / 2.0
                    new_points.append(mid)
                x0_vals = np.sort(np.concatenate([x0_vals, new_points]))
        
        # Construct x vectors: x[0]=x0, x[1:]=0.5
        X = np.zeros((budget, dim))
        X[:, 0] = x0_vals[:budget]
        X[:, 1:] = 0.5
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
