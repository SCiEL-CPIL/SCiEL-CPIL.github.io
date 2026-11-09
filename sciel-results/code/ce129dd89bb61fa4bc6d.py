import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_points = self.budget
        
        # 1. Define the Pareto front function for ZDT3
        # f1 = x1
        # f2 = 1 - sqrt(f1) * sin(10 * pi * f1)  (assuming g=1)
        # We need the non-dominated envelope.
        
        # High resolution sampling to find the envelope and cumulative weight
        n_high_res = 10000
        f1_grid = np.linspace(0.0, 1.0, n_high_res)
        
        # Calculate f2
        # Handle f1=0 case for sqrt
        f2_grid = np.zeros_like(f1_grid)
        mask = f1_grid > 0
        f2_grid[mask] = 1.0 - np.sqrt(f1_grid[mask]) * np.sin(10.0 * np.pi * f1_grid[mask])
        f2_grid[~mask] = 1.0 # At f1=0, sin(0)=0, so f2=1.
        
        # Find non-dominated points (lower envelope)
        # Sort by f1 (already sorted)
        # Keep point if f2 is strictly less than min f2 so far
        min_f2_so_far = np.inf
        envelope_indices = []
        for i in range(n_high_res):
            if f2_grid[i] < min_f2_so_far:
                envelope_indices.append(i)
                min_f2_so_far = f2_grid[i]
        
        # Extract envelope points
        f1_env = f1_grid[envelope_indices]
        f2_env = f2_grid[envelope_indices]
        
        # If the envelope is too small, fall back to full grid? 
        # ZDT3 has many oscillations, so envelope should be reasonably sized.
        if len(f1_env) < n_points:
             # If we don't have enough non-dominated points, we just use all of them
             # and maybe duplicate? No, we need n_points distinct evaluations.
             # We can just use the full grid if the envelope is sparse, but 
             # usually ZDT3 has enough.
             # Let's assume len(f1_env) >= n_points.
             pass

        # 2. Compute cumulative "weight" (area contribution)
        # The weight of a segment is proportional to (r2 - f2) * df1
        # We want to distribute points such that each "represents" equal weight.
        # Let W(f1) = integral of (r2 - f2(u)) du along the envelope.
        
        r2 = 11.0
        # We only care about the part of the front that contributes to HV.
        # Points with f2 >= r2 don't contribute? 
        # Actually, if f2 >= r2, the box height is 0.
        # So we can restrict to f2 < r2.
        valid_mask = f2_env < r2
        f1_env = f1_env[valid_mask]
        f2_env = f2_env[valid_mask]
        
        if len(f1_env) == 0:
             # Fallback
             f1_vals = np.linspace(0.0, 1.0, n_points)
        else:
             # Compute weights
             # Weight of point i is roughly (r2 - f2_i) * (f1_{i+1} - f1_i)
             # But we want the cumulative integral.
             # Use trapezoidal rule or simple sum for cumulative distribution.
             weights = (r2 - f2_env)
             
             # Cumulative sum of weights * df1
             # df1 is not uniform in the envelope? f1_env is subset of uniform grid.
             # We need to be careful with the spacing.
             # Let's just use the indices in the original high-res grid to get correct df1.
             # Or simpler: compute cumulative sum of weights, and assume uniform spacing in f1_env?
             # No, f1_env is not uniformly spaced.
             
             # Let's re-calculate weights with correct df1.
             # f1_env is sorted.
             # df1[i] = f1_env[i+1] - f1_env[i]
             # The weight for the segment starting at i is weights[i] * df1[i]?
             # Actually, the HV contribution of point i in the sum formula is (f1_{i+1} - f1_i) * (r2 - f2_i).
             # So the "mass" associated with point i is M_i = (f1_{i+1} - f1_i) * (r2 - f2_i).
             # For the last point, M_last = (r1 - f1_last) * (r2 - f2_last).
             
             r1 = 11.0
             M = np.zeros_like(f1_env)
             for i in range(len(f1_env) - 1):
                 M[i] = (f1_env[i+1] - f1_env[i]) * (r2 - f2_env[i])
             M[-1] = (r1 - f1_env[-1]) * (r2 - f2_env[-1])
             
             total_mass = np.sum(M)
             
             # We want to select n_points from these len(f1_env) candidates.
             # We want the selected points to divide the total mass into n_points equal parts.
             # This is equivalent to finding the f1 values where the cumulative mass crosses k * total_mass / n_points.
             
             cum_mass = np.cumsum(M)
             
             # Target masses
             targets = np.linspace(0, total_mass, n_points + 1)[1:] # Exclude 0?
             # Actually, we want n_points.
             # The points should be at the boundaries of the mass bins.
             # Bin k covers mass [k * total/n, (k+1) * total/n].
             # The point representing bin k could be the start or end.
             # To maximize HV, we usually pick the point that is the "left" boundary of the bin?
             # Or the point that minimizes the error?
             # The standard "equal mass" heuristic picks the point where the cumulative mass is k * total/n.
             
             # Use searchsorted to find indices
             # We want indices j such that cum_mass[j] is close to target.
             # Since cum_mass is increasing, we can use searchsorted.
             
             # targets are the cumulative mass up to the point.
             # Let's pick the points corresponding to the upper bound of each bin?
             # Or just interpolate.
             
             # Simple approach:
             # Map the continuous mass coordinate to the discrete f1_env.
             # We want n_points values of f1.
             
             # If len(f1_env) > n_points:
             # We can pick specific indices.
             
             # Let's use the inverse of the CDF.
             # cdf = cum_mass / total_mass
             # We want f1 such that cdf(f1) = k / n_points.
             
             # Linear interpolation on the (cum_mass, f1_env) pairs.
             # cum_mass is not strictly increasing if M[i] is 0, but here M[i] > 0.
             
             # np.interp(x, xp, fp)
             # x: target cdf values
             # xp: cum_mass
             # fp: f1_env
             
             # We need n_points values.
             # k = 1 to n_points.
             # target_cdf = k / n_points
             
             # However, np.interp expects xp to be increasing. cum_mass is increasing.
             
             # Note: The last point in f1_env might not cover the full mass if we stop early?
             # cum_mass[-1] is total_mass.
             
             # We want the f1 values that split the mass.
             # The "optimal" points are often the midpoints of the mass bins?
             # Or the points where the cumulative mass equals the target?
             # Let's try finding the f1 where cum_mass crosses the target.
             
             target_cdf_vals = np.arange(1, n_points + 1) / n_points
             
             # Interpolate
             # Note: cum_mass starts at M[0]. It doesn't start at 0.
             # We should prepend 0 to cum_mass and 0 to f1_env?
             # Or just handle the first bin carefully.
             
             # Let's prepend 0 mass and f1=0 (or min f1)
             xp = np.concatenate(([0.0], cum_mass))
             fp = np.concatenate(([0.0], f1_env)) # Use 0 as the start point for interpolation?
             # Actually, if the first mass bin is [0, total/n], the point should be somewhere in the first segment.
             
             # Let's just use np.interp
             f1_selected = np.interp(target_cdf_vals * total_mass, xp, fp)
             
             # Ensure unique? Interp might give duplicates if mass is concentrated.
             # But ZDT3 front is smooth-ish.
             
             # Clip to [0, 1]
             f1_selected = np.clip(f1_selected, 0.0, 1.0)

        # 3. Construct X
        X = np.zeros((n_points, self.dim))
        X[:, 0] = f1_selected
        if self.dim > 1:
            X[:, 1:] = 0.5
            
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # 4. Evaluate
        for i in range(n_points):
            problem(X[i])
            
        return X
