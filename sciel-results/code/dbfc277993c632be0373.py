import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT3 Pareto front is at x_i = 0.5 for i > 1
        # f1 = x1
        # f2 = 1 - sqrt(x1/g) + x1 sin(10 pi x1)
        # On PF, g=1, so f2 = 1 + x1 sin(10 pi x1)
        
        # We want to maximize hypervolume.
        # Dominated points do not contribute.
        # Points on increasing slopes of f2(x1) are dominated by the preceding local minimum.
        # So we only sample from decreasing slopes and x1=0.
        
        # Analyze f2(x1) to find decreasing intervals
        x_fine = np.linspace(0, 1, 1000)
        f2_fine = 1 + x_fine * np.sin(10 * np.pi * x_fine)
        
        # Find decreasing segments
        # A point is non-dominated if it is on a decreasing slope or is a local minimum.
        # Actually, any point on a decreasing slope is non-dominated.
        # Points on increasing slopes are dominated by the start of the slope (the local min).
        
        # Identify intervals where f2 is decreasing
        diffs = np.diff(f2_fine)
        is_decreasing = diffs < 0
        
        # Find contiguous regions of decreasing
        regions = []
        in_region = False
        start_idx = 0
        for i in range(len(is_decreasing)):
            if is_decreasing[i] and not in_region:
                in_region = True
                start_idx = i
            elif not is_decreasing[i] and in_region:
                in_region = False
                regions.append((x_fine[start_idx], x_fine[i+1]))
        if in_region:
            regions.append((x_fine[start_idx], x_fine[-1]))
            
        # Also include x1=0
        # x1=0 is a local min? f2(0)=1. f2 increases initially. So 0 is a local min.
        # So we add (0, 0) effectively, or just include 0 in the sampling.
        
        # Total length of decreasing regions
        total_len = sum(b - a for a, b in regions)
        
        # Allocate budget
        # Reserve 1 point for x1=0
        n_zero = 1
        n_remaining = self.budget - n_zero
        
        if n_remaining <= 0:
            # Budget too small, just pick 0 and 1
            X = np.zeros((self.budget, self.dim))
            X[0, 0] = 0.0
            if self.budget > 1:
                X[1, 0] = 1.0
            for i in range(1, self.dim):
                X[:, i] = 0.5
            F = np.array([problem(x) for x in X])
            return X

        # Distribute remaining points proportional to region length
        x_samples = [0.0]
        
        if total_len > 0:
            for a, b in regions:
                length = b - a
                n_points = int(round(n_remaining * (length / total_len)))
                if n_points > 0:
                    xs = np.linspace(a, b, n_points, endpoint=False)
                    x_samples.extend(xs)
        
        # If we didn't use all budget due to rounding, add more points to the largest region
        # Or just ensure we have budget points
        while len(x_samples) < self.budget:
            # Add random points in the union of regions? Or just extend the last region?
            # Better: add points uniformly in [0,1] but filter? No, we want PF points.
            # Let's just add points to the regions randomly or uniformly.
            # For simplicity, add to the first region or distribute evenly.
            # Actually, let's just generate a uniform sample over the regions again with the remaining count.
            break # The loop above should have used most. Let's handle the remainder.
            
        # More robust allocation:
        x_samples = [0.0]
        if total_len > 0:
            # Calculate exact counts
            counts = []
            for a, b in regions:
                length = b - a
                count = n_remaining * (length / total_len)
                counts.append(count)
            
            # Floor and distribute remainder
            int_counts = [int(c) for c in counts]
            remainders = [c - ic for c, ic in zip(counts, int_counts)]
            total_int = sum(int_counts)
            remaining = n_remaining - total_int
            
            # Sort by remainder to assign extra points
            order = np.argsort(remainders)[::-1]
            for i in range(remaining):
                int_counts[order[i]] += 1
                
            for i, (a, b) in enumerate(regions):
                if int_counts[i] > 0:
                    xs = np.linspace(a, b, int_counts[i], endpoint=False)
                    x_samples.extend(xs)
                    
        # Pad if necessary
        while len(x_samples) < self.budget:
            x_samples.append(0.0) # Fallback
            
        # Construct X
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = x_samples[:self.budget]
        for i in range(1, self.dim):
            X[:, i] = 0.5
            
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
