import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine split: more LHS for coverage, less local search
        n_lhs = int(self.budget * 0.6)
        n_local = self.budget - n_lhs
        
        # 1. Generate LHS samples using global np.random
        lhs_samples = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            bin_width = (hi[d] - lo[d]) / n_lhs
            offsets = np.random.random(n_lhs) * bin_width
            points_in_bin = lo[d] + (np.arange(n_lhs) + offsets) * bin_width
            np.random.shuffle(points_in_bin)
            lhs_samples[:, d] = points_in_bin
        
        # Evaluate LHS
        archive_X = []
        archive_F = []
        
        for i in range(n_lhs):
            x = lhs_samples[i]
            f = problem(x)
            
            # Check if x is dominated by any in archive
            dominated = False
            for j in range(len(archive_F)):
                if np.all(archive_F[j] <= f) and np.any(archive_F[j] < f):
                    dominated = True
                    break
            
            if not dominated:
                # Remove any points in archive dominated by x
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if np.all(f <= archive_F[j]) and np.any(f < archive_F[j]):
                        continue
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(archive_F[j])
                
                new_archive_X.append(x)
                new_archive_F.append(f)
                archive_X = new_archive_X
                archive_F = new_archive_F
        
        # 2. Local Search with improved mutation and selection
        for _ in range(n_local):
            if not archive_X:
                x = lo + (hi - lo) * np.random.random(self.dim)
                f = problem(x)
                archive_X.append(x)
                archive_F.append(f)
                continue
            
            # Tournament selection: pick 2 random, choose better (non-dominated or closer to ideal)
            if len(archive_X) >= 2:
                i1 = np.random.randint(len(archive_X))
                i2 = np.random.randint(len(archive_X))
                if i1 == i2:
                    i2 = (i2 + 1) % len(archive_X)
                f1, f2 = archive_F[i1], archive_F[i2]
                # Check dominance
                if np.all(f1 <= f2) and np.any(f1 < f2):
                    parent_idx = i1
                elif np.all(f2 <= f1) and np.any(f2 < f1):
                    parent_idx = i2
                else:
                    # Both non-dominated: pick randomly or by crowding (simplified: random)
                    parent_idx = i1 if np.random.random() < 0.5 else i2
            else:
                parent_idx = 0
            
            parent_x = archive_X[parent_idx]
            
            # Adaptive mutation: step size based on current spread
            # Use smaller steps as we converge
            progress = _ / max(1, n_local)
            base_sigma = 0.15 * (1.0 - 0.5 * progress)
            sigma = base_sigma * (hi - lo)
            
            # Gaussian mutation
            child_x = parent_x + np.random.normal(0, 1, self.dim) * sigma
            child_x = np.clip(child_x, lo, hi)
            
            child_f = problem(child_x)
            
            # Check dominance
            dominated = False
            for j in range(len(archive_F)):
                if np.all(archive_F[j] <= child_f) and np.any(archive_F[j] < child_f):
                    dominated = True
                    break
            
            if not dominated:
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if np.all(child_f <= archive_F[j]) and np.any(child_f < archive_F[j]):
                        continue
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(archive_F[j])
                
                new_archive_X.append(child_x)
                new_archive_F.append(child_f)
                archive_X = new_archive_X
                archive_F = new_archive_F
        
        return archive_X
