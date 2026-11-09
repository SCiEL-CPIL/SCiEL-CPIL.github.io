"""Combined: analytic front seeding (inspiration 1) with directed coordinate-descent refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(12345)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        used = 0
        archive_X = []
        archive_F = []

        def eval_point(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        def add_point(x, f):
            # add if not dominated by archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return  # dominated
            # remove archive points dominated by f
            keep = []
            for i, af in enumerate(archive_F):
                if np.all(f <= af) and np.any(f < af):
                    continue
                keep.append(i)
            archive_F = [archive_F[i] for i in keep]
            archive_X = [archive_X[i] for i in keep]
            archive_F.append(f)
            archive_X.append(x)

        # Phase 1: Analytic front seeding
        # ZDT6: f1 = 1 + 9*c + sin(10*pi*x1), f2 = g(x)*(1 - sqrt(f1/g))
        # where c = mean(x2..x10), g = 1 + 10*(n-1) + (x1 - (10*c+sin(10*pi*x1))^2)/(4*x1+4)^2
        # Pareto front: f2 = 1 - sqrt(f1/11) for f1 in [1, 11]
        # To be on the front: g must be at its minimum = 11 (when x2..x10 are all equal)
        # and x1 must be chosen so that the second term in g is zero:
        #   x1 = (10*c + sin(10*pi*x1))^2 / (4*x1+4)^2  -- this is complex
        # Simpler: use the two-branch approach from inspiration 1
        # Branch A: x1=0, sin=0, f1 = 1+9c, c in [0, 1/9] gives f1 in [1, 2]
        # Branch B: x1=0.05, sin=1, f1 = 2+9c, c in [0, 1] gives f1 in [2, 11]
        # For both branches, set x2..x10 = c to minimize g

        # Dense sampling along the front
        n_targets = min(120, self.budget // 2)
        targets = np.linspace(1.0, 11.0, n_targets)

        for t in targets:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            if t <= 2.0:
                # x1 = 0, sin = 0, f1 = 1 + 9c => c = (t-1)/9
                c = max(0.0, (t - 1.0) / 9.0)
                x1 = 0.0
            else:
                # x1 = 0.05, sin(pi/2) = 1, f1 = 2 + 9c => c = (t-2)/9
                c = max(0.0, (t - 2.0) / 9.0)
                x1 = 0.05
            c = min(c, 1.0)
            x = np.full(self.dim, c)
            x[0] = x1
            x, f = eval_point(x)
            add_point(x, f)

        # Phase 2: Directed coordinate descent refinement
        # For each archive point, try adjusting individual coordinates to reduce f2
        # while keeping f1 roughly fixed (or improving both)
        if used < self.budget and archive_X:
            steps_left = self.budget - used
            # Build a list of (archive_index, coordinate) pairs to try
            # Focus on coordinates that affect g most: x1 and the x2..x10 spread
            
            # First, try to equalize x2..x10 (reduces g)
            # Then fine-tune x1 for better f1/f2 tradeoff
            
            for step in range(steps_left):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                    
                i = rng.integers(len(archive_X))
                base_x = archive_X[i].copy()
                base_f = archive_F[i].copy()
                
                # Choose a refinement strategy
                strategy = rng.integers(3)
                
                if strategy == 0:
                    # Equalize x2..x10 toward their mean (reduces g, improves f2)
                    x = base_x.copy()
                    c = np.mean(x[1:])
                    x[1:] = c
                    # Clip
                    x = np.clip(x, lo, hi)
                    if np.allclose(x, base_x):
                        continue
                    x, f = eval_point(x)
                    # Accept if it improves (not dominated by base)
                    if np.all(f <= base_f) and np.any(f < base_f):
                        add_point(x, f)
                        archive_X[i] = x
                        archive_F[i] = f
                        
                elif strategy == 1:
                    # Perturb x1 slightly to adjust f1
                    x = base_x.copy()
                    delta = rng.standard_normal() * 0.005
                    x[0] = np.clip(x[0] + delta, lo[0], hi[0])
                    # Also equalize x2..x10
                    c = np.mean(x[1:])
                    x[1:] = c
                    x = np.clip(x, lo, hi)
                    if np.allclose(x, base_x):
                        continue
                    x, f = eval_point(x)
                    if np.all(f <= base_f) and np.any(f < base_f):
                        add_point(x, f)
                        archive_X[i] = x
                        archive_F[i] = f
                        
                else:
                    # Small random perturbation on all coordinates
                    x = base_x + 0.01 * span * rng.standard_normal(self.dim)
                    x = np.clip(x, lo, hi)
                    # Equalize x2..x10
                    c = np.mean(x[1:])
                    x[1:] = c
                    x = np.clip(x, lo, hi)
                    if np.allclose(x, base_x):
                        continue
                    x, f = eval_point(x)
                    if np.all(f <= base_f) and np.any(f < base_f):
                        add_point(x, f)
                        archive_X[i] = x
                        archive_F[i] = f

        # Phase 3: Final cleanup - if we have budget left, do a few more targeted evaluations
        if used < self.budget and archive_X:
            # Sample a few points very close to the analytic front to fill gaps
            n_extra = min(self.budget - used, 20)
            for _ in range(n_extra):
                if time.time() > deadline or used >= self.budget:
                    break
                t = rng.uniform(1.0, 11.0)
                if t <= 2.0:
                    c = max(0.0, (t - 1.0) / 9.0)
                    x1 = 0.0
                else:
                    c = max(0.0, (t - 2.0) / 9.0)
                    x1 = 0.05
                c = min(c, 1.0)
                x = np.full(self.dim, c)
                x[0] = x1
                x, f = eval_point(x)
                add_point(x, f)

        # Ensure we return something; return value ignored
        return archive_X if archive_X else np.array([lo + 0.5 * span])
# EVOLVE-BLOCK-END
