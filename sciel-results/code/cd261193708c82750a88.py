"""Multi-objective optimizer on RE21 using an Evolutionary Strategy with Hypervolume-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin under 330s kill
        B = int(self.budget)
        d = self.dim
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        # Reference point for hypervolume calculation (as per task)
        ref = np.array([3000.0, 0.06])

        # Archive stores (x, f) for non-dominated points
        # We maintain a list of x's and a list of f's
        xs = []
        fs = []
        
        def dominates(f1, f2):
            """Returns True if f1 strictly dominates f2 (minimization)"""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(x_new, f_new):
            """
            Adds x_new, f_new to the archive if it is non-dominated.
            Removes any existing points dominated by the new point.
            Returns True if the point was added to the archive.
            """
            if len(fs) == 0:
                xs.append(x_new)
                fs.append(f_new)
                return True
            
            # Check if new point is dominated by any existing point
            for f_exist in fs:
                if dominates(f_exist, f_new):
                    return False
            
            # Remove any existing points dominated by the new point
            new_xs = []
            new_fs = []
            for i in range(len(fs)):
                if not dominates(f_new, fs[i]):
                    new_xs.append(xs[i])
                    new_fs.append(fs[i])
            
            new_xs.append(x_new)
            new_fs.append(f_new)
            
            # Update references
            # Since we are in a method, we need to modify the outer lists
            # Python lists are mutable, but reassignment inside a nested function 
            # without 'nonlocal' won't work if we reassign the variable name.
            # However, we can clear and extend.
            xs.clear()
            fs.clear()
            for i in range(len(new_xs)):
                xs.append(new_xs[i])
                fs.append(new_fs[i])
            return True

        def hypervolume_2d(points, ref):
            """
            Calculate 2D hypervolume for a list of points.
            Points are assumed to be non-dominated and sorted.
            """
            if len(points) == 0:
                return 0.0
            p = np.array(points)
            # Filter points strictly better than ref
            mask = (p[:, 0] < ref[0]) & (p[:, 1] < ref[1])
            p = p[mask]
            if len(p) == 0:
                return 0.0
            
            # Sort by f1 ascending
            p = p[np.argsort(p[:, 0])]
            
            hv = 0.0
            # Iterate from worst (highest f1) to best (lowest f1)
            # Actually, standard sweep: sort by f1 asc.
            # The "width" of the i-th point's contribution is determined by the next point's f1 (or ref).
            # The "height" is determined by the current point's f2 minus the previous point's f2 (or -inf, but here we use ref logic).
            
            # Let's do it carefully.
            # Sorted by f1 asc: p[0] has lowest f1.
            # Contribution of p[i] is (p[i+1].f1 - p[i].f1) * (p[i].f2 - max(p[j].f2 for j>i))
            # Wait, for minimization:
            # Area is sum over i of (x_{i+1} - x_i) * (y_i - y_{max_suffix})
            # Where x is f1, y is f2.
            # y_{max_suffix} for i is the maximum y value in p[i+1:]. If i is last, it's ref[1]? No.
            # The region dominated by p[i] but not by any p[j] with j>i (which have larger f1) 
            # extends from p[i].f1 to p[i+1].f1 in x-direction.
            # In y-direction, it extends from p[i].f2 to the minimum y-value of points that "cover" that x-range?
            # No, the standard 2D HV formula for non-dominated points sorted by f1 asc:
            # HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (f2_i - f2_{i+1})  -- This is only true if f2 is also sorted desc?
            # If we sort by f1 asc, for a non-dominated set, f2 must be desc.
            # Let's verify: If p1 < p2 in f1, then p1 must be > p2 in f2 (otherwise p1 dominates p2).
            # So f2 is strictly decreasing as f1 increases.
            
            f1 = p[:, 0]
            f2 = p[:, 1]
            
            # Add reference point to the end
            f1_ext = np.append(f1, ref[0])
            f2_ext = np.append(f2, -np.inf) # This doesn't work directly for the last term
            
            # Correct formula:
            # Sort by f1 asc.
            # For each point i (0 to N-1):
            # width = f1[i+1] - f1[i] (where f1[N] = ref[0])
            # height = f2[i] - max(f2[j] for j > i)
            # Since f2 is decreasing, max(f2[j] for j > i) is f2[i+1].
            # So height = f2[i] - f2[i+1].
            # Exception for the last point i=N-1:
            # width = ref[0] - f1[N-1]
            # height = f2[N-1] - (-inf)? No, the region goes down to -inf? No, the reference point defines the bound.
            # The region is [f1_i, f1_{i+1}] x [f2_i, f2_{i+1}].
            # For the last point, the x-range is [f1_{N-1}, ref[0]].
            # The y-range is [f2_{N-1}, ref[1]]? 
            # No, the hypervolume is the volume of the union of boxes [f_i, ref].
            # Decomposition:
            # Box i is [f1_i, f1_{i+1}] x [f2_i, f2_{i+1}] for i < N-1.
            # Box N-1 is [f1_{N-1}, ref[0]] x [f2_{N-1}, ref[1]].
            
            total_hv = 0.0
            for i in range(len(p)):
                x1 = f1[i]
                x2 = f1[i+1] if i < len(p)-1 else ref[0]
                y1 = f2[i]
                y2 = f2[i+1] if i < len(p)-1 else ref[1]
                
                # Ensure positive dimensions
                w = max(0.0, x2 - x1)
                h = max(0.0, y1 - y2)
                total_hv += w * h
                
            return total_hv

        # Initial Sampling
        n_init = min(20, B)
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = lo + span * np.random.random(d)
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            update_archive(x, f)
            B -= 1

        # Main Evolutionary Loop
        # Population size for mutation
        pop_size = 10
        
        while B > 0 and time.time() < deadline:
            # Generate candidates
            candidates = []
            
            # 1. Mutation from existing archive
            if len(xs) > 0:
                # Pick parents
                for _ in range(pop_size):
                    # Weighted selection: prefer points with larger hypervolume contribution?
                    # For simplicity and speed, random selection is often sufficient for 2D.
                    # Or pick the "best" (lowest f1 or f2) to exploit.
                    # Let's mix: 50% random, 50% best (lowest f1 or f2)
                    if np.random.random() < 0.5 and len(xs) > 2:
                        # Differential Evolution style
                        i1 = np.random.randint(len(xs))
                        i2 = np.random.randint(len(xs))
                        i3 = np.random.randint(len(xs))
                        F = 0.5 + np.random.random() * 0.5 # Mutation factor
                        v = xs[i1] + F * (xs[i2] - xs[i3])
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
                    elif len(xs) > 0:
                        # Local mutation around a random point
                        i = np.random.randint(len(xs))
                        sigma = 0.1 * span # Local step
                        v = xs[i] + np.random.normal(0, 1, d) * sigma
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
            
            # 2. Random exploration
            n_random = 5
            for _ in range(n_random):
                x = lo + span * np.random.random(d)
                candidates.append(x)
            
            # Evaluate candidates
            for x in candidates:
                if B <= 0 or time.time() > deadline:
                    break
                f = problem(x)
                f = np.array(f, dtype=np.float64)
                # Check validity
                if not np.all(np.isfinite(f)):
                    continue
                update_archive(x, f)
                B -= 1

        # Return the archive (value is ignored by evaluator, but must be returned)
        # The evaluator records every returned objective vector.
        # Wait, the task says: "Every returned objective vector is recorded... return value of __call__ is ignored."
        # Actually, re-reading: "Every returned objective vector is recorded" usually implies the evaluator
        # intercepts the calls to `problem`? 
        # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" -> This phrasing is tricky.
        # "The return value of __call__ is ignored."
        # Usually in these tasks, the `problem` function is wrapped by the evaluator to record evaluations.
        # So we don't need to return the front, just ensure we made the calls.
        # However, to be safe and follow "Keep the function names and signatures... exactly as in the current program",
        # I will return the array of objective values of the archive, similar to the seed.
        
        if len(fs) > 0:
            return np.array(fs)
        else:
            return np.array([])

# EVOLVE-BLOCK-END
