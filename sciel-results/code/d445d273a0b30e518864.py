import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin: evaluator kills at 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        archive_x = []
        archive_f = []
        evals = 0
        
        def time_ok():
            return time.time() < deadline

        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            af = np.array(archive_f_list)
            leq = np.all(af <= f_new, axis=1)
            lt = np.any(af < f_new, axis=1)
            return np.any(leq & lt)
            
        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def add_to_archive(x, f):
            nonlocal evals
            if is_dominated(f, archive_f):
                return False
            
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
                
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        def compute_hv_2d(f_list, ref_point):
            if not f_list:
                return 0.0
            pts = [f for f in f_list if np.all(f < ref_point)]
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda x: x[0])
            hv = 0.0
            ref = np.array(ref_point)
            # 2D HV:
            # Sort by f1 ascending.
            # The Pareto front is a staircase.
            # For each point i, the width of the rectangle is determined by the next point's f1 (or ref).
            # The height is ref_f2 - f2_i.
            # However, we must ensure we don't double count.
            # Actually, the standard formula for 2D HV with non-dominated points sorted by f1:
            # HV = sum_{i=0}^{N-1} (f1_{i+1} - f1_i) * (ref_f2 - f2_i)
            # where f1_N = ref_f1.
            # This works because the points are non-dominated, so f2 is strictly decreasing as f1 increases.
            # Thus, the "skyline" is well-defined.
            for i in range(len(pts)):
                f1, f2 = pts[i]
                next_f1 = pts[i+1][0] if i < len(pts) - 1 else ref[0]
                width = next_f1 - f1
                height = ref[1] - f2
                if width > 0 and height > 0:
                    hv += width * height
            return hv

        def hv_contribution(idx):
            # Calculate HV of full archive
            full_hv = compute_hv_2d(archive_f, [3000.0, 0.06])
            
            # Calculate HV of archive without idx
            # Since archive is small, we can just filter
            reduced_f = [f for i, f in enumerate(archive_f) if i != idx]
            reduced_hv = compute_hv_2d(reduced_f, [3000.0, 0.06])
            
            return full_hv - reduced_hv

        def crowding_distance(f_list):
            # Calculate crowding distance for each point in f_list
            # Returns a list of distances
            n = len(f_list)
            if n <= 2:
                return [float('inf')] * n
            
            distances = [0.0] * n
            f_arr = np.array(f_list)
            
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(f_arr[:, m])
                # End points get infinite distance
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                
                # Normalize objective range
                f_min = f_arr[sorted_idx[0], m]
                f_max = f_arr[sorted_idx[-1], m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                
                for i in range(1, n - 1):
                    idx = sorted_idx[i]
                    prev_idx = sorted_idx[i-1]
                    next_idx = sorted_idx[i+1]
                    distances[idx] += (f_arr[next_idx, m] - f_arr[prev_idx, m]) / f_range
            
            return distances

        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            p = np.zeros(dim)
            # Use primes for bases
            primes = [2, 3, 5, 7]
            for d in range(dim):
                p[d] = halton_sequence(n, primes[d])
            return p

        # 1. Initialization: Use Halton sequence for better coverage
        init_size = 20
        for i in range(1, init_size + 1):
            if not time_ok():
                break
            p = halton_point(i, self.dim)
            x = lo + range_ * p
            f = problem(x)
            evals += 1
            add_to_archive(x, f)
        
        if not time_ok():
            return archive_x
            
        # 2. Main search: NSGA-II style with hypervolume contribution selection
        # We will use the remaining budget for the main loop.
        # Since we don't have a population, we treat the archive as the population.
        # We generate offspring from the archive.
        
        while evals < self.budget and time_ok():
            if len(archive_x) == 0:
                break
            
            # Selection: Choose parent based on HV contribution or random
            # If archive is small, random is fine. If large, use HV contribution.
            if len(archive_x) <= 5:
                idx = np.random.randint(len(archive_x))
            else:
                # Calculate contributions for a subset to save time? 
                # Or just random? 
                # Let's try a mix: 50% random, 50% HV-based if archive > 10
                if len(archive_x) > 10 and np.random.random() < 0.5:
                    # Find point with max HV contribution
                    # This is O(N^2) per step. N is small.
                    max_contrib = -1
                    best_idx = 0
                    for i in range(len(archive_x)):
                        c = hv_contribution(i)
                        if c > max_contrib:
                            max_contrib = c
                            best_idx = i
                    idx = best_idx
                else:
                    idx = np.random.randint(len(archive_x))
            
            parent = archive_x[idx]
            
            # Mutation: Gaussian
            # Adaptive step size
            # Start with a moderate step
            step = 0.1 * range_
            # We can adapt step based on success?
            # For now, fixed small step for exploration, rely on selection for exploitation.
            # Or use a larger step initially?
            # Let's use a step that decreases slightly over time?
            # Or just fixed 0.1 * range.
            
            child = parent + step * np.random.normal(size=self.dim)
            child = np.clip(child, lo, hi)
            
            f_child = problem(child)
            evals += 1
            
            # Add to archive if non-dominated
            add_to_archive(child, f_child)
            
            # If archive is getting too large, we might need to prune?
            # But budget is only 200, so archive size is limited.
            # However, if we have many non-dominated points, HV calculation might be slow.
            # But N^2 is fine for N<200.
            
            # If we have too many points, we could remove the one with lowest crowding distance?
            # But we want to maximize HV, so we should keep all non-dominated points.
            # The scoring is based on the HV of ALL recorded vectors.
            # So we should keep all non-dominated points.
            
        return archive_x
