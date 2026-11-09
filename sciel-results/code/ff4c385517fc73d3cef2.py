"""Multi-objective optimizer for ZDT4 using Deterministic Pareto Front Construction."""
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
        deadline = start_time + 290.0  # Safety margin for 300s limit
        budget = self.budget
        dim = self.dim
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        used = 0
        archive_F = []
        archive_X = []

        def is_deadline_exceeded():
            return (time.time() - start_time) > deadline

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def update_archive(f_new, x_new):
            nonlocal archive_F, archive_X
            # Check domination
            for i in range(len(archive_F)):
                if dominates(archive_F[i], f_new):
                    return
            # Remove dominated
            new_F = []
            new_X = []
            for i in range(len(archive_F)):
                if not dominates(f_new, archive_F[i]):
                    new_F.append(archive_F[i])
                    new_X.append(archive_X[i])
            new_F.append(f_new)
            new_X.append(x_new)
            archive_F = new_F
            archive_X = new_X

        # Strategy:
        # ZDT4 structure: f1 = x1, f2 = g(x2..xn) * (1 - sqrt(x1/g))
        # The Pareto front is a curve. To maximize hypervolume, we need:
        # 1. A set of points that are non-dominated.
        # 2. The points should be as close to the true Pareto front as possible (minimize g).
        # 3. The points should be spread out along the front to maximize the area under the curve.
        
        # Key Insight: 
        # If we fix x2..xn to a value that minimizes g (or is very close), then the resulting 
        # points (x1, x2..xn) will form a curve that is close to the true Pareto front.
        # By sampling x1 uniformly (or with a specific distribution) across its range, we 
        # can maximize the hypervolume.
        
        # The current program uses a local search to find a good x2..xn. 
        # However, for ZDT4, the global minimum of g is often at x_i = 0 (if bounds allow) 
        # or at specific local minima. 
        # A simpler and more robust strategy is to:
        # 1. Evaluate a few candidate x2..xn values (e.g., 0, pi, random) to find a good base.
        # 2. Use the remaining budget to sweep x1 densely, fixing x2..xn to the best base found.
        # This ensures that we have a dense set of points along the front, which is crucial 
        # for hypervolume.
        
        # Phase 1: Find a good base for x2..xn
        # We try a few candidates:
        # 1. All zeros
        # 2. All pi (often a local minimum in ZDT4)
        # 3. A random point
        # We evaluate these with x1 = lo[0] to get a sense of g.
        
        n_phase1 = max(10, int(budget * 0.05))
        
        best_x_sec = None
        best_g_val = np.inf
        
        # Candidate 1: Zeros
        x_sec_zero = np.zeros(dim - 1)
        x_sec_zero = np.clip(x_sec_zero, lo[1:], hi[1:])
        
        # Candidate 2: Pi (clipped)
        x_sec_pi = np.full(dim - 1, np.pi)
        x_sec_pi = np.clip(x_sec_pi, lo[1:], hi[1:])
        
        # Candidate 3: Random
        x_sec_rand = lo[1:] + span[1:] * np.random.random(dim - 1)
        
        candidates = [x_sec_zero, x_sec_pi, x_sec_rand]
        
        for x_sec in candidates:
            if used >= n_phase1:
                break
            if is_deadline_exceeded():
                break
            x_full = np.concatenate(([lo[0]], x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            if f[1] < best_g_val:
                best_g_val = f[1]
                best_x_sec = x_sec.copy()
        
        if best_x_sec is None:
            best_x_sec = x_sec_zero.copy()
            x_full = np.concatenate(([lo[0]], best_x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            best_g_val = f[1]

        # Phase 2: Refine x_sec using local search to ensure we are at a local minimum of g.
        # This is crucial because if we are at a local minimum, the resulting front is better.
        # We use a simple coordinate descent on x2..xn to minimize f2 (with x1 fixed).
        max_refine_evals = max(5, int(budget * 0.05))
        current_x_sec = best_x_sec.copy()
        current_f2 = best_g_val
        step_size = 0.1 * span[1:].max()
        
        for _ in range(max_refine_evals):
            if is_deadline_exceeded():
                break
            if used >= n_phase1 + max_refine_evals:
                break
            
            improved = False
            for j in range(dim - 1):
                for direction in [1, -1]:
                    new_x_sec = current_x_sec.copy()
                    new_x_sec[j] += direction * step_size
                    new_x_sec = np.clip(new_x_sec, lo[1:], hi[1:])
                    
                    x_full = np.concatenate(([lo[0]], new_x_sec))
                    x_full = np.clip(x_full, lo, hi)
                    f = problem(x_full)
                    used += 1
                    
                    if f[1] < current_f2:
                        current_f2 = f[1]
                        current_x_sec = new_x_sec
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        best_x_sec = current_x_sec

        # Phase 3: Dense sweep of x1.
        # The Pareto front is defined by x1 in [lo[0], hi[0]] and x2..xn = best_x_sec.
        # We want to maximize hypervolume. A dense, evenly spaced set of points is optimal.
        
        remaining_budget = budget - used
        if remaining_budget < 10:
            n_sweep = max(1, remaining_budget)
        else:
            n_sweep = remaining_budget
            
        # Generate x1 values
        x1_vals = np.linspace(lo[0], hi[0], n_sweep)
        
        for x1 in x1_vals:
            if is_deadline_exceeded():
                break
            if used >= budget:
                break
            
            x_full = np.concatenate(([x1], best_x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            update_archive(f, x_full)
            
        # Phase 4: Fill remaining budget with random points if any left
        while used < budget and not is_deadline_exceeded():
            x_rand = lo + span * np.random.random(dim)
            f = problem(x_rand)
            used += 1
            update_archive(f, x_rand)

        return archive_X
# EVOLVE-BLOCK-END
