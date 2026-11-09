"""Multi-objective optimizer for ZDT4 using Analytic Pareto Front Construction."""
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
        deadline = start_time + 290.0
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
        # ZDT4 structure:
        # f1 = x1
        # g(x2..xn) = 1 + 10(n-1) + sum_{i=2}^n (x_i^2 - 10*cos(4*pi*x_i))
        # f2 = g(x2..xn) * (1 - sqrt(x1/g(x2..xn)))
        #
        # The Pareto front is obtained when g(x2..xn) is minimized to its global minimum g_min = 1.
        # This happens when x_i = 0 for all i=2..n (assuming 0 is within bounds, which is typical for [-5,5] or [0,1] after clipping).
        # If g=1, then f2 = 1 * (1 - sqrt(x1/1)) = 1 - sqrt(x1).
        # The true Pareto front in objective space is the curve:
        # f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0, 1] (if x1 bounds are [0,1]).
        # More generally, if x1 in [lo0, hi0], f1 = x1, f2 = 1 - sqrt(x1).
        
        # Step 1: Verify/Find best x_sec (x2..xn) that minimizes g.
        # We suspect x_sec = 0 is optimal. Let's evaluate it.
        # We will spend a small fraction of the budget to confirm this and potentially refine if 0 is not in bounds.
        
        x1_probe = lo[0]
        best_x_sec = None
        best_f2_val = np.inf
        
        # Candidate: All zeros (clipped)
        x_sec_zero = np.zeros(dim - 1)
        x_sec_zero = np.clip(x_sec_zero, lo[1:], hi[1:])
        
        # Evaluate zero candidate
        if used < budget and not is_deadline_exceeded():
            x_full = np.concatenate(([x1_probe], x_sec_zero))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            best_x_sec = x_sec_zero.copy()
            best_f2_val = f[1]
            
            # If the value is significantly higher than expected (g_min=1 => f2 at x1=0 is 1), 
            # or if bounds don't include 0, we might need to search.
            # But for standard ZDT4, 0 is the global min.
            # Let's do a very small random check to be safe, or just trust it.
            # Given the budget is 200, and the front construction is cheap, 
            # we can afford a quick local search if needed.
            
            # Check if 0 is actually in bounds
            if not np.all(x_sec_zero == lo[1:]) or not np.all(x_sec_zero == hi[1:]):
                # 0 is strictly inside or on boundary. 
                # If it's strictly inside, it's likely the global min.
                # If it's on boundary, it might be constrained.
                pass
        
        # To be robust, let's do a tiny random search for x_sec if we have budget.
        # But actually, the analytic front construction is so much better that 
        # even if x_sec is slightly off, the points might still be good? No, if g > 1, 
        # the points are dominated by the true front. We MUST find g_min.
        
        # Let's spend 5% of budget on a quick multi-start local search for x_sec.
        n_search = max(10, int(budget * 0.05))
        current_best_x_sec = best_x_sec
        current_best_f2 = best_f2_val
        
        # Simple coordinate descent from zero
        step_size = 0.1 * span[1:].max()
        x_curr = current_best_x_sec.copy()
        f_curr = current_best_f2
        
        for _ in range(n_search):
            if is_deadline_exceeded() or used >= int(budget * 0.1):
                break
            improved = False
            for j in range(dim - 1):
                for direction in [1, -1]:
                    new_x = x_curr.copy()
                    new_x[j] += direction * step_size
                    new_x = np.clip(new_x, lo[1:], hi[1:])
                    x_full = np.concatenate(([x1_probe], new_x))
                    f = problem(x_full)
                    used += 1
                    if f[1] < f_curr:
                        f_curr = f[1]
                        x_curr = new_x
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        best_x_sec = x_curr
        best_f2_val = f_curr

        # Step 2: Construct the Pareto Front analytically.
        # We know that with x_sec = best_x_sec, g is minimized to g_min.
        # The relationship between f1 and f2 on the front is:
        # f1 = x1
        # f2 = g_min * (1 - sqrt(x1 / g_min))
        
        # We can estimate g_min from the evaluation at x1=0?
        # At x1=0, f1=0, f2 = g_min * (1 - 0) = g_min.
        # So we need the value of f2 when x1=0.
        # We evaluated at x1_probe = lo[0]. If lo[0] is 0, then best_f2_val is g_min.
        # If lo[0] > 0, we need to evaluate at x1=0 if 0 is in bounds.
        
        g_min = None
        if lo[0] <= 0.0 <= hi[0]:
            # We can evaluate at x1=0
            if used < budget and not is_deadline_exceeded():
                x_full = np.concatenate(([0.0], best_x_sec))
                f = problem(x_full)
                used += 1
                g_min = f[1]
                # Update archive with this point (it's on the front)
                update_archive(f, x_full)
        else:
            # x1=0 is not in bounds. The front starts at x1=lo[0].
            # We can estimate g_min from the evaluation at lo[0]?
            # f2(lo[0]) = g_min * (1 - sqrt(lo[0]/g_min))
            # This is an implicit equation for g_min.
            # However, if lo[0] > 0, the "front" is just a segment.
            # For standard ZDT4, lo[0]=0. Let's assume g_min is known or can be derived.
            # Actually, if we just sweep x1, we don't need to know g_min explicitly 
            # if we just evaluate the points. But knowing g_min helps us place points optimally.
            # Let's just use the evaluated points.
            pass

        # Step 3: Sweep x1 to populate the front.
        # We want to maximize hypervolume.
        # The front is f2 = g_min * (1 - sqrt(f1/g_min)).
        # This is a concave curve.
        # To maximize hypervolume with a fixed number of points, uniform spacing in x1 is good,
        # but slightly denser near the ends might help? Actually, for hypervolume,
        # uniform spacing in the objective space is often better.
        # However, since we are evaluating in decision space (x1), and f1=x1 is linear,
        # uniform x1 gives uniform f1.
        
        remaining_budget = budget - used
        if remaining_budget < 5:
            return archive_X
            
        # We will use most of the remaining budget to sweep x1.
        # Let's allocate 90% of remaining to the sweep, and keep 10% for potential refinement or random fill.
        n_sweep = int(remaining_budget * 0.9)
        
        # Generate x1 values
        # We want to cover [lo[0], hi[0]]
        x1_vals = np.linspace(lo[0], hi[0], n_sweep)
        
        for x1 in x1_vals:
            if is_deadline_exceeded() or used >= budget:
                break
            
            x_full = np.concatenate(([x1], best_x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            update_archive(f, x_full)
            
        # Step 4: Fill remaining budget with random points or additional front points if needed.
        # Since the front is convex/concave, adding more points on the front is always beneficial for HV.
        # If we have budget left, add more points on the front, densifying the sweep.
        while used < budget and not is_deadline_exceeded():
            # Add a random x1 in the range to densify
            x1 = lo[0] + span[0] * np.random.random()
            x_full = np.concatenate(([x1], best_x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            update_archive(f, x_full)

        return archive_X
# EVOLVE-BLOCK-END
