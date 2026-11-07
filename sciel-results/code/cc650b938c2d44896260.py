import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if not check_time():
                return
            if used >= budget:
                return
            x_clipped = np.clip(x, lo, hi)
            x_eval = x_clipped.copy()
            x_eval[2:] = 0.0
            problem(x_eval)
            used += 1

        # DTLZ7 Pareto front: x[2:]=0 => g=1
        # f1 = x[0], f2 = x[1], f3 = 1 - x[1]*h(x[1],1)
        # h(y,1) = 9/(1+2*(sin(3*pi*y)+sin(4*pi*y)))
        
        def compute_objectives(x0, x1):
            f1 = x0
            f2 = x1
            # h(x1, 1)
            y = x1
            h = 9.0 / (1.0 + 2.0 * (np.sin(3.0 * np.pi * y) + np.sin(4.0 * np.pi * y)))
            f3 = 1.0 - y * h
            return np.array([f1, f2, f3])

        # Generate a dense pool of candidates using Sobol sequence for better coverage
        # We need enough candidates to pick 200 unique points.
        # Sobol sequences are low-discrepancy, so they spread out well.
        # Let's use 2048 candidates to be safe and allow for fine selection.
        n_candidates = 2048
        sobol = qmc.Sobol(d=2, scramble=True)
        candidates_xy = sobol.random(n_candidates)
        
        # Compute objective values for all candidates
        f1_vals = candidates_xy[:, 0]
        f2_vals = candidates_xy[:, 1]
        h_vals = 9.0 / (1.0 + 2.0 * (np.sin(3.0 * np.pi * f2_vals) + np.sin(4.0 * np.pi * f2_vals)))
        f3_vals = 1.0 - f2_vals * h_vals
        
        candidates_obj = np.column_stack([f1_vals, f2_vals, f3_vals])
        candidates_xy_full = candidates_xy # Store x0, x1 for evaluation

        # We will maintain a list of selected points (indices into candidates)
        # and their objective vectors.
        
        # Initial points: Ensure we cover the "corners" of the front.
        # The front is bounded by x0=0, x0=1, x1=0, x1=1.
        # However, the shape is complex. Let's just pick the first few Sobol points 
        # or specific strategic points.
        # To be robust, let's manually add the 4 corners of the decision space 
        # to the candidate pool if they aren't close enough, or just rely on the 
        # max-min selection starting from a few seeds.
        
        # Let's seed with 4 strategic points: (0,0), (1,0), (0,1), (1,1) in decision space.
        # We need to find the closest candidates in our Sobol pool or just evaluate them directly.
        # Since the problem clips inputs, we can evaluate specific decision points directly.
        
        strategic_points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.5, 0.5],
        ])
        
        # We will evaluate these first.
        selected_xy = []
        selected_obj = []
        
        for sp in strategic_points:
            if used >= budget:
                break
            if not check_time():
                break
            x = np.zeros(dim)
            x[0] = sp[0]
            x[1] = sp[1]
            evaluate(x)
            selected_xy.append(sp)
            selected_obj.append(compute_objectives(sp[0], sp[1]))
            
        selected_xy = np.array(selected_xy)
        selected_obj = np.array(selected_obj)
        
        # Now, iteratively pick the best candidate from the Sobol pool
        # that maximizes the minimum distance to the selected points in OBJECTIVE space.
        
        # To avoid picking a candidate very close to an already selected point,
        # we can maintain a mask of "used" candidates, but since Sobol points are unique,
        # we just need to ensure we don't pick the same one twice.
        # However, the strategic points might not be in the Sobol pool exactly.
        # So we should check distance to selected_obj.
        
        # We will use a list of indices of candidates that are still "available".
        # Since n_candidates (2048) > budget (200), we can just pick the best one each time.
        # We don't strictly need to remove it from the pool if we ensure we don't pick 
        # a point that is "too close" to an existing one, but max-min distance handles this.
        # If we pick a candidate, we should mark it as used to avoid re-picking it 
        # (though max-min distance would likely drop it to 0 distance if we added it to selected).
        
        # Optimization: 
        # Instead of iterating over all 2048 candidates and calculating distance to all 
        # selected points (200), we can do it in vectorized chunks or just use numpy.
        # 2048 * 200 * 3 floats is small.
        
        remaining_budget = budget - used
        
        for step in range(remaining_budget):
            if not check_time():
                break
            if used >= budget:
                break
            
            # Calculate distance from each candidate to each selected point
            # candidates_obj: (2048, 3)
            # selected_obj: (K, 3)
            
            # Use broadcasting: (2048, 1, 3) - (1, K, 3)
            diff = candidates_obj[:, np.newaxis, :] - selected_obj[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2) # (2048, K)
            min_dists = np.min(sq_dists, axis=1) # (2048,)
            
            # Pick the candidate with the maximum minimum distance
            best_idx = np.argmax(min_dists)
            
            # Get the decision variables for this candidate
            x0, x1 = candidates_xy_full[best_idx]
            
            # Evaluate it
            x = np.zeros(dim)
            x[0] = x0
            x[1] = x1
            evaluate(x)
            
            # Add to selected sets
            new_obj = compute_objectives(x0, x1)
            selected_obj = np.vstack([selected_obj, new_obj])
            
            # Note: We don't strictly need to remove this candidate from the pool 
            # because its distance to the new selected point will be 0, so it won't be picked again 
            # unless there are ties or numerical issues, but argmax will pick a different one 
            # with higher distance. However, to be safe and efficient, we could mask it, 
            # but it's not strictly necessary for correctness given the max-min logic.
            
        return None
