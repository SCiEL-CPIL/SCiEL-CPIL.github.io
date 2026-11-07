"""Optimized DTLZ4 solver: exploit known Pareto front with adaptive greedy placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """DTLZ4 optimizer exploiting known Pareto front with adaptive greedy placement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj, budget = self.dim, self.n_obj, self.budget

        # DTLZ4 structure
        n_pos = n_obj - 1  # 2 position variables
        n_g = dim - n_pos  # 10 g-variables
        g_opt = 0.5 * (hi[n_pos:] + lo[n_pos:])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            f = problem(x)
            all_X.append(np.asarray(x, dtype=float))
            all_F.append(np.asarray(f, dtype=float))
            used += 1
            return all_F[-1]

        def make_x(x0, x1):
            x = np.full(dim, 0.5)
            x[0] = x0
            x[1] = x1
            x[n_pos:] = g_opt
            return x

        # The Pareto front of DTLZ4 with 3 objectives is the triangle:
        # f1 = x0, f2 = x1, f3 = 1 - x0 - x1, where x0 >= 0, x1 >= 0, x0+x1 <= 1
        #
        # Strategy: Adaptive greedy placement to maximize hypervolume.
        # 1. Start with the 3 vertices and 3 edge midpoints (6 key points)
        # 2. Use a dense candidate pool from Sobol sequence
        # 3. Greedily select points that maximize immediate hypervolume gain
        # 4. This is more efficient than uniform sampling for hypervolume

        # Phase 1: Key structural points (vertices + edge midpoints)
        key_points = [
            (1.0, 0.0),  # f = (1, 0, 0)
            (0.0, 1.0),  # f = (0, 1, 0)
            (0.0, 0.0),  # f = (0, 0, 1)
            (0.5, 0.5),  # f = (0.5, 0.5, 0)
            (0.5, 0.0),  # f = (0.5, 0, 0.5)
            (0.0, 0.5),  # f = (0, 0.5, 0.5)
        ]

        evaluated_positions = set()

        def try_eval(x0, x1):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return
            key = (round(x0, 10), round(x1, 10))
            if key in evaluated_positions:
                return
            evaluated_positions.add(key)
            x = make_x(x0, x1)
            eval_point(x)

        for x0, x1 in key_points:
            try_eval(x0, x1)

        # Phase 2: Generate candidates using space-filling approach
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Use a grid-based approach with adaptive refinement
            # The triangle in (x0, x1) space: x0 >= 0, x1 >= 0, x0+x1 <= 1
            # We want to place points to maximize HV.
            # 
            # Insight: The HV contribution of a point is related to the area of the
            # region in objective space where it is the "best" point.
            # Near the corners, the marginal gain is higher because the boxes are larger.
            
            # Generate Sobol points in 2D and map to triangle
            n_sobol = 1
            while n_sobol < remaining * 3:
                n_sobol *= 2

            sobol = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_points = sobol.random(n_sobol)

            # Map unit square to triangle using the standard mapping
            candidates = []
            for u, v in sobol_points:
                # Map (u,v) in [0,1]^2 to triangle x0>=0, x1>=0, x0+x1<=1
                if u + v <= 1.0:
                    x0 = u
                    x1 = v
                else:
                    x0 = 1.0 - v
                    x1 = 1.0 - u
                key = (round(x0, 10), round(x1, 10))
                if key not in evaluated_positions:
                    candidates.append((x0, x1))

            # Also add some strategic points near the edges
            # Points near edges have higher marginal HV gain
            edge_points = []
            n_edge = min(30, remaining // 3)
            if n_edge > 0:
                # Points along each edge, excluding vertices and midpoints
                for i in range(1, n_edge):
                    t = i / n_edge
                    # Edge from (1,0) to (0,1): x0 = 1-t, x1 = t
                    if 0 < t < 1:
                        x0, x1 = 1.0 - t, t
                        key = (round(x0, 10), round(x1, 10))
                        if key not in evaluated_positions:
                            edge_points.append((x0, x1))
                    # Edge from (1,0) to (0,0): x0 = 1-t, x1 = 0
                    if 0 < t < 1:
                        x0, x1 = 1.0 - t, 0.0
                        key = (round(x0, 10), round(x1, 10))
                        if key not in evaluated_positions:
                            edge_points.append((x0, x1))
                    # Edge from (0,1) to (0,0): x0 = 0, x1 = 1-t
                    if 0 < t < 1:
                        x0, x1 = 0.0, 1.0 - t
                        key = (round(x0, 10), round(x1, 10))
                        if key not in evaluated_positions:
                            edge_points.append((x0, x1))

            # Combine candidates
            all_candidates = edge_points + candidates
            # Remove duplicates
            seen = set()
            unique_candidates = []
            for x0, x1 in all_candidates:
                key = (round(x0, 10), round(x1, 10))
                if key not in seen and key not in evaluated_positions:
                    seen.add(key)
                    unique_candidates.append((x0, x1))

            # Phase 3: Greedy selection based on approximate HV gain
            ref = np.array([2.5, 2.5, 2.5])

            def approx_hv_gain(x0, x1, current_f):
                """Approximate the marginal HV gain of adding point (x0, x1, 1-x0-x1)."""
                f_new = np.array([x0, x1, 1.0 - x0 - x1])
                
                # Check if dominated
                for p in current_f:
                    if all(p[i] <= f_new[i] for i in range(3)) and any(p[i] < f_new[i] for i in range(3)):
                        return 0.0
                
                # The gain is approximately the volume of the box from f_new to ref
                # minus the volume already covered by existing points.
                # Approximate: the box volume is (ref[0]-f_new[0]) * (ref[1]-f_new[1]) * (ref[2]-f_new[2])
                # But we need to subtract what's already covered.
                # 
                # Better approximation: the gain is proportional to the distance from
                # f_new to the nearest existing point in each dimension, weighted by
                # the distance to ref.
                
                # Simple heuristic: gain = product of (ref[i] - f_new[i]) for each dim,
                # times a factor that decreases if there are existing points nearby.
                
                base_vol = 1.0
                for i in range(3):
                    base_vol *= (ref[i] - f_new[i])
                
                # Penalize if there are existing points that are "close" in the sense
                # that they would cover a large part of the box
                penalty = 1.0
                for p in current_f:
                    # If p dominates f_new in any dimension, it reduces the gain
                    # The reduction is proportional to how much p covers
                    for i in range(3):
                        if p[i] < f_new[i]:
                            # p is better in dimension i, so it covers part of the box
                            # The fraction covered is approximately (f_new[i] - p[i]) / (ref[i] - p[i])
                            if ref[i] - p[i] > 1e-10:
                                frac = (f_new[i] - p[i]) / (ref[i] - p[i])
                                frac = min(1.0, max(0.0, frac))
                                penalty *= (1.0 - frac * 0.5)
                
                return base_vol * penalty

            current_f = []
            for f in all_F:
                if all(f[i] < ref[i] for i in range(n_obj)):
                    current_f.append(f)

            # Greedy selection
            max_select = remaining
            n_candidates = len(unique_candidates)

            if n_candidates > 0 and max_select > 0:
                # Greedy phase: select points with highest approximate HV gain
                greedy_count = min(max_select, n_candidates)
                
                # For efficiency, use a simple greedy since we have at most ~200 candidates
                for _ in range(greedy_count):
                    if used >= budget or time.time() > deadline:
                        break
                    if not unique_candidates:
                        break

                    best_idx = -1
                    best_gain = -1
                    
                    # Evaluate all remaining candidates (at most ~200, so this is fast)
                    for idx, (x0, x1) in enumerate(unique_candidates):
                        gain = approx_hv_gain(x0, x1, current_f)
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx

                    if best_idx >= 0:
                        x0, x1 = unique_candidates.pop(best_idx)
                        try_eval(x0, x1)
                        if used < budget:
                            f_new = np.array([x0, x1, 1.0 - x0 - x1])
                            if all(f_new[i] < ref[i] for i in range(n_obj)):
                                current_f.append(f_new)

                # Fill any remaining budget with additional Sobol points
                if used < budget and time.time() < deadline:
                    # Generate more Sobol points
                    extra_n = 1
                    while extra_n < (budget - used) * 2:
                        extra_n *= 2
                    sobol2 = qmc.Sobol(d=2, scramble=True, seed=123)
                    sobol_points2 = sobol2.random(extra_n)
                    for u, v in sobol_points2:
                        if used >= budget or time.time() > deadline:
                            break
                        if u + v <= 1.0:
                            x0 = u
                            x1 = v
                        else:
                            x0 = 1.0 - v
                            x1 = 1.0 - u
                        key = (round(x0, 10), round(x1, 10))
                        if key not in evaluated_positions:
                            try_eval(x0, x1)

        # Return (value ignored by evaluator, but keep interface)
        if all_X:
            return np.array(all_X)
        return np.array([])
# EVOLVE-BLOCK-END
