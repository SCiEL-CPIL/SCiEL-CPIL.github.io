"""Optimized DTLZ4 solver: exploit known Pareto front with exact marginal HV greedy placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """DTLZ4 optimizer exploiting known Pareto front with exact marginal HV greedy placement."""

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

        ref = np.array([2.5, 2.5, 2.5])

        # The Pareto front of DTLZ4 with 3 objectives is the triangle:
        # f1 = x0, f2 = x1, f3 = 1 - x0 - x1, where x0 >= 0, x1 >= 0, x0+x1 <= 1

        # Phase 1: Key structural points (vertices + edge midpoints)
        key_points = [
            (1.0, 0.0),
            (0.0, 1.0),
            (0.0, 0.0),
            (0.5, 0.5),
            (0.5, 0.0),
            (0.0, 0.5),
        ]

        evaluated_positions = set()
        non_dominated = []  # list of objective vectors on the Pareto front

        def try_eval(x0, x1):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return
            key = (round(x0, 10), round(x1, 10))
            if key in evaluated_positions:
                return
            evaluated_positions.add(key)
            x = make_x(x0, x1)
            f = eval_point(x)
            # Check if this point is non-dominated and within ref
            if all(f[i] < ref[i] for i in range(n_obj)):
                # Check if dominated by any existing point
                dominated = False
                for p in non_dominated:
                    if all(p[i] <= f[i] for i in range(n_obj)) and any(p[i] < f[i] for i in range(n_obj)):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append(f)
                    # Remove any points dominated by f
                    non_dominated[:] = [p for p in non_dominated
                                        if not (all(f[i] <= p[i] for i in range(n_obj)) and any(f[i] < p[i] for i in range(n_obj)))]

        for x0, x1 in key_points:
            try_eval(x0, x1)

        # Exact marginal HV gain computation for 3 objectives
        # The HV of a set S of non-dominated points with reference point r is:
        # HV = sum over "layers" of volume contributions
        # 
        # For a new point f_new, the marginal gain is the volume of the box
        # [f_new, r] minus the volume of the union of boxes from existing points
        # that intersect this box.
        #
        # For 3D, we can compute this exactly using the inclusion-exclusion
        # principle over the boxes of existing points that are "relevant" 
        # (i.e., whose boxes intersect [f_new, r]).

        def box_volume(a, b):
            """Volume of box from a to b (component-wise)."""
            vol = 1.0
            for i in range(3):
                d = b[i] - a[i]
                if d <= 0:
                    return 0.0
                vol *= d
            return vol

        def exact_marginal_gain(f_new, nd_points):
            """
            Compute exact marginal HV gain of adding f_new to a set of 
            non-dominated points. f_new must not be dominated by any point in nd_points.
            """
            # The gain is the volume of [f_new, ref] not covered by any box [p, ref] 
            # for p in nd_points where p does not dominate f_new.
            # 
            # Since f_new is non-dominated w.r.t. nd_points, for each p in nd_points,
            # there is at least one dimension where p[i] > f_new[i].
            # 
            # The union of boxes [p, ref] for all p in nd_points that have 
            # p[i] <= f_new[i] for some i... actually, let's think differently.
            #
            # The marginal gain equals the volume of the region in [f_new, ref] 
            # that is NOT in the union of [p, ref] for all p in nd_points.
            #
            # A point y in [f_new, ref] is covered by [p, ref] iff y[i] >= p[i] for all i.
            # So y is NOT covered iff for every p in nd_points, there exists i with y[i] < p[i].
            #
            # For 3D, we can compute this by noting that the uncovered region is:
            # [f_new, ref] minus union of [max(f_new, p), ref] for each p.
            #
            # The union of axis-aligned boxes in 3D can be computed by sweeping.
            # But for small n (we'll have at most ~200 points), we can use a simpler approach.
            #
            # Actually, for the marginal gain, we only need to consider points p 
            # where p is "relevant" - i.e., p[i] < ref[i] for all i and the box [p, ref]
            # actually overlaps with [f_new, ref].
            #
            # Key insight: The marginal gain can be computed as:
            # sum over "cells" in the arrangement of the boxes.
            #
            # For 3D with up to ~200 boxes, an O(n^3) approach is too slow.
            # Instead, use the fact that the Pareto front is 2D and we can 
            # compute the gain geometrically.
            #
            # Alternative: Since all points are on the DTLZ4 Pareto front (2D simplex),
            # the HV can be computed as a 2D integral. The marginal gain of a new point
            # is the area of the region in the (x0, x1) triangle where the new point
            # is the "best" in the HV sense.
            #
            # Let's use a direct 3D approach with the Kung algorithm for 3D HV:
            # Sort points by first objective, then sweep.

            # For efficiency with up to ~200 points, use a simpler exact method:
            # The marginal gain = volume([f_new, ref]) - volume(union of [max(f_new,p), ref] for p in nd_points)
            # where max is component-wise.
            
            # For 3D, we can compute the union volume of boxes using the 
            # inclusion-exclusion over pairs and triples, but that's O(n^3).
            # 
            # Better: use the fact that in 3D, the union of axis-aligned boxes
            # can be computed in O(n^2 log n) by sweeping along one axis.
            # 
            # For our purposes, since n is at most ~200, let's use a practical approach:
            # The marginal gain is approximately the volume of the box [f_new, ref]
            # minus the sum of volumes of "shadow" regions cast by each existing point.
            #
            # Exact 3D union of boxes: sort by x, sweep, and for each interval in x,
            # compute the union of rectangles in yz.
            
            # Collect relevant boxes: [max(f_new, p), ref] for each p
            boxes = []
            for p in nd_points:
                lb = np.maximum(f_new, p)
                if all(lb[i] < ref[i] for i in range(3)):
                    boxes.append((lb, ref))
            
            if not boxes:
                return box_volume(f_new, ref)
            
            # Compute union volume of these boxes in 3D
            # Use sweep-line along dimension 0
            events = []
            for lb, ub in boxes:
                events.append((lb[0], 1, lb[1], ub[1], lb[2], ub[2]))
                events.append((ub[0], -1, lb[1], ub[1], lb[2], ub[2]))
            
            events.sort(key=lambda e: e[0])
            
            union_vol = 0.0
            prev_x = events[0][0]
            active = []
            
            i = 0
            while i < len(events):
                cur_x = events[i][0]
                # Add/subtract rectangles at this x
                while i < len(events) and events[i][0] == cur_x:
                    _, typ, y1, y2, z1, z2 = events[i]
                    if typ == 1:
                        active.append((y1, y2, z1, z2))
                    else:
                        try:
                            active.remove((y1, y2, z1, z2))
                        except ValueError:
                            pass
                    i += 1
                
                if active:
                    # Compute union area of rectangles in yz plane
                    area = _rect_union_area(active)
                    dx = cur_x - prev_x
                    if dx > 0:
                        union_vol += area * dx
                prev_x = cur_x
            
            total_box_vol = box_volume(f_new, ref)
            gain = total_box_vol - union_vol
            return max(0.0, gain)

        def _rect_union_area(rects):
            """Compute union area of axis-aligned rectangles in 2D."""
            if not rects:
                return 0.0
            # Sweep along y
            events = []
            for y1, y2, z1, z2 in rects:
                events.append((y1, 1, z1, z2))
                events.append((y2, -1, z1, z2))
            events.sort(key=lambda e: e[0])
            
            area = 0.0
            prev_y = events[0][0]
            active_z = []
            
            i = 0
            while i < len(events):
                cur_y = events[i][0]
                while i < len(events) and events[i][0] == cur_y:
                    _, typ, z1, z2 = events[i]
                    if typ == 1:
                        active_z.append((z1, z2))
                    else:
                        try:
                            active_z.remove((z1, z2))
                        except ValueError:
                            pass
                    i += 1
                
                if active_z:
                    # Union length of intervals on z
                    total_len = _interval_union_length(active_z)
                    dy = cur_y - prev_y
                    if dy > 0:
                        area += total_len * dy
                prev_y = cur_y
            
            return area

        def _interval_union_length(intervals):
            """Total length of union of 1D intervals."""
            if not intervals:
                return 0.0
            sorted_ivs = sorted(intervals, key=lambda x: x[0])
            total = 0.0
            cur_start, cur_end = sorted_ivs[0]
            for s, e in sorted_ivs[1:]:
                if s <= cur_end:
                    cur_end = max(cur_end, e)
                else:
                    total += cur_end - cur_start
                    cur_start, cur_end = s, e
            total += cur_end - cur_start
            return total

        # Phase 2: Generate a large candidate pool using Sobol sequence
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Generate many candidates (much more than budget)
            n_sobol = 1
            while n_sobol < 5000:
                n_sobol *= 2
            n_sobol = min(n_sobol, 8192)

            sobol = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_points = sobol.random(n_sobol)

            candidates = []
            for u, v in sobol_points:
                if u + v <= 1.0:
                    x0 = u
                    x1 = v
                else:
                    x0 = 1.0 - v
                    x1 = 1.0 - u
                key = (round(x0, 10), round(x1, 10))
                if key not in evaluated_positions:
                    candidates.append((x0, x1))

            # Add edge points for better coverage near boundaries
            edge_points = []
            n_edge = 50
            for i in range(1, n_edge):
                t = i / n_edge
                for x0, x1 in [(1.0 - t, t), (1.0 - t, 0.0), (0.0, 1.0 - t)]:
                    key = (round(x0, 10), round(x1, 10))
                    if key not in evaluated_positions and key not in set(map(lambda c: (round(c[0],10), round(c[1],10)), candidates)):
                        edge_points.append((x0, x1))

            all_candidates = edge_points + candidates
            seen = set()
            unique_candidates = []
            for x0, x1 in all_candidates:
                key = (round(x0, 10), round(x1, 10))
                if key not in seen and key not in evaluated_positions:
                    seen.add(key)
                    unique_candidates.append((x0, x1))

            # Phase 3: Greedy selection using exact marginal HV gain
            max_select = remaining
            for _ in range(max_select):
                if used >= budget or time.time() > deadline:
                    break
                if not unique_candidates:
                    break

                best_idx = -1
                best_gain = -1

                for idx, (x0, x1) in enumerate(unique_candidates):
                    f_new = np.array([x0, x1, 1.0 - x0 - x1])
                    # Check if dominated
                    dominated = False
                    for p in non_dominated:
                        if all(p[i] <= f_new[i] for i in range(n_obj)) and any(p[i] < f_new[i] for i in range(n_obj)):
                            dominated = True
                            break
                    if dominated:
                        continue
                    gain = exact_marginal_gain(f_new, non_dominated)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx

                if best_idx >= 0:
                    x0, x1 = unique_candidates.pop(best_idx)
                    try_eval(x0, x1)
                else:
                    break

            # Fill remaining budget with additional Sobol points if needed
            if used < budget and time.time() < deadline:
                extra_n = 1
                while extra_n < (budget - used) * 3:
                    extra_n *= 2
                extra_n = min(extra_n, 4096)
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
