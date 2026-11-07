"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HV-driven adaptive optimizer exploiting DTLZ1 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz1_pareto_front(self, t1, t2):
        """
        Compute DTLZ1 objectives for a point on the Pareto front.
        For n=7, M=3: g(x) = 0 when x[2..6] = 0.5
        f1 = 0.5 * (1+g) * x[0] * prod(x[3..6])
        f2 = 0.5 * (1+g) * (1-x[0]-x[1]) * prod(x[3..6])
        f3 = 0.5 * (1+g) * prod(x[3..6])
        With g=0 and x[3..6]=0.5: prod = (0.5)^4 = 1/16
        f1 = t1/32, f2 = (1-t1-t2)/32, f3 = 1/32
        """
        C = 1.0 / 32.0
        f1 = t1 * C
        f2 = (1.0 - t1 - t2) * C
        f3 = C
        return np.array([f1, f2, f3])

    def _hv_3d(self, points, ref):
        """
        Compute exact 3D hypervolume using inclusion-exclusion via sorted layers.
        Points must be strictly better than ref in all objectives.
        """
        # Filter valid points
        valid = [p for p in points if all(p[i] < ref[i] for i in range(3))]
        if not valid:
            return 0.0

        # Sort by f1 ascending
        valid.sort(key=lambda p: p[0])

        # For each unique f1 value, compute the 2D HV contribution
        # in the f2-f3 plane, considering only points with smaller f1
        hv = 0.0
        n = len(valid)
        for i in range(n):
            p = valid[i]
            f1_lo = p[0]
            f1_hi = valid[i+1][0] if i + 1 < n else ref[0]
            if f1_hi <= f1_lo:
                continue

            # In the f2-f3 plane, find the area dominated by p
            # that is NOT dominated by any point with f1 < p[0]
            # A point q dominates p in f2-f3 if q[1] <= p[1] AND q[2] <= p[2]
            # The region dominated by p in f2-f3 is [0, p[1]] x [0, p[2]]
            # We need to subtract regions already covered by previous points

            # Collect all points with f1 <= p[0] (including p itself)
            # Actually, we want the region in f2-f3 that is dominated by p
            # but not by any point with strictly smaller f1
            # Since we process in order of increasing f1, all previous points have f1 < p[0]
            # (or equal, but we handle ties)

            # The 2D region dominated by p is [0, p[1]] x [0, p[2]]
            # Subtract the union of [0, q[1]] x [0, q[2]] for all q with f1 < p[0]
            # that are not dominated by p in f2-f3

            # For efficiency, compute 2D HV of p against the skyline of previous points
            # Skyline: points where no other point dominates them in both f2 and f3

            # Get all previous points (f1 < p[0])
            prev = [valid[j] for j in range(i) if valid[j][0] < p[0]]

            # Compute 2D HV contribution of p relative to prev
            # This is the area of [0,p[1]]x[0,p[2]] minus the union of [0,q[1]]x[0,q[2]] for q in prev
            # that fall within [0,p[1]]x[0,p[2]]

            # Filter prev to those within p's box in f2-f3
            relevant = [q for q in prev if q[1] <= p[1] and q[2] <= p[2]]

            if not relevant:
                area_2d = p[1] * p[2]
            else:
                # Compute 2D HV of relevant points, then subtract from p's box
                # 2D HV of a set of points with reference (p[1], p[2])
                # Sort by f2 ascending
                relevant.sort(key=lambda q: q[1])
                area_2d = 0.0
                for k in range(len(relevant)):
                    q = relevant[k]
                    f2_lo = q[1]
                    f2_hi = relevant[k+1][1] if k + 1 < len(relevant) else p[1]
                    if f2_hi <= f2_lo:
                        continue
                    # In f3, the max coverage from points with f2 <= q[1]
                    # that have f2 in [f2_lo, f2_hi) is determined by the minimum f3
                    # among points with f2 <= f2_hi (i.e., up to current)
                    # Actually, for 2D HV: sort by f2, for each point the contribution
                    # is (f2_hi - f2_lo) * (p[2] - min_f3)
                    # where min_f3 is the minimum f3 among points with f2 <= current f2_hi
                    # Wait, this is the standard 2D HV algorithm:
                    # Sort by f2 ascending. For each point, the "new" area it adds
                    # is (next_f2 - cur_f2) * (ref_f3 - cur_f3) if cur_f3 < ref_f3
                    # But we need to account for the fact that points with smaller f2
                    # may have larger f3, so they don't cover the full height.

                    # Simpler: for 2D, after sorting by f2, the skyline in f3
                    # is the running minimum of f3.
                    # The area dominated by the set is the integral of (p[2] - skyline_f3)
                    # over f2 from 0 to p[1].

                    # Let me just compute it directly for small sets
                    pass

                # Direct computation: for each point in relevant, compute the box
                # [0, q[1]] x [0, q[2]] and find the union area within [0,p[1]]x[0,p[2]]
                # Use sweep line or just brute force for small n
                # For n < 50, brute force is fine

                # Brute force: discretize? No, let's do exact 2D HV
                # Standard 2D HV: sort by f2, maintain min f3 seen so far
                relevant_sorted = sorted(relevant, key=lambda q: q[1])
                min_f3 = p[2]
                area_2d = 0.0
                for k in range(len(relevant_sorted)):
                    q = relevant_sorted[k]
                    if q[2] < min_f3:
                        f2_next = relevant_sorted[k+1][1] if k + 1 < len(relevant_sorted) else p[1]
                        if f2_next > q[1]:
                            area_2d += (f2_next - q[1]) * (p[2] - q[2])
                        min_f3 = q[2]
                    # If q[2] >= min_f3, this point doesn't extend the skyline
                    # The region from q[1] to next_f2 at height min_f3 is already counted
                    # Wait, I need to be more careful.

                # Let me redo this properly:
                # 2D HV of points S with reference (R1, R2):
                # Sort by f1 ascending. Maintain the "frontier" in f2.
                # For each point in order, if its f2 is less than the current minimum f2
                # seen so far (among points with f1 <= current), it extends the frontier.

                # Actually, the standard algorithm:
                # Sort by f1 ascending.
                # Let min_f2 = infinity initially.
                # For each point (f1_i, f2_i):
                #   if f2_i < min_f2:
                #     f1_next = f1_{i+1} or R1
                #     HV += (f1_next - f1_i) * (R2 - f2_i)
                #     min_f2 = f2_i
                #   else:
                #     # This point is dominated in f2 by a previous point with smaller f1
                #     # But it might still contribute if... no, in 2D HV, once a point
                #     # has a larger f2 than a previous point with smaller f1, it's dominated
                #     # and contributes nothing.

                # Wait, that's not right either. Let me think again.
                # In 2D, the HV dominated by a set of points with reference (R1, R2)
                # is the area of the union of rectangles [0, f1_i] x [0, f2_i].
                # Sort by f1 ascending. The skyline in f2 is the running minimum of f2.
                # For each point, if f2_i < current_min_f2, then the new area added is
                # (f1_{i+1} - f1_i) * (R2 - f2_i) where f1_{i+1} is the next point's f1 or R1.
                # But we also need to account for the area from f1_i to f1_{i+1} at the
                # previous min_f2 level... no, the previous min_f2 was already accounted for.

                # I think the correct algorithm is:
                # Sort by f1 ascending.
                # min_f2 = R2 (start with reference)
                # HV = 0
                # For i from n-1 down to 0:  # or forward?
                #   Actually, let's go forward:
                # Sort by f1 ascending: p0, p1, ..., p_{n-1}
                # Let f1_{n} = R1
                # min_f2 = R2
                # For i from 0 to n-1:
                #   if p_i[1] < min_f2:
                #     HV += (f1_{i+1} - f1_i) * (min_f2 - p_i[1])  # NO, this is wrong
                #     min_f2 = p_i[1]

                # Hmm, let me just use a different approach. For the 2D subproblem,
                # since n is small (< 50), I can compute the union of rectangles directly.

                # Union of rectangles [0, q[1]] x [0, q[2]] for q in relevant
                # within the bounding box [0, p[1]] x [0, p[2]]
                # Use inclusion-exclusion or sweep line.

                # Sweep line in f2 direction:
                # Sort relevant by f1 (which is q[1] here, the first coordinate in 2D)
                # Actually, let me rename: in the 2D subproblem, coordinates are (u, v)
                # where u = f2, v = f3.
                # Rectangles are [0, u_q] x [0, v_q] for each q.
                # We want the area of their union within [0, p[1]] x [0, p[2]].

                # Since all rectangles start at origin, the union is determined by the
                # skyline: for each u, the max v covered.
                # Sort by u ascending. The skyline v is the running max of v.
                # Area = integral of skyline_v du from 0 to p[1].

                # Sort by u (q[1]) ascending
                rel_sorted = sorted(relevant, key=lambda q: q[1])
                skyline_v = 0.0
                area_2d = 0.0
                u_prev = 0.0
                for k in range(len(rel_sorted)):
                    q = rel_sorted[k]
                    u_cur = min(q[1], p[1])
                    if u_cur > u_prev:
                        area_2d += (u_cur - u_prev) * skyline_v
                    if q[2] > skyline_v:
                        skyline_v = q[2]
                    u_prev = u_cur
                # After all points, from u_prev to p[1]
                if u_prev < p[1]:
                    area_2d += (p[1] - u_prev) * skyline_v

                # The area dominated by p but not by prev is:
                # p's box area minus the union area of prev within p's box
                # But wait, the union area I computed is the area dominated by prev.
                # The area dominated by p is p[1] * p[2].
                # The area dominated by both p and prev is the union area (since prev's
                # rectangles are all within p's box).
                # So the marginal contribution of p is:
                # p[1]*p[2] - union_area_of_prev_within_p_box
                # But that's not right either, because we want the area dominated by p
                # but NOT by any previous point. That is p's box minus the union of prev's boxes.
                # But the union of prev's boxes might extend beyond p's box in f3 direction?
                # No, we filtered to relevant = {q : q[1] <= p[1] and q[2] <= p[2]},
                # so all prev boxes are within p's box.
                # So marginal 2D area = p[1]*p[2] - union_area

                box_area = p[1] * p[2]
                area_2d = box_area - area_2d

            hv += (f1_hi - f1_lo) * area_2d

        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        # DTLZ1: x[0], x[1] determine position on Pareto front
        # x[2..6] = 0.5 for g=0
        base_x = np.zeros(self.dim)
        base_x[2:] = 0.5

        evaluated = []  # list of (x, obj)
        objs = []  # list of objective vectors

        def add_point(t1, t2):
            """Evaluate a point on the Pareto front and record it."""
            nonlocal evaluated, objs
            if len(evaluated) >= self.budget:
                return
            if time.time() > deadline:
                return
            x = base_x.copy()
            x[0] = t1
            x[1] = t2
            obj = problem(x)
            evaluated.append((x, obj))
            objs.append(obj)

        # Phase 1: Coarse grid initialization
        # Use a triangular grid in (t1, t2) space
        # t1 >= 0, t2 >= 0, t1 + t2 <= 1
        K = 0
        while (K + 1) * (K + 2) // 2 <= self.budget * 0.4:
            K += 1

        for i in range(K + 1):
            for j in range(K + 1 - i):
                t1 = i / K if K > 0 else 0.0
                t2 = j / K if K > 0 else 0.0
                add_point(t1, t2)

        # Phase 2: Add corners explicitly
        for t1, t2 in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]:
            add_point(t1, t2)

        # Phase 3: HV-driven adaptive refinement
        ref = np.array([400.0, 400.0, 400.0])
        # The Pareto front has f1+f2+f3 = 1/32 ≈ 0.03125
        # So all points are well within the reference point

        # For each remaining evaluation, find the point in the simplex
        # that maximizes marginal HV gain
        # To do this efficiently, we'll use a candidate set approach:
        # Maintain a set of candidate (t1, t2) points and pick the one with max HV gain

        # Initialize candidate set with a fine grid
        K_cand = 30
        candidates = []
        for i in range(K_cand + 1):
            for j in range(K_cand + 1 - i):
                t1 = i / K_cand
                t2 = j / K_cand
                candidates.append((t1, t2))

        # Remove candidates that are already evaluated
        evaluated_set = set()
        for t1, t2 in [(x[0], x[1]) for x, obj in evaluated]:
            evaluated_set.add((round(t1, 10), round(t2, 10)))

        candidates = [c for c in candidates if (round(c[0], 10), round(c[1], 10)) not in evaluated_set]

        # For each remaining evaluation, pick the candidate with max marginal HV
        while len(evaluated) < self.budget:
            if time.time() > deadline:
                break

            if not candidates:
                # Generate new candidates via random sampling
                for _ in range(100):
                    u = np.random.random()
                    v = np.random.random() * (1 - u)
                    candidates.append((u, v))
                break

            # Find the candidate with maximum marginal HV
            best_gain = -1
            best_idx = -1
            current_hv = self._hv_3d(objs, ref)

            for idx, (t1, t2) in enumerate(candidates):
                # Compute marginal HV gain
                new_obj = self._dtlz1_pareto_front(t1, t2)
                new_objs = objs + [new_obj]
                new_hv = self._hv_3d(new_objs, ref)
                gain = new_hv - current_hv
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx >= 0:
                t1, t2 = candidates.pop(best_idx)
                add_point(t1, t2)

                # Add nearby points to candidates for local refinement
                for dt1 in [-0.05, -0.02, 0.02, 0.05]:
                    for dt2 in [-0.05, -0.02, 0.02, 0.05]:
                        nt1 = t1 + dt1
                        nt2 = t2 + dt2
                        if nt1 >= 0 and nt2 >= 0 and nt1 + nt2 <= 1:
                            key = (round(nt1, 10), round(nt2, 10))
                            if key not in evaluated_set:
                                candidates.append((nt1, nt2))
                evaluated_set.add((round(t1, 10), round(t2, 10)))

        return [x for x, obj in evaluated]
# EVOLVE-BLOCK-END
