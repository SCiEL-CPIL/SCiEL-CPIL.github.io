# EVOLVE-BLOCK-START
import time
import numpy as np

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 620

    # --- Analytical initial construction: 2 large (0.5) + 8 small (0.25) ---
    # Layout:
    #   Large squares: bottom-left (0.25,0.25) and bottom-right (0.75,0.25), side 0.5
    #   4 small squares: top row, centers (0.125,0.875),(0.375,0.875),(0.625,0.875),(0.875,0.875), side 0.25
    #   4 small squares: middle row, centers (0.125,0.625),(0.375,0.625),(0.625,0.625),(0.875,0.625), side 0.25
    centers = np.array([
        [0.25, 0.25], [0.75, 0.25],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
        [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
    ], dtype=np.float64)
    sides = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    def is_feasible(c, s):
        """Check if configuration (centers, sides) is feasible."""
        if np.any(s <= 0):
            return False
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si / 2 < -tol or cx + si / 2 > 1 + tol:
                return False
            if cy - si / 2 < -tol or cy + si / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                if abs(c[i][0] - c[j][0]) < t and abs(c[i][1] - c[j][1]) < t:
                    return False
        return True

    def free_space_1d(i, axis, c, s):
        """
        Compute the interval of centers[i][axis] values such that square i
        does not overlap any other square (given all other positions/sides fixed).
        axis: 0 for x, 1 for y.
        Returns (lo, hi) as the feasible interval for c[i][axis].
        """
        lo = s[i] / 2 - tol
        hi = 1.0 - s[i] / 2 + tol
        for j in range(n):
            if j == i:
                continue
            # For no overlap in axis 'axis', we need:
            #   |c[i][axis] - c[j][axis]| >= (s[i] + s[j]) / 2 - tol
            #   OR
            #   |c[i][other] - c[j][other]| >= (s[i] + s[j]) / 2 - tol
            other = 1 - axis
            t_axis = (s[i] + s[j]) / 2 - tol
            t_other = (s[i] + s[j]) / 2 - tol
            sep_other = abs(c[i][other] - c[j][other])
            if sep_other >= t_other:
                # Already separated in the other axis; no constraint on this axis
                continue
            # Need separation in this axis:
            # c[i][axis] <= c[j][axis] - t_axis  OR  c[i][axis] >= c[j][axis] + t_axis
            left_bound = c[j][axis] - t_axis
            right_bound = c[j][axis] + t_axis
            # The forbidden interval is (left_bound, right_bound)
            # So feasible: [lo, left_bound] U [right_bound, hi]
            # We keep the interval that contains the current position, or the larger one
            cur = c[i][axis]
            if cur <= left_bound:
                hi = min(hi, left_bound)
            elif cur >= right_bound:
                lo = max(lo, right_bound)
            else:
                # Current position is in the forbidden zone; pick the closer boundary
                # This shouldn't happen if we start from feasible, but handle gracefully
                if left_bound - cur < cur - right_bound:
                    hi = min(hi, left_bound)
                else:
                    lo = max(lo, right_bound)
        return lo, hi

    def coordinate_ascent(c, s, max_rounds=200):
        """
        1D coordinate ascent: for each square, optimize its side length
        (and recenter if needed) within its free space.
        """
        c = c.copy()
        s = s.copy()
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                # Try to increase side of square i
                # The max side is limited by:
                # 1. Boundary: s_i <= 2 * min(c[i][0], 1-c[i][0]) and s_i <= 2 * min(c[i][1], 1-c[i][1])
                # 2. For each other square j, if we increase s_i, the forbidden zone grows
                #    We need to find the max s_i such that there exists a position for square i
                #    that doesn't overlap any j.
                # Simpler approach: try candidate side lengths and check feasibility.
                # For efficiency, binary search on side length.
                # Max possible side from boundary alone:
                max_s_boundary = min(2 * c[i][0], 2 * (1 - c[i][0]),
                                     2 * c[i][1], 2 * (1 - c[i][1])) + 2 * tol
                max_s_boundary = min(max_s_boundary, 1.0 + tol)

                # Binary search for max feasible side
                lo_s = s[i]
                hi_s = max_s_boundary
                if hi_s <= lo_s:
                    continue

                # Check if we can grow at all
                best_s = lo_s
                for _ in range(30):  # 30 bisection steps ~ good precision
                    mid_s = (lo_s + hi_s) / 2
                    if mid_s <= best_s + 1e-12:
                        break
                    # Check if there exists a position for square i with side mid_s
                    # that doesn't overlap others
                    # The free space for center i with side mid_s:
                    # For each j, the constraint is:
                    #   |cx_i - cx_j| >= (mid_s + s[j])/2 - tol  OR  |cy_i - cy_j| >= (mid_s + s[j])/2 - tol
                    # We need to check if the intersection of all these constraints
                    # (plus boundary) is non-empty.
                    # This is a 2D feasibility check. For 10 squares, we can do it
                    # by checking the "free rectangle" approach.
                    # Actually, let's just check if the current center works with mid_s,
                    # or if we can shift to make it work.
                    # Simpler: just check if current position is feasible with mid_s.
                    # If not, try shifting.
                    feasible = True
                    new_cx, new_cy = c[i][0], c[i][1]
                    for j in range(n):
                        if j == i:
                            continue
                        t = (mid_s + s[j]) / 2 - tol
                        dx = abs(new_cx - c[j][0])
                        dy = abs(new_cy - c[j][1])
                        if dx < t and dy < t:
                            # Overlap; try to resolve by shifting
                            # Shift in x or y to just clear
                            if dx < dy:
                                # Need more x separation
                                if new_cx < c[j][0]:
                                    new_cx = c[j][0] - t
                                else:
                                    new_cx = c[j][0] + t
                            else:
                                # Need more y separation
                                if new_cy < c[j][1]:
                                    new_cy = c[j][1] - t
                                else:
                                    new_cy = c[j][1] + t
                    # Check boundary
                    if new_cx - mid_s / 2 < -tol or new_cx + mid_s / 2 > 1 + tol:
                        feasible = False
                    if new_cy - mid_s / 2 < -tol or new_cy + mid_s / 2 > 1 + tol:
                        feasible = False
                    # Re-check all overlaps with new position
                    if feasible:
                        for j in range(n):
                            if j == i:
                                continue
                            t = (mid_s + s[j]) / 2 - tol
                            if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                                feasible = False
                                break
                    if feasible:
                        best_s = mid_s
                        c[i] = [new_cx, new_cy]
                        lo_s = mid_s
                    else:
                        hi_s = mid_s

                if best_s > s[i] + 1e-10:
                    s[i] = best_s
                    improved = True

            if not improved:
                break

        return c, s

    # Run coordinate ascent
    c_opt, s_opt = coordinate_ascent(centers, sides, max_rounds=100)

    # Verify feasibility
    if not is_feasible(c_opt, s_opt):
        # Fallback to initial
        c_opt = centers.copy()
        s_opt = sides.copy()

    # Ensure all values are finite and sides > 0
    if not (np.all(np.isfinite(c_opt)) and np.all(np.isfinite(s_opt))):
        c_opt = centers.copy()
        s_opt = sides.copy()
    s_opt = np.maximum(s_opt, 1e-9)

    return c_opt, s_opt


# EVOLVE-BLOCK-END
