"""Rewritten program: search for integral point sets with rational distances."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Known valid 5-point set (from seed)
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    def is_perfect_square(n):
        if n < 0:
            return False
        if n == 0:
            return True
        r = int(math.isqrt(n))
        return r * r == n

    def check_rational_distances(pts):
        """Check all pairwise distances are rational (perfect square of integer since coords are integers)."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d2 = dx * dx + dy * dy
                if not is_perfect_square(d2):
                    return False
        return True

    def check_no_three_collinear(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    if (p2[1] - p1[1]) * (p3[0] - p1[0]) == (p3[1] - p1[1]) * (p2[0] - p1[0]):
                        return False
        return True

    def check_no_four_concyclic(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4

                        # Circle through p1, p2, p3: x^2+y^2+Ax+By+C=0
                        # A(x2-x1)+B(y2-y1) = -(x2^2+y2^2-x1^2-y1^2)
                        # A(x3-x1)+B(y3-y1) = -(x3^2+y3^2-x1^2-y1^2)
                        dx2 = x2 - x1
                        dy2 = y2 - y1
                        rhs2 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
                        dx3 = x3 - x1
                        dy3 = y3 - y1
                        rhs3 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)

                        det = dx2 * dy3 - dx3 * dy2
                        if det == 0:
                            return False  # collinear

                        # A = (rhs2*dy3 - rhs3*dy2) / det
                        # B = (dx2*rhs3 - dx3*rhs2) / det
                        # Check p4: x4^2+y4^2 + A*x4 + B*y4 + C = 0
                        # C = -(x1^2+y1^2) - A*x1 - B*y1
                        # So: x4^2+y4^2 + A*(x4-x1) + B*(y4-y1) - (x1^2+y1^2) = 0
                        # Multiply by det:
                        # det*(x4^2+y4^2 - x1^2-y1^2) + (rhs2*dy3-rhs3*dy2)*(x4-x1) + (dx2*rhs3-dx3*rhs2)*(y4-y1) = 0

                        val = (det * (x4*x4 + y4*y4 - x1*x1 - y1*y1)
                               + (rhs2*dy3 - rhs3*dy2) * (x4 - x1)
                               + (dx2*rhs3 - dx3*rhs2) * (y4 - y1))
                        if val == 0:
                            return False
        return True

    def is_valid(pts):
        n = len(pts)
        # Check distinct
        seen = set()
        for p in pts:
            if p in seen:
                return False
            seen.add(p)
        return (check_rational_distances(pts) and
                check_no_three_collinear(pts) and
                check_no_four_concyclic(pts))

    # Verify base set
    if not is_valid(base_points):
        # Should not happen, but fallback
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in base_points]}

    # Precompute distances from base points to speed up search
    # For a candidate point (x, y), distance to base point (bx, by) is:
    # (x-bx)^2 + (y-by)^2 must be a perfect square

    # Search for 6th point
    # Strategy: iterate over possible (x, y) values and check constraints
    # Use the fact that distances to multiple base points must all be rational

    best_points = list(base_points)

    # Try to find a 6th point by searching in a reasonable range
    # The base points span roughly x in [-56, 16], y in [0, 63]
    # Search a wider range

    # For efficiency, precompute for each base point the set of (dx, dy) that give rational distance
    # Actually, let's just do a direct search with pruning

    # Key insight: if (x,y) has rational distance to (0,0), (0,33), and (16,63),
    # then x^2+y^2, x^2+(y-33)^2, and (x-16)^2+(y-63)^2 are all perfect squares.

    # Let's use a smarter approach: iterate over possible distances from origin
    # If distance from origin is d (integer), then x^2+y^2=d^2
    # We need x, y integers such that:
    # x^2 + y^2 = d1^2
    # (x+56)^2 + (y-33)^2 = d2^2
    # (x+16)^2 + (y-63)^2 = d3^2
    # x^2 + (y-33)^2 = d4^2
    # (x-16)^2 + (y-63)^2 = d5^2

    # This is a system of Diophantine equations. Let's search over d1 (distance from origin).

    # For each d1, find all integer (x,y) with x^2+y^2=d1^2 (Pythagorean-like)
    # Then check the other constraints.

    # Limit d1 to keep computation manageable
    max_d = 500

    found_6 = None
    elapsed_check = time.time()

    for d1 in range(1, max_d + 1):
        if time.time() > deadline:
            break

        # Find all (x, y) with x^2 + y^2 = d1^2
        # x ranges from -d1 to d1
        d1_sq = d1 * d1
        for x in range(-d1, d1 + 1):
            y2 = d1_sq - x * x
            if y2 < 0:
                continue
            y = math.isqrt(y2)
            if y * y != y2:
                continue

            # Check both (x, y) and (x, -y) if y != 0
            candidates = [(x, y)]
            if y != 0:
                candidates.append((x, -y))

            for (cx, cy) in candidates:
                # Skip if already in base points
                if (cx, cy) in base_points:
                    continue

                # Check distance to P2=(-56,33)
                dx2 = cx + 56
                dy2 = cy - 33
                d2_sq = dx2 * dx2 + dy2 * dy2
                if not is_perfect_square(d2_sq):
                    continue

                # Check distance to P3=(-16,63)
                dx3 = cx + 16
                dy3 = cy - 63
                d3_sq = dx3 * dx3 + dy3 * dy3
                if not is_perfect_square(d3_sq):
                    continue

                # Check distance to P4=(0,33)
                dx4 = cx
                dy4 = cy - 33
                d4_sq = dx4 * dx4 + dy4 * dy4
                if not is_perfect_square(d4_sq):
                    continue

                # Check distance to P5=(16,63)
                dx5 = cx - 16
                dy5 = cy - 63
                d5_sq = dx5 * dx5 + dy5 * dy5
                if not is_perfect_square(d5_sq):
                    continue

                # All 5 distances are rational! Now check collinearity and concyclicity
                candidate_set = base_points + [(cx, cy)]
                if check_no_three_collinear(candidate_set) and check_no_four_concyclic(candidate_set):
                    found_6 = (cx, cy)
                    break

            if found_6:
                break

        if found_6:
            break

    if found_6:
        best_points = base_points + [found_6]

    # If we found 6 points, try to find a 7th
    if len(best_points) == 6 and time.time() < deadline - 30:
        # Try to find 7th point
        max_d2 = 300
        found_7 = None

        for d1 in range(1, max_d2 + 1):
            if time.time() > deadline:
                break

            d1_sq = d1 * d1
            for x in range(-d1, d1 + 1):
                y2 = d1_sq - x * x
                if y2 < 0:
                    continue
                y = math.isqrt(y2)
                if y * y != y2:
                    continue

                candidates = [(x, y)]
                if y != 0:
                    candidates.append((x, -y))

                for (cx, cy) in candidates:
                    if (cx, cy) in best_points:
                        continue

                    # Check distances to all 6 points
                    valid = True
                    for (bx, by) in best_points:
                        dx = cx - bx
                        dy = cy - by
                        d2 = dx * dx + dy * dy
                        if not is_perfect_square(d2):
                            valid = False
                            break

                    if not valid:
                        continue

                    candidate_set = best_points + [(cx, cy)]
                    if check_no_three_collinear(candidate_set) and check_no_four_concyclic(candidate_set):
                        found_7 = (cx, cy)
                        break

                if found_7:
                    break

            if found_7:
                break

        if found_7:
            best_points = best_points + [found_7]

    # Return the best set found
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
