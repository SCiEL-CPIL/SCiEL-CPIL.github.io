import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def is_int_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        r = math.isqrt(d2)
        return r*r == d2

    def is_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])

    def is_concyclic(p1, p2, p3, p4):
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
                    m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
                    m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        m = [
            [p1[0]**2 + p1[1]**2, p1[0], p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[0], p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[0], p3[1], 1],
            [p4[0]**2 + p4[1]**2, p4[0], p4[1], 1]
        ]
        def det4(m):
            a,b,c,d = m[0]
            e,f,g,h = m[1]
            i,j,k,l = m[2]
            m_,n,o,p = m[3]
            return (
                a*(f*(k*p-l*o) - g*(j*p-l*n) + h*(j*o-k*n)) -
                b*(e*(k*p-l*o) - g*(i*p-l*m_) + h*(i*o-k*m_)) +
                c*(e*(j*p-l*n) - f*(i*p-l*m_) + h*(i*n-j*m_)) -
                d*(e*(j*o-k*n) - f*(i*o-k*m_) + g*(i*n-j*m_))
            )
        return det4(m) == 0

    def find_extension(points, max_range, deadline, time_budget=30.0):
        n = len(points)
        min_x = min(p[0] for p in points) - max_range
        max_x = max(p[0] for p in points) + max_range
        min_y = min(p[1] for p in points) - max_range
        max_y = max(p[1] for p in points) + max_range

        point_set = set(points)
        start_t = time.time()

        for x in range(min_x, max_x + 1):
            if time.time() > deadline or (time.time() - start_t) > time_budget:
                return None
            for y in range(min_y, max_y + 1):
                p_new = (x, y)
                if p_new in point_set:
                    continue

                # Check distances first (most restrictive)
                valid_dist = True
                for p in points:
                    if not is_int_dist(p_new, p):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue

                # Check collinearity
                valid_col = True
                for i in range(n):
                    for j in range(i+1, n):
                        if is_collinear(points[i], points[j], p_new):
                            valid_col = False
                            break
                    if not valid_col:
                        break
                if not valid_col:
                    continue

                # Check concyclicity
                valid_con = True
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(j+1, n):
                            if is_concyclic(points[i], points[j], points[k], p_new):
                                valid_con = False
                                break
                        if not valid_con:
                            break
                    if not valid_con:
                        break
                if not valid_con:
                    continue

                return p_new
        return None

    def find_extension_multi(points, max_range, deadline, time_budget=30.0):
        """Try multiple ranges and return best (largest range) valid point."""
        best = None
        for r in [100, 200, 300, 500, 800]:
            if time.time() > deadline:
                break
            p = find_extension(points, r, deadline, time_budget)
            if p:
                if best is None or abs(p[0]) + abs(p[1]) > abs(best[0]) + abs(best[1]):
                    best = p
                break  # Found one, use it
        return best

    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = list(seed_points)

    # Try to find 6th point with larger range
    remaining_time = deadline - elapsed()
    if remaining_time > 5:
        p6 = find_extension_multi(best_points, 500, deadline, min(remaining_time * 0.4, 30.0))
        if p6:
            best_points = best_points + [p6]
            # Try to find 7th point
            remaining_time = deadline - elapsed()
            if remaining_time > 5:
                p7 = find_extension_multi(best_points, 500, deadline, min(remaining_time * 0.4, 30.0))
                if p7:
                    best_points = best_points + [p7]
                    # Try to find 8th point
                    remaining_time = deadline - elapsed()
                    if remaining_time > 3:
                        p8 = find_extension_multi(best_points, 400, deadline, min(remaining_time * 0.5, 20.0))
                        if p8:
                            best_points = best_points + [p8]

    # If we only got 5 points, try a different approach with backtracking
    if len(best_points) == 5 and elapsed() < deadline - 10:
        # Try different 6th points by searching in different regions
        tried = set()
        for offset_x in [-100, -50, 0, 50, 100]:
            if time.time() > deadline - 5:
                break
            for offset_y in [-80, -40, 0, 40, 80]:
                if time.time() > deadline - 5:
                    break
                # Search in a smaller region around seed + offset
                min_x = min(p[0] for p in seed_points) - 150
                max_x = max(p[0] for p in seed_points) + 150
                min_y = min(p[1] for p in seed_points) - 150
                max_y = max(p[1] for p in seed_points) + 150

                found = None
                for x in range(min_x, max_x + 1):
                    if time.time() > deadline - 5:
                        break
                    for y in range(min_y, max_y + 1):
                        p_new = (x, y)
                        if p_new in set(seed_points) or p_new in tried:
                            continue
                        valid_dist = True
                        for p in seed_points:
                            if not is_int_dist(p_new, p):
                                valid_dist = False
                                break
                        if not valid_dist:
                            continue
                        valid_col = True
                        for i in range(len(seed_points)):
                            for j in range(i+1, len(seed_points)):
                                if is_collinear(seed_points[i], seed_points[j], p_new):
                                    valid_col = False
                                    break
                            if not valid_col:
                                break
                        if not valid_col:
                            continue
                        valid_con = True
                        for i in range(len(seed_points)):
                            for j in range(i+1, len(seed_points)):
                                for k in range(j+1, len(seed_points)):
                                    if is_concyclic(seed_points[i], seed_points[j], seed_points[k], p_new):
                                        valid_con = False
                                        break
                                    if not valid_con:
                                        break
                                if not valid_con:
                                    break
                            if not valid_con:
                                break
                        if not valid_con:
                            continue
                        found = p_new
                        break
                    if found:
                        break
                if found:
                    tried.add(found)
                    candidate = seed_points + [found]
                    # Try to extend this candidate to 7 points
                    remaining_time = deadline - elapsed()
                    if remaining_time > 3:
                        p7 = find_extension(candidate, 300, deadline, min(remaining_time * 0.3, 15.0))
                        if p7:
                            candidate = candidate + [p7]
                    if len(candidate) > len(best_points):
                        best_points = candidate

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
