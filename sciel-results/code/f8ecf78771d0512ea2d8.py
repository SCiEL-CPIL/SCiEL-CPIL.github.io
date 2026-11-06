import time
import math
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit

    # Helper to check if a set of points is valid
    # Points are tuples of Fractions: (x, y)
    def is_valid(points):
        n = len(points)
        # Check distinct
        if len(set(points)) != n:
            return False
        
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    det = p1[0]*(p2[1]-p3[1]) - p1[1]*(p2[0]-p3[0]) + (p2[0]*p3[1] - p3[0]*p2[1])
                    if det == 0:
                        return False
        
        # Check no four concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1 = points[i]
                        p2 = points[j]
                        p3 = points[k]
                        p4 = points[l]
                        def det4(m):
                            return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1])) -
                                    m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0])) +
                                    m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])) -
                                    m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
                        
                        m = [
                            [p1[0]**2 + p1[1]**2, p1[0], p1[1], Fraction(1)],
                            [p2[0]**2 + p2[1]**2, p2[0], p2[1], Fraction(1)],
                            [p3[0]**2 + p3[1]**2, p3[0], p3[1], Fraction(1)],
                            [p4[0]**2 + p4[1]**2, p4[0], p4[1], Fraction(1)]
                        ]
                        if det4(m) == 0:
                            return False
        return True

    # Strategy: Use a known 5-point set and try to find a 6th point.
    # The seed points are:
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # These are integer points. Distances are integers.
    # We will try to find a 6th integer point (x,y) such that:
    # 1. Distances to all 5 are integers.
    # 2. No three collinear, no four concyclic.
    
    seed_pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Pre-compute squared distances from origin for quick check? No, just check all 5.
    
    best_points = [(Fraction(x), Fraction(y)) for x,y in seed_pts]
    
    # Verify seed is valid (should be)
    if not is_valid(best_points):
        # Fallback: try a simpler set? The task says seed is valid.
        pass

    # Search for 6th point
    # We search integer coordinates in a range.
    # If we find one, we try to find a 7th.
    
    found_6 = False
    # Use a larger range if possible, but keep it efficient.
    # 1000x1000 = 1,000,000 checks. Each check is 5 distance checks + 1 validity check (which is expensive).
    # The distance check is cheap (integer arithmetic). The validity check is only run if distances are integers.
    # So most of the time is spent in the distance check loop.
    
    # Let's estimate: 1M points * 5 dist checks = 5M operations. Very fast.
    # The bottleneck is is_valid() which is O(N^4) for concyclic check.
    # For N=6, it's C(6,3)=20 collinear checks + C(6,4)=15 concyclic checks.
    # Each concyclic check involves 4x4 determinant of Fractions.
    # This might be slow if we call it for many candidates.
    # But candidates with integer distances to 5 specific points are rare.
    
    # Let's try a range of -1000 to 1000.
    # If we find a 6th point, we can stop searching for 6th and start searching for 7th.
    
    # Optimization: Check distinctness first.
    seed_set = set(seed_pts)
    
    for x in range(-1000, 1001):
        if time.time() - start_time > timeout:
            break
        for y in range(-1000, 1001):
            if time.time() - start_time > timeout:
                break
            new_pt = (x, y)
            if new_pt in seed_set:
                continue
            
            valid_dist = True
            for p in seed_pts:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx*dx + dy*dy
                root = math.isqrt(d2)
                if root*root != d2:
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            # Candidate found. Check general position.
            new_set = [(Fraction(px), Fraction(py)) for px, py in seed_pts] + [(Fraction(x), Fraction(y))]
            if is_valid(new_set):
                best_points = new_set
                found_6 = True
                break
        if found_6:
            break

    # If we found a 6th point, try to find a 7th
    if found_6:
        # Search for 7th point
        # Now we have 6 points.
        # Search range can be smaller or larger?
        # Let's try -1000 to 1000 again.
        found_7 = False
        # Convert best_points back to integers for fast distance check?
        # They are Fractions but they are integers.
        int_pts = [(int(p[0]), int(p[1])) for p in best_points]
        int_set = set(int_pts)
        
        for x in range(-1000, 1001):
            if time.time() - start_time > timeout:
                break
            for y in range(-1000, 1001):
                if time.time() - start_time > timeout:
                    break
                new_pt = (x, y)
                if new_pt in int_set:
                    continue
                
                valid_dist = True
                for p in int_pts:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx*dx + dy*dy
                    root = math.isqrt(d2)
                    if root*root != d2:
                        valid_dist = False
                        break
                
                if not valid_dist:
                    continue
                
                # Candidate found. Check general position.
                new_set = best_points + [(Fraction(x), Fraction(y))]
                if is_valid(new_set):
                    best_points = new_set
                    found_7 = True
                    break
            if found_7:
                break

    # Format output
    D = 1
    points_out = []
    for p in best_points:
        x, y = p
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': points_out}
