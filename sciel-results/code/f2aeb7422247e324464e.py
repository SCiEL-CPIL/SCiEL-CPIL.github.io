import math
from fractions import Fraction
from itertools import combinations
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Start with known 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_count = len(best_points)
    
    # Helper functions
    def dist_sq(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    
    def is_perfect_square(n):
        if n < 0:
            return False
        if n == 0:
            return True
        r = math.isqrt(n)
        return r*r == n
    
    def all_rational_dists(points):
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if not is_perfect_square(dist_sq(points[i], points[j])):
                    return False
        return True
    
    def no_three_collinear(points):
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    ax, ay = points[i]
                    bx, by = points[j]
                    cx, cy = points[k]
                    if (bx-ax)*(cy-ay) - (by-ay)*(cx-ax) == 0:
                        return False
        return True
    
    def no_four_concyclic(points):
        # Four points are concyclic iff the determinant is zero
        # Using the general circle equation: x^2+y^2+Dx+Ey+F=0
        # For integer points with D=1, check if all 4 lie on same circle
        for combo in combinations(range(len(points)), 4):
            p0, p1, p2, p3 = [points[i] for i in combo]
            x0, y0 = p0
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            
            # Circle through first 3 points: x^2+y^2+Ax+By+C=0
            # Solve for A, B, C
            # x0^2+y0^2+A*x0+B*y0+C=0
            # x1^2+y1^2+A*x1+B*y1+C=0
            # x2^2+y2^2+A*x2+B*y2+C=0
            
            a0 = x0; b0 = y0; c0 = -(x0*x0+y0*y0)
            a1 = x1; b1 = y1; c1 = -(x1*x1+y1*y1)
            a2 = x2; b2 = y2; c2 = -(x2*x2+y2*y2)
            
            # Solve linear system
            # Use Cramer's rule
            det = a0*(b1*c2 - b2*c1) - b0*(a1*c2 - a2*c1) + c0*(a1*b2 - a2*b1)
            if det == 0:
                # Collinear, skip (already checked)
                continue
            
            det_A = (x0*x0+y0*y0)*(b1*c2 - b2*c1) - y0*(a1*c2 - a2*c1) + c0*(a1*b2 - a2*b1)
            # Actually let me use a different approach
            # A = (det_A)/det where det_A replaces first column
            
            # Let me just check if p3 satisfies the circle
            # Using the determinant form:
            # |x^2+y^2  x  y  1|
            # |x0^2+y0^2 x0 y0 1| = 0 for all 4 points
            # |x1^2+y1^2 x1 y1 1|
            # |x2^2+y2^2 x2 y2 1|
            # |x3^2+y3^2 x3 y3 1|
            
            # This 4x4 determinant must be zero
            # Compute it
            r0 = [x0*x0+y0*y0, x0, y0, 1]
            r1 = [x1*x1+y1*y1, x1, y1, 1]
            r2 = [x2*x2+y2*y2, x2, y2, 1]
            r3 = [x3*x3+y3*y3, x3, y3, 1]
            
            # Compute 4x4 determinant
            def det4(m):
                # m is 4x4 list of lists
                return (
                    m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                    - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                    + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                    - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                )
            
            if det4([r0, r1, r2, r3]) == 0:
                return False
        return True
    
    def is_valid(points):
        if not all_rational_dists(points):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True
    
    # Verify initial set
    assert is_valid(best_points), "Initial set is invalid"
    
    # Try to add points to the 5-point set
    # Strategy: search for a 6th point that works with all 5
    # Then try to add a 7th
    
    # First, let's try known 6-point integral sets from literature
    # The smallest known 6-point integral sets have been found
    
    # Try to search for 6th point near existing points
    # For each existing point, try adding a point at integer distance from all 5
    
    def try_add_point(existing_points, search_radius=200):
        """Try to find a new point that maintains validity"""
        n = len(existing_points)
        # Try points in a region around the centroid
        cx = sum(p[0] for p in existing_points) // n
        cy = sum(p[1] for p in existing_points) // n
        
        # Try points in expanding rings
        for r in range(1, search_radius+1):
            if time.time() > deadline:
                return None
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    new_p = (cx + dx, cy + dy)
                    # Check it's not already in the set
                    if new_p in existing_points:
                        continue
                    candidate = existing_points + [new_p]
                    # Quick check: rational distances
                    if not all_rational_dists(candidate):
                        continue
                    # Check collinearity
                    if not no_three_collinear(candidate):
                        continue
                    # Check concyclicity
                    if not no_four_concyclic(candidate):
                        continue
                    return new_p
        return None
    
    # Try to find a 6th point
    sixth = try_add_point(best_points, search_radius=150)
    if sixth is not None:
        best_points = best_points + [sixth]
        best_count = len(best_points)
        print(f"Found 6th point: {sixth}")
    
    # Try to find a 7th point
    if time.time() < deadline - 30:
        seventh = try_add_point(best_points, search_radius=100)
        if seventh is not None:
            best_points = best_points + [seventh]
            best_count = len(best_points)
            print(f"Found 7th point: {seventh}")
    
    # Also try a different approach: search for 6-point sets from scratch
    # using a smarter method - find points that are at integer distances
    # from multiple existing points
    
    if time.time() < deadline - 20:
        # Try to find a 6-point set by searching more broadly
        # Start with pairs of points at integer distance, then build up
        pass
    
    # Final verification
    if not is_valid(best_points):
        # Fallback to known valid 5-point set
        best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_count = 5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
