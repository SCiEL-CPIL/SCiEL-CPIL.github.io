import math
import random
import time

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _check_pairwise_distances_rational(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx * dx + D * dy * dy
            if not _is_square(dist_sq):
                return False
    return True

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # (x2-x1)(y3-y1) == (y2-y1)(x3-x1)
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    return False
    return True

def _check_no_four_concyclic(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    # Determinant of the matrix:
                    # x^2 + D*y^2, x, y, 1
                    # Check if det == 0
                    # Using Cramer's rule or direct expansion
                    # For D=1, standard circle equation.
                    # Generalized circle: x^2 + D*y^2 + Ax + By + C = 0
                    # Determinant of:
                    # [x1^2 + D*y1^2, x1, y1, 1]
                    # [x2^2 + D*y2^2, x2, y2, 1]
                    # [x3^2 + D*y3^2, x3, y3, 1]
                    # [x4^2 + D*y4^2, x4, y4, 1]
                    a1 = x1*x1 + D*y1*y1
                    a2 = x2*x2 + D*y2*y2
                    a3 = x3*x3 + D*y3*y3
                    a4 = x4*x4 + D*y4*y4
                    
                    # 4x4 determinant
                    # | a1 x1 y1 1 |
                    # | a2 x2 y2 1 |
                    # | a3 x3 y3 1 |
                    # | a4 x4 y4 1 |
                    # Expand along last column:
                    # 1 * (x2*y3 - x3*y2)*a1 - 1 * (x2*y3 - x3*y2)*a2 ... no
                    # Minor 1: | x2 y2 |
                    #          | x3 y3 | -> x2*y3 - x3*y2
                    # Minor 2: | a2 y2 |
                    #          | a3 y3 | -> a2*y3 - a3*y2
                    # Minor 3: | a2 x2 |
                    #          | a3 x3 | -> a2*x3 - a3*x2
                    # Minor 4: | a2 x2 |
                    #          | a4 x4 | ... wait, rows are i,j,k,l
                    # Det = a1*(x2*y3 - x3*y2) - x1*(a2*y3 - a3*y2) + y1*(a2*x3 - a3*x2) - 1*(a2*x3*y4 - a2*x4*y3 - a3*x2*y4 + a3*x4*y2 + a4*x2*y3 - a4*x3*y2)
                    # This is getting messy. Let's use a simpler check.
                    # Ptolemy's theorem? No, that's for cyclic quads.
                    # If 4 points are concyclic, the determinant is 0.
                    # Let's compute it.
                    det = 0
                    # Using rule of Sarrus for 4x4? No.
                    # Let's just use numpy if available, or manual expansion.
                    # Manual expansion of 4x4 det:
                    # | a1 x1 y1 1 |
                    # | a2 x2 y2 1 |
                    # | a3 x3 y3 1 |
                    # | a4 x4 y4 1 |
                    # = a1(x2(y3*1 - 1*y4) - y2(x3*1 - 1*x4)) - x1(a2(y3*1 - 1*y4) - y2(a3*1 - 1*a4)) + y1(a2(x3*1 - 1*x4) - x2(a3*1 - 1*a4)) - 1(a2(x3*y4 - y3*x4) - y2(x3*a4 - a3*x4))
                    # Wait, the last term expansion is for column 4.
                    # Let's do it carefully.
                    # Term 1: a1 * (x2*y3 - x3*y2 - x2*y4 + x4*y2) ... No.
                    # Let's use the property that if 4 points are concyclic, the cross ratio is real? No.
                    # Let's just compute the determinant using a helper function.
                    
                    def det4(m):
                        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2] - m[2][2]*m[3][1])) -
                                m[0][1]*(m[1][0]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2] - m[2][2]*m[3][0])) +
                                m[0][2]*(m[1][0]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1] - m[2][1]*m[3][0])) -
                                m[0][3]*(m[1][0]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1] - m[2][1]*m[3][0])))
                    
                    m = [[a1, x1, y1, 1], [a2, x2, y2, 1], [a3, x3, y3, 1], [a4, x4, y4, 1]]
                    if det4(m) == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    D = 1
    # Seed 5 points
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify seed
    if not (_check_pairwise_distances_rational(points, D) and _check_no_three_collinear(points) and _check_no_four_concyclic(points, D)):
        # Fallback to even simpler 3 points if seed is bad (shouldn't happen)
        points = [(0,0), (3,0), (0,4)]
        
    best_points = points
    
    # Try to add more points
    # Random search for new points
    # We'll try to add points one by one.
    # To speed up, we can restrict the search area.
    
    attempts = 0
    while time.time() < deadline and len(points) < 8:
        attempts += 1
        # Random point in a range
        # The seed points are around x in [-56, 16], y in [0, 63]
        # Let's search in a larger box
        x = random.randint(-200, 200)
        y = random.randint(-200, 200)
        
        # Skip if too close to existing points
        # Check if candidate can be added
        new_points = points + [(x, y)]
        
        if not _check_pairwise_distances_rational(new_points, D):
            continue
        if not _check_no_three_collinear(new_points):
            continue
        if not _check_no_four_concyclic(new_points, D):
            continue
            
        # If we get here, the point is valid to add
        points = new_points
        best_points = points
        # Reset attempts or continue?
        # Continue to try to add more.
        
    # Return the best points found
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
