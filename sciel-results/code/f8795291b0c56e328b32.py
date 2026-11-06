"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import math

# EVOLVE-BLOCK-START
def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def check_collinear(p1, p2, p3):
    # (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1)
    return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])

def check_concyclic(p1, p2, p3, p4):
    # Determinant of the circle equation matrix
    # x^2 + y^2 + Dx + Ey + F = 0
    # For points (x,y), we solve for D,E,F.
    # Alternatively, use the determinant:
    # | x1^2+y1^2, x1, y1, 1 |
    # | x2^2+y2^2, x2, y2, 1 | = 0
    # | x3^2+y3^2, x3, y3, 1 |
    # | x4^2+y4^2, x4, y4, 1 |
    
    def det4(m):
        # m is 4x4
        return (
            m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
            - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
            + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
            - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
        )

    rows = []
    for p in [p1, p2, p3, p4]:
        rows.append([p[0]**2 + p[1]**2, p[0], p[1], 1])
    return det4(rows) == 0

def validate_set(points):
    n = len(points)
    if n < 3:
        return False
    
    # Check distances
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx*dx + dy*dy
            if not is_square(d2):
                return False
            
    # Check collinearity
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if check_collinear(points[i], points[j], points[k]):
                    return False
                    
    # Check concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if check_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def find_point_for_set(existing_points, search_range=100):
    """
    Finds a new point (x,y) such that distances to all existing points are integers,
    and the new set remains valid (no 3 collinear, no 4 concyclic).
    """
    # Pre-calculate squares up to search_range^2 * 2
    max_d2 = (2*search_range)**2
    squares = {i*i: i for i in range(math.isqrt(max_d2) + 1)}
    
    # We search for x, y in [-search_range, search_range]
    # To optimize, we can iterate x, then y.
    # For each candidate (x,y), check if d^2 to all existing points are in squares.
    
    for x in range(-search_range, search_range + 1):
        for y in range(-search_range, search_range + 1):
            # Skip if already in set
            if (x,y) in existing_points:
                continue
            
            # Check distances
            valid_dist = True
            for px, py in existing_points:
                d2 = (x-px)**2 + (y-py)**2
                if d2 not in squares:
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
                
            # Check geometric constraints with new point
            new_set = list(existing_points) + [(x,y)]
            
            # Check 3-point collinearity involving new point
            collinear = False
            for i in range(len(existing_points)):
                for j in range(i+1, len(existing_points)):
                    if check_collinear(existing_points[i], existing_points[j], (x,y)):
                        collinear = True
                        break
                if collinear:
                    break
            
            if collinear:
                continue
                
            # Check 4-point concyclic involving new point
            concyclic = False
            n = len(existing_points)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if check_concyclic(existing_points[i], existing_points[j], existing_points[k], (x,y)):
                            concyclic = True
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
            
            if concyclic:
                continue
                
            return (x,y)
            
    return None

def construct():
    # Start with the known 5-point set from the seed
    # This set is verified to be valid.
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to extend to 6 points
    current_points = list(base_points)
    
    # Search for a 6th point
    p6 = find_point_for_set(current_points, search_range=100)
    if p6:
        current_points.append(p6)
        
        # Try to extend to 7 points
        p7 = find_point_for_set(current_points, search_range=100)
        if p7:
            current_points.append(p7)
            
            # Try to extend to 8 points
            p8 = find_point_for_set(current_points, search_range=100)
            if p8:
                current_points.append(p8)

    # Format the output
    # If we only have the base 5, we return them.
    # If we found more, we return those.
    
    # Ensure we have at least 3 points (we have 5)
    if len(current_points) < 3:
         # Fallback to a simple valid triangle if something went wrong
         current_points = [(0,0), (1,0), (0,1)] # Check if valid: 1,1,sqrt(2). No, sqrt(2) not rational.
         # (0,0), (3,0), (0,4) -> 3,4,5. Valid.
         current_points = [(0,0), (3,0), (0,4)]
         
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in current_points]}
# EVOLVE-BLOCK-END
