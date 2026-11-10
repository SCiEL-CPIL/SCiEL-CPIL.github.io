import random
import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    """
    Finds a set of points in general position with rational distances.
    Uses integer coordinates (D=1) for simplicity and robustness.
    Maximizes the number of points.
    """
    
    # Time limit management
    start_time = time.time()
    TIME_LIMIT = 100  # Leave 20s buffer for evaluation overhead
    
    # Configuration
    MAX_COORD = 3000  # Search in [-MAX_COORD, MAX_COORD]
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def are_concyclic(p1, p2, p3, p4):
        # Check if 4 points lie on a circle.
        # Using the determinant of the matrix:
        # | x^2+y^2  x  y  1 |
        # We compute the circle equation through p1, p2, p3 and check p4.
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # System of linear equations for B, C, D in:
        # x^2 + y^2 + Bx + Cy + D = 0
        # B(x2-x1) + C(y2-y1) = -(x2^2+y2^2 - x1^2-y1^2)
        # B(x3-x1) + C(y3-y1) = -(x3^2+y3^2 - x1^2-y1^2)
        
        a11 = x2 - x1
        a12 = y2 - y1
        b1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
        
        a21 = x3 - x1
        a22 = y3 - y1
        b2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
        
        det = a11 * a22 - a12 * a21
        if det == 0:
            # Collinear, technically not a circle, but handled by collinearity check
            return False 
            
        # Cramer's rule
        B_num = b1 * a22 - a12 * b2
        C_num = a11 * b2 - b1 * a21
        # D = -(x1^2+y1^2 + B*x1 + C*y1)
        # We want to check if x4^2+y4^2 + B*x4 + C*y4 + D == 0
        # Substitute B, C, D:
        # x4^2+y4^2 - (x1^2+y1^2) + B*(x4-x1) + C*(y4-y1) == 0
        # Multiply by det to clear denominators:
        # (x4^2+y4^2 - x1^2-y1^2)*det + B_num*(x4-x1) + C_num*(y4-y1) == 0
        
        val = (x4*x4 + y4*y4 - x1*x1 - y1*y1) * det + B_num * (x4 - x1) + C_num * (y4 - y1)
        return val == 0

    def check_general_position_incremental(current_set, new_point):
        """
        Checks if adding new_point to current_set maintains general position.
        current_set is a list of tuples (x,y).
        """
        # 1. Check collinearity with any pair in current_set
        for i in range(len(current_set)):
            for j in range(i+1, len(current_set)):
                if are_collinear(current_set[i], current_set[j], new_point):
                    return False
        
        # 2. Check concyclicity with any triple in current_set
        # Only needed if len(current_set) >= 3
        if len(current_set) >= 3:
            n = len(current_set)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if are_concyclic(current_set[i], current_set[j], current_set[k], new_point):
                            return False
                            
        return True

    # Generate candidate pool
    # Points must have integer distance to origin (0,0) to make the first step easy
    # x^2 + y^2 = z^2
    candidates = []
    # Pre-compute squares for fast lookup
    max_dist_sq = 2 * MAX_COORD * MAX_COORD
    squares = set()
    for i in range(math.isqrt(max_dist_sq) + 1):
        squares.add(i*i)
        
    for x in range(-MAX_COORD, MAX_COORD + 1):
        for y in range(-MAX_COORD, MAX_COORD + 1):
            if x == 0 and y == 0:
                continue
            if (x*x + y*y) in squares:
                candidates.append((x, y))
                
    # Optimization: Remove duplicates and sort by distance to origin to prefer closer points?
    # Actually, randomizing is better to avoid bias.
    
    best_set = [(0, 0)]
    best_size = 1
    
    # We will run several greedy passes with different random orderings
    num_restarts = 50
    # Adjust number of restarts based on time
    # Each restart iterates through candidates, which is O(N_cand * N_set * (Collinear + Concyclic))
    # N_cand is roughly O(MAX_COORD^2 / log(MAX_COORD)) for Pythagorean points? No, density is lower.
    # Number of Pythagorean points in range R is approx R^2 / log(R) * constant?
    # For R=3000, this might be large. Let's estimate.
    # Actually, we can just iterate.
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        random.seed(12345 + restart)
        shuffled_cands = candidates[:]
        random.shuffle(shuffled_cands)
        
        current_set = [(0, 0)]
        
        # Greedy addition
        for p in shuffled_cands:
            # Check distance to all points in current_set
            valid_dist = True
            for q in current_set:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                if not is_square(dx*dx + dy*dy):
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            # Check general position
            if check_general_position_incremental(current_set, p):
                current_set.append(p)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    
    # Format output
    points_out = []
    for x, y in best_set:
        points_out.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': points_out}

# EVOLVE-BLOCK-END
