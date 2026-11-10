import time
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def check_dist(p1, p2, D=1):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + D*dy*dy
        return is_square(d2)

    def check_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        a1 = x1 - x4
        b1 = y1 - y4
        c1 = (x1*x1 + y1*y1) - (x4*x4 + y4*y4)
        
        a2 = x2 - x4
        b2 = y2 - y4
        c2 = (x2*x2 + y2*y2) - (x4*x4 + y4*y4)
        
        a3 = x3 - x4
        b3 = y3 - y4
        c3 = (x3*x3 + y3*y3) - (x4*x4 + y4*y4)
        
        det = a1 * (b2*c3 - c2*b3) - b1 * (a2*c3 - c2*a3) + c1 * (a2*b3 - b2*a3)
        return det == 0

    def is_valid_set(points, D=1):
        n = len(points)
        if len(set(points)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                if not check_dist(points[i], points[j], D):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        for combo in combinations(range(n), 4):
            if check_concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
                return False
        return True

    # Known valid set of 5
    current_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    if not is_valid_set(current_set, D):
        # Fallback
        current_set = [(0,0), (3,0), (0,4)]
        if not is_valid_set(current_set, D):
            return {'D': 1, 'points': [[[0,1],[0,1]]]}

    best_set = current_set[:]
    best_D = D

    # Search for 6th point using Numpy
    # Range [-500, 500]
    x_min, x_max = -500, 500
    y_min, y_max = -500, 500
    
    # Create grid
    xs = np.arange(x_min, x_max + 1)
    ys = np.arange(y_min, y_max + 1)
    X, Y = np.meshgrid(xs, ys)
    X = X.flatten()
    Y = Y.flatten()
    
    # Points as numpy arrays
    px = np.array([p[0] for p in current_set])
    py = np.array([p[1] for p in current_set])
    
    # Compute squared distances to each point in current_set
    # d2_i = (X - px_i)^2 + D * (Y - py_i)^2
    # We need d2_i to be a perfect square.
    
    # Since D=1, we can check is_square on the array.
    # math.isqrt doesn't work on numpy arrays directly in older versions, 
    # but we can use np.sqrt and check integer-ness, or loop if small.
    # Better: use integer arithmetic.
    
    # Filter candidates
    # For each point in current_set, compute d2
    # Keep indices where all d2 are squares.
    
    candidates = []
    
    # To speed up, we can check one point first, say (0,0)
    # d2_0 = X^2 + Y^2
    # Only keep those where d2_0 is a square.
    
    # This reduces the candidate set significantly.
    # Number of points in radius R with integer distance to origin is ~ R^2 / pi * (1/sqrt(R))? No.
    # Density of integer distance points is low.
    
    # Let's just loop over the flattened arrays, but use numpy for the math.
    # d2_0 = X**2 + Y**2
    # is_sq_0 = np.sqrt(d2_0).astype(int) ** 2 == d2_0
    
    # This might be memory heavy for 1000x1000 = 10^6 points.
    # 10^6 integers is 8MB. OK.
    
    d2_0 = X**2 + Y**2
    sq_0 = np.sqrt(d2_0).astype(np.int64)
    mask_0 = (sq_0**2 == d2_0)
    
    X_cand = X[mask_0]
    Y_cand = Y[mask_0]
    
    # Now check other points
    for i in range(1, len(current_set)):
        if len(X_cand) == 0:
            break
        px_i = px[i]
        py_i = py[i]
        d2_i = (X_cand - px_i)**2 + (Y_cand - py_i)**2
        sq_i = np.sqrt(d2_i).astype(np.int64)
        mask_i = (sq_i**2 == d2_i)
        X_cand = X_cand[mask_i]
        Y_cand = Y_cand[mask_i]
        
    # X_cand, Y_cand are the candidates with integer distance to all 5 points.
    # Convert to list of tuples
    cand_list = list(zip(X_cand.tolist(), Y_cand.tolist()))
    
    # Filter out points already in current_set
    cand_set = set(current_set)
    cand_list = [p for p in cand_list if p not in cand_set]
    
    # Now check geometric constraints for each candidate
    for p in cand_list:
        if time.time() > deadline - 5.0:
            break
        # Check collinearity
        valid_geo = True
        for i in range(len(current_set)):
            for j in range(i+1, len(current_set)):
                if check_collinear(current_set[i], current_set[j], p):
                    valid_geo = False
                    break
            if not valid_geo:
                break
        if not valid_geo:
            continue
        
        # Check concyclicity
        for combo in combinations(range(len(current_set)), 3):
            p1, p2, p3 = current_set[combo[0]], current_set[combo[1]], current_set[combo[2]]
            if check_concyclic(p1, p2, p3, p):
                valid_geo = False
                break
        
        if valid_geo:
            best_set = current_set[:] + [p]
            best_D = D
            break
            
    # If no 6th found, try D=2 with a smaller grid?
    # Or just return the 5 points.
    # Given the time, returning 5 is safe. Finding 6 is a win.
    
    points_out = [[[x, 1], [y, 1]] for x, y in best_set]
    
    return {'D': best_D, 'points': points_out}
