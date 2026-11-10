# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    
    Strategy:
    1. Start with all 2401 points in N.
    2. Greedily remove points from N (add to excluded set S).
       A point x can be removed if, for at least one direction v, all other points
       on the line through x with direction v are in N.
       
       Equivalent formulation:
       Let S be the excluded set. For every point x in F_49^2, there must exist a line
       through x such that no other point on that line is in S.
       
       We maintain clear_lines[x] = number of lines through x that contain no points of S \ {x}.
       Initially, clear_lines[x] = 50 for all x.
       
       When we add a point s to S:
       - For each of the 50 lines through s:
         - For each other point x on that line:
           - clear_lines[x] -= 1
           - If clear_lines[x] becomes 0, then x is now covered. This is a conflict.
       
       So, adding s is valid if and only if no x on any line through s has clear_lines[x] == 1.
       
       This is O(50 * 49) per candidate s.
       Total for all 2401 points: 2401 * 50 * 49 ≈ 5.9 million operations.
       
       We try multiple random orders to find a larger S.
    """
    p = 7
    q = p * p
    total_points = q * q
    
    # Precompute field operations
    # F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    # Generate all points
    points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    point_index = {pt: i for i, pt in enumerate(points)}
    
    # Precompute lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, 1, mb)) # (1, m)
    directions.append((0, 1, 0, 1)) # (0, 1)
    
    # For each point, store the list of other points on each line through it
    # lines_through[i] = list of 50 lists, each list contains 49 point indices
    lines_through = []
    for i, pt in enumerate(points):
        pt_lines = []
        for d in directions:
            line_pts = []
            for t in range(q):
                if t == 0:
                    continue
                # x + t*v
                # v = (vx, vy), x = (x, y)
                # t is in F_49
                ta, tb = t // p, t % p
                t_val = (ta, tb)
                
                # t * v
                tvx = mul(t_val, (d[0], d[1]))
                tvy = mul(t_val, (d[2], d[3]))
                
                # x + t*v
                nx = add((pt[0], pt[1]), tvx)
                ny = add((pt[2], pt[3]), tvy)
                
                npt = (nx[0], nx[1], ny[0], ny[1])
                line_pts.append(point_index[npt])
            pt_lines.append(line_pts)
        lines_through.append(pt_lines)
        
    # Greedy search
    # clear_lines[i] = number of lines through point i that are clear of S (excluding i)
    
    def try_greedy(order):
        S = set()
        clear_lines = np.full(total_points, 50, dtype=np.int32)
        
        for s_idx in order:
            if s_idx in S:
                continue
            # Check if adding s_idx causes any conflict
            conflict = False
            for line_pts in lines_through[s_idx]:
                for x_idx in line_pts:
                    if clear_lines[x_idx] == 1:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                S.add(s_idx)
                for line_pts in lines_through[s_idx]:
                    for x_idx in line_pts:
                        clear_lines[x_idx] -= 1
        
        return S

    # Try a few random orders
    best_S = set()
    random.seed(42)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590  # Leave 10 seconds margin
    
    # Try default order first
    order = list(range(total_points))
    S = try_greedy(order)
    if len(S) > len(best_S):
        best_S = S
    
    # Try random orders
    while time.time() < deadline:
        order = list(range(total_points))
        random.shuffle(order)
        S = try_greedy(order)
        if len(S) > len(best_S):
            best_S = S
    
    # Construct N
    N = []
    for i, pt in enumerate(points):
        if i not in best_S:
            N.append([[pt[0], pt[1]], [pt[2], pt[3]]])
    
    return np.array(N, dtype=np.int64)

# EVOLVE-BLOCK-END
