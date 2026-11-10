# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a Grid structure.
    
    Strategy:
    Let N be the union of vertical lines x in A and horizontal lines y in B.
    N = { (x, y) : x in A or y in B }.
    We search for sets A, B that minimize |N| while satisfying the Nikodym condition:
    For every point x in F_49^2, there exists a line L through x such that L \ {x} is a subset of N.
    
    For x not in N (x_a not in A and x_b not in B), we need a slope m such that
    for all t != 0, (x_a + t in A) or (x_b + m*t in B).
    """
    start_time = time.time()
    deadline = start_time + 590
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)

    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
    direction_lines = []
    
    # Vertical lines (v = (0, 1))
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions v = (1, m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
            
    # point_line[pt][dir] = line_idx
    point_line = [[0] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx

    def is_valid_Nikodym_S(S_set):
        """Check if complement of S_set is a valid Nikodym set."""
        for x_idx in range(n):
            has_safe_dir = False
            for d in range(50):
                li = point_line[x_idx][d]
                line = direction_lines[d][li]
                # Check if any point in line (other than x_idx) is in S
                blocked = False
                for q in line:
                    if q != x_idx and S_set[q]:
                        blocked = True
                        break
                if not blocked:
                    has_safe_dir = True
                    break
            if not has_safe_dir:
                return False
        return True

    # Try Grid structures
    # N = { (x,y) : x in A or y in B }
    # S = { (x,y) : x not in A and y not in B }
    
    best_S = [False] * n
    best_size = 0
    
    # Iterate over possible sizes for A and B
    # |N| = a*p + b*p - a*b. We want to minimize this.
    # Constraint for validity: a + b >= p - 1 is a heuristic from analysis.
    
    for a in range(1, p + 1):
        for b in range(0, p + 1):
            if time.time() > deadline - 10:
                break
                
            # Estimate size
            size_N = a * p + b * p - a * b
            size_S = n - size_N
            
            if size_S <= best_size:
                continue
                
            # Try specific sets A and B of size a and b
            # For efficiency, we can just try A={0..a-1}, B={0..b-1}
            # Due to symmetry, this is a good representative.
            
            A = set(range(a))
            B = set(range(b))
            
            S_candidate = [False] * n
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                if xa not in A and ya not in B:
                    S_candidate[i] = True
            
            # Verify
            if is_valid_Nikodym_S(S_candidate):
                best_S = S_candidate
                best_size = size_S

    # If no Grid worked, fallback to the previous QR-based method or empty S
    if best_size == 0:
        # Fallback to QR method from original code (simplified)
        QR = {0, 1, 2, 4}
        S_fallback = [False] * n
        for i, (xa, xb, ya, yb) in enumerate(all_points):
            if ya in QR:
                S_fallback[i] = True
        if is_valid_Nikodym_S(S_fallback):
            best_S = S_fallback
            best_size = sum(best_S)
        else:
            # Last resort: Empty S (N = All)
            best_S = [False] * n
            best_size = 0

    # Build result N = complement of S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    if not pts:
        pts = all_points
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
