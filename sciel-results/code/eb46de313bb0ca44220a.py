# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Strategy:
    1. Greedy construction of the complement set S (points NOT in N).
       We add points to S one by one. A point x can be added to S if for every
       other point y on a line L through x, y still has at least one direction
       where the line through y in that direction contains no points from S 
       other than possibly y.
       
       Equivalently, for N = All \ S to be a Nikodym set:
       For every x in All, there exists a direction d such that the line L(x, d)
       contains no points from S other than possibly x.
       
       We maintain safe_mask[q] as a bitmask of directions d where L(q, d) \ {q} 
       does not intersect S. Initially all 1s.
       When adding p to S:
         For each q != p, and each direction d such that p in L(q, d):
           safe_mask[q].remove(d)
         If any safe_mask[q] becomes empty, we cannot add p.
       
       This is a greedy independent set problem. We randomize the order of points
       to try to maximize |S|.
    """
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin
    
    p = 7
    q_field = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
    direction_lines = []
    
    # Vertical
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions
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
    
    # Precompute affected[p] as a list of (q, d) pairs that would be affected if p is added to S.
    # For a given p, the set of (q, d) pairs affected is:
    # For each d in 0..49:
    #   L = L(p, d)
    #   For each q in L, q != p:
    #     q loses direction d.
    
    # To make the greedy step fast, we can maintain safe_count[q] = number of safe directions for q.
    # Initially safe_count[q] = 50.
    # When adding p:
    #   For each (q, d) in affected[p]:
    #     if safe_count[q] > 0 and d is currently safe for q:
    #       safe_count[q] -= 1
    #       if safe_count[q] == 0:
    #         adding p is invalid (unless q == p? No, q != p).
    
    # We will perform multiple random restarts with different orderings.
    # Also, we can try to remove points from N (greedy removal) as a post-processing step,
    # or just rely on the greedy addition of S.
    
    # The greedy addition of S is equivalent to finding a maximal independent set in a conflict graph,
    # where two points conflict if one blocks the last safe direction of the other.
    # The order matters. Random shuffling helps.
    
    best_S = [False] * n
    best_size = 0
    
    num_restarts = 100  # Increase restarts since each run is fast
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Randomize order
        order = list(range(n))
        random.shuffle(order)
        
        in_S = [False] * n
        safe_mask = [(1 << 50) - 1] * n
        current_size = 0
        
        for idx in order:
            # Check if we can add idx to S
            # We need to verify that no other point q loses its last safe direction.
            # We can simulate the addition and rollback if invalid.
            # Or just check first.
            
            # To check:
            # For each d in 0..49:
            #   li = point_line[idx][d]
            #   line = direction_lines[d][li]
            #   for q in line:
            #     if q != idx:
            #       if (safe_mask[q] >> d) & 1:
            #         # q would lose direction d
            #         if safe_mask[q] == (1 << d):
            #           # q would have no safe directions left
            #           invalid
            #         else:
            #           safe_mask[q] &= ~(1 << d)
            
            # Let's do a trial addition and rollback if invalid
            valid = True
            changes = []
            
            for d in range(50):
                li = point_line[idx][d]
                line = direction_lines[d][li]
                for q in line:
                    if q == idx:
                        continue
                    if (safe_mask[q] >> d) & 1:
                        # q would lose direction d
                        if safe_mask[q] == (1 << d):
                            # q would have no safe directions left
                            valid = False
                            break
                        else:
                            # Record the change for rollback
                            changes.append((q, d))
                            safe_mask[q] &= ~(1 << d)
                if not valid:
                    break
            
            if valid:
                in_S[idx] = True
                current_size += 1
            else:
                # Rollback changes
                for q, d in changes:
                    safe_mask[q] |= (1 << d)
        
        if current_size > best_size:
            best_size = current_size
            best_S = in_S[:]
    
    # Build result: N = All \ S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
