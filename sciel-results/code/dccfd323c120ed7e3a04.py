# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict
from scipy.optimize import linprog

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Strategy:
    1. Greedy construction of the complement set S (points NOT in N) with bitmask tracking.
    2. Local search refinement: try to swap points in S with points not in S to increase |S|
       while maintaining the Nikodym property.
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
    
    # Precompute for each point p, the list of (q, d) pairs that would be affected if p is added to S.
    # affected[p] = list of (q, d) where q loses direction d if p is added to S
    affected = [[] for _ in range(n)]
    for p_idx in range(n):
        for d in range(50):
            li = point_line[p_idx][d]
            line = direction_lines[d][li]
            for q in line:
                if q != p_idx:
                    affected[p_idx].append((q, d))
    
    # Helper function to check if a set S (as a boolean array) is valid
    # Returns True if N = All \ S is a Nikodym set
    def is_valid_S(in_S):
        # For each point x, check if there exists a direction d such that 
        # L(x, d) \ {x} does not intersect S
        for x in range(n):
            has_safe_dir = False
            for d in range(50):
                li = point_line[x][d]
                line = direction_lines[d][li]
                # Check if any point on this line (other than x) is in S
                blocked = False
                for q in line:
                    if q != x and in_S[q]:
                        blocked = True
                        break
                if not blocked:
                    has_safe_dir = True
                    break
            if not has_safe_dir:
                return False
        return True
    
    # Helper function to compute safe_mask for a given in_S
    def compute_safe_mask(in_S):
        safe_mask = [(1 << 50) - 1] * n
        for p_idx in range(n):
            if in_S[p_idx]:
                for q, d in affected[p_idx]:
                    safe_mask[q] &= ~(1 << d)
        return safe_mask
    
    # Check if safe_mask has any zero entries (invalid)
    def is_mask_valid(safe_mask):
        for mask in safe_mask:
            if mask == 0:
                return False
        return True
    
    # Greedy construction of S
    best_S = [False] * n
    best_size = 0
    
    num_restarts = 150
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Randomize order
        order = list(range(n))
        random.shuffle(order)
        
        in_S = [False] * n
        safe_mask = [(1 << 50) - 1] * n
        current_size = 0
        
        for idx in order:
            # Check if we can add idx to S
            valid = True
            changes = []
            
            for q, d in affected[idx]:
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
            
            if valid:
                in_S[idx] = True
                current_size += 1
            else:
                # Rollback changes
                for q, d in changes:
                    safe_mask[q] |= (1 << d)
        
        # Local search: try to improve by swapping
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            
            # Try to add a point not in S
            not_in_S = [i for i in range(n) if not in_S[i]]
            random.shuffle(not_in_S)
            
            for add_idx in not_in_S:
                # Check if we can add add_idx to S
                valid = True
                changes = []
                
                for q, d in affected[add_idx]:
                    if (safe_mask[q] >> d) & 1:
                        if safe_mask[q] == (1 << d):
                            valid = False
                            break
                        else:
                            changes.append((q, d))
                            safe_mask[q] &= ~(1 << d)
                
                if valid:
                    in_S[add_idx] = True
                    current_size += 1
                    improved = True
                    break
            
            # Try to remove a point from S to free up capacity, then add another
            in_S_list = [i for i in range(n) if in_S[i]]
            if in_S_list:
                random.shuffle(in_S_list)
                for rem_idx in in_S_list[:20]:  # Try up to 20 removals
                    # Remove rem_idx from S
                    in_S[rem_idx] = False
                    current_size -= 1
                    
                    # Restore safe_mask for the points that were affected by rem_idx
                    for q, d in affected[rem_idx]:
                        safe_mask[q] |= (1 << d)
                    
                    # Try to add a different point
                    not_in_S = [i for i in range(n) if not in_S[i]]
                    random.shuffle(not_in_S)
                    
                    added = False
                    for add_idx in not_in_S:
                        if add_idx == rem_idx:
                            continue
                        # Check if we can add add_idx to S
                        valid = True
                        changes = []
                        
                        for q, d in affected[add_idx]:
                            if (safe_mask[q] >> d) & 1:
                                if safe_mask[q] == (1 << d):
                                    valid = False
                                    break
                                else:
                                    changes.append((q, d))
                                    safe_mask[q] &= ~(1 << d)
                        
                        if valid:
                            in_S[add_idx] = True
                            current_size += 1
                            added = True
                            improved = True
                            break
                    
                    if not added:
                        # Put rem_idx back
                        in_S[rem_idx] = True
                        current_size += 1
                        for q, d in affected[rem_idx]:
                            safe_mask[q] &= ~(1 << d)
        
        if current_size > best_size:
            best_size = current_size
            best_S = in_S[:]
    
    # Build result: N = All \ S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
