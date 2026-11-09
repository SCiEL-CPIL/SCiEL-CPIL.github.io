# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses baseline construction with optimized greedy improvement.
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines for each direction
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # Total 50 directions
    
    # For each direction, precompute the lines (each line is a list of point indices)
    # direction_lines[dir_idx] = list of lines, each line is a list of point indices
    
    # Direction 0: vertical v = (0,1)
    # Directions 1-49: v = (1, m) for m in F_49
    
    direction_lines = []
    
    # Vertical direction: v = (0, 1)
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions: v = (1, m) for m in F_49, m != (0,0)
    # Actually v = (1, m) where m ranges over all F_49 (49 values)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                # Line through (xa,xb, ya,yb) with direction (1, m):
                # Points: (xa, xb, ya, yb) + t*(1, m) for t in F_49
                # x' = xa + t_a, y' = ya + t*m where t = t_a + t_b*alpha
                # The line is determined by the "intercept" c = y - x*m
                # c = (ya, yb) - (xa, xb) * (ma, mb)
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # Now for each point, precompute which line it's on for each direction
    # point_line[point_idx][dir_idx] = line_idx within that direction
    point_line = [[None] * 50 for _ in range(n)]
    
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # Baseline construction: exclude points on Hermitian unital with y NOT in special set
    y_special = set((u*u % p, v) for u in range(p) for v in range(p))
    
    in_N = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0 and (ya, yb) not in y_special:
            in_N[i] = False
    
    # Compute initial witness counts
    # For each point Q and direction v, Q has v as witness iff
    # all other points on the line through Q in direction v are in N
    
    # For each direction and line, count points not in N
    # not_in_N_count[dir_idx][line_idx] = number of points on that line not in N
    
    not_in_N_count = [[0] * len(direction_lines[d]) for d in range(50)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            count = 0
            for pt_idx in line:
                if not in_N[pt_idx]:
                    count += 1
            not_in_N_count[dir_idx][line_idx] = count
    
    # witness_count[pt_idx] = number of directions for which pt has a witness
    witness_count = [0] * n
    for pt_idx in range(n):
        for dir_idx in range(50):
            line_idx = point_line[pt_idx][dir_idx]
            count = not_in_N_count[dir_idx][line_idx]
            # pt has dir as witness iff count == 0 or (count == 1 and not in_N[pt_idx])
            if count == 0 or (count == 1 and not in_N[pt_idx]):
                witness_count[pt_idx] += 1
    
    # Greedy improvement: try to exclude additional points
    # Candidate points: those currently in N that are on the unital
    candidates = []
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if in_N[i] and (2*xa + ya*ya - yb*yb*k) % p == 0:
            candidates.append(i)
    
    # Also try non-unital points? Maybe later. Start with unital points.
    
    # For each candidate, try to exclude it if it doesn't break any point's witness
    # When we exclude point P:
    #   For each direction d, let L = line through P in direction d
    #   not_in_N_count[d][L] increases by 1
    #   For each other point Q on L, check if Q loses direction d as witness
    #   Q loses d as witness if:
    #     Before: Q had d as witness (count_old == 0 or (count_old == 1 and not in_N[Q]))
    #     After: Q does NOT have d as witness (count_new > 1, or (count_new == 1 and in_N[Q]))
    
    # Precompute for each direction and line, the list of points on that line
    # We already have direction_lines[dir_idx][line_idx]
    
    # For efficiency, precompute for each point P and direction d, the line and other points
    # But that's 2401 * 50 * 49 which is too much memory. Instead, compute on the fly.
    
    # Try candidates in order of "least impact" - maybe those with most witness directions first?
    # Or just try all and see.
    
    # Sort candidates by witness_count descending (points with more witnesses are safer to exclude? No, 
    # actually points with fewer witnesses are more critical. Let's try excluding points that have 
    # many witness directions, as they're less likely to be the sole witness for others.)
    
    # Actually, let's just try all candidates and keep the ones that work.
    
    excluded = 0
    for cand in candidates:
        if time.time() > deadline - 20:
            break
        
        # Tentatively exclude cand
        in_N[cand] = False
        
        # Check if this breaks any point's witness
        # Only points on lines through cand in each direction are affected
        broken = False
        
        for dir_idx in range(50):
            line_idx = point_line[cand][dir_idx]
            line = direction_lines[dir_idx][line_idx]
            
            # Before: not_in_N_count was C, now it's C+1
            old_count = not_in_N_count[dir_idx][line_idx]
            new_count = old_count + 1
            
            # For each other point Q on this line, check if Q loses dir_idx as witness
            for Q in line:
                if Q == cand:
                    continue
                # Q's witness status for dir_idx:
                # Before: had witness iff old_count == 0 or (old_count == 1 and not in_N[Q])
                # After: has witness iff new_count == 0 or (new_count == 1 and not in_N[Q])
                
                had_witness = (old_count == 0) or (old_count == 1 and not in_N[Q])
                has_witness = (new_count == 0) or (new_count == 1 and not in_N[Q])
                
                if had_witness and not has_witness:
                    # Q loses this direction
                    # Check if Q has other witness directions
                    if witness_count[Q] <= 1:
                        # Q would have 0 or negative witness count -> invalid
                        broken = True
                        break
            
            if broken:
                break
        
        if broken:
            # Revert
            in_N[cand] = True
        else:
            # Update data structures
            for dir_idx in range(50):
                line_idx = point_line[cand][dir_idx]
                line = direction_lines[dir_idx][line_idx]
                old_count = not_in_N_count[dir_idx][line_idx]
                new_count = old_count + 1
                not_in_N_count[dir_idx][line_idx] = new_count
                
                for Q in line:
                    if Q == cand:
                        continue
                    had_witness = (old_count == 0) or (old_count == 1 and not in_N[Q])
                    has_witness = (new_count == 0) or (new_count == 1 and not in_N[Q])
                    if had_witness and not has_witness:
                        witness_count[Q] -= 1
                    elif not had_witness and has_witness:
                        witness_count[Q] += 1
            excluded += 1
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
