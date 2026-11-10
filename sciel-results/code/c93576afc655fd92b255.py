# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a fast greedy local search
    based on line satisfaction counts.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    n = q * q  # 2401
    num_dirs = 50  # 1 vertical + 49 slopes

    # Precompute line memberships
    # lines[dir_idx][line_idx] = list of point indices
    # point_to_lines[pt_idx] = list of (dir_idx, line_idx)
    
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)

    lines = []
    point_to_lines = [[] for _ in range(n)]
    
    # Vertical lines
    vert_lines = {}
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        key = (xa, xb)
        if key not in vert_lines:
            vert_lines[key] = []
        vert_lines[key].append(i)
    vert_list = list(vert_lines.values())
    lines.append(vert_list)
    for line_idx, line in enumerate(vert_list):
        for pt in line:
            point_to_lines[pt].append((0, line_idx))

    # Slope lines
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = {}
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                key = c
                if key not in lines_dict:
                    lines_dict[key] = []
                lines_dict[key].append(i)
            slope_list = list(lines_dict.values())
            lines.append(slope_list)
            dir_idx = 1 + ma * p + mb
            for line_idx, line in enumerate(slope_list):
                for pt in line:
                    point_to_lines[pt].append((dir_idx, line_idx))

    # Total lines
    total_lines = 0
    for d in range(num_dirs):
        total_lines += len(lines[d])
        
    # line_id mapping: (dir, line_idx) -> global_id
    # We'll just use (dir, line_idx) as keys in dicts or flatten.
    # Let's flatten for speed.
    line_global_id = {}
    global_line_points = []
    gid = 0
    for d in range(num_dirs):
        for li in range(len(lines[d])):
            line_global_id[(d, li)] = gid
            global_line_points.append(lines[d][li])
            gid += 1
            
    # Initialize state for a run
    def run_greedy(seed):
        random.seed(seed)
        # in_S: boolean array
        in_S = np.zeros(n, dtype=bool)
        # line_count[gid]: number of points in S on this line
        line_count = np.zeros(gid, dtype=np.int32)
        # line_single[gid]: index of point if count==1, else -1
        line_single = np.full(gid, -1, dtype=np.int32)
        # satisfied[pt]: number of lines satisfying pt
        satisfied = np.full(n, num_dirs, dtype=np.int32)
        
        current_size = 0
        candidates = list(range(n))
        random.shuffle(candidates)
        
        # We will iterate through candidates. If we add a point, we update state.
        # We can do multiple passes or just one pass with restarts.
        # Let's do a single pass, then restart.
        
        for x in candidates:
            if time.time() > deadline - 10:
                break
            if in_S[x]:
                continue
            
            # Check if x can be added
            # Calculate potential drops in 'satisfied'
            can_add = True
            drops = [] # list of (pt_idx, drop_amount)
            
            for d, li in point_to_lines[x]:
                gid = line_global_id[(d, li)]
                cnt = line_count[gid]
                
                if cnt == 0:
                    # Adding x makes cnt=1.
                    # Before: All pts in line were satisfied by this line.
                    # After: Only x is satisfied by this line.
                    # Drop for y in line \ {x}: 1
                    # Drop for x: 0 (was satisfied, still satisfied)
                    for y in global_line_points[gid]:
                        if y != x:
                            if satisfied[y] <= 1:
                                can_add = False
                                break
                            drops.append((y, 1))
                    if not can_add:
                        break
                elif cnt == 1:
                    # Adding x makes cnt=2.
                    # Before: Only z (line_single) was satisfied.
                    # After: No one is satisfied.
                    # Drop for z: 1
                    z = line_single[gid]
                    if satisfied[z] <= 1:
                        can_add = False
                        break
                    drops.append((z, 1))
                else:
                    # cnt >= 2
                    # Before: No one satisfied.
                    # After: No one satisfied.
                    # No drop.
                    pass
            
            if can_add:
                in_S[x] = True
                current_size += 1
                
                # Apply drops
                for y, drop in drops:
                    satisfied[y] -= drop
                
                # Update line states
                for d, li in point_to_lines[x]:
                    gid = line_global_id[(d, li)]
                    old_cnt = line_count[gid]
                    if old_cnt == 0:
                        line_count[gid] = 1
                        line_single[gid] = x
                    elif old_cnt == 1:
                        line_count[gid] = 2
                        line_single[gid] = -1
                    else:
                        line_count[gid] += 1
                        # line_single remains -1

        return current_size, in_S

    best_size = 0
    best_S = None
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        size, S = run_greedy(i)
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
    # Final verification and construction
    if best_S is None:
        # Fallback to empty S (N is full set)
        pts = all_points
    else:
        # Verify best_S
        # Recompute satisfied to be sure
        in_S = best_S
        line_count = np.zeros(gid, dtype=np.int32)
        for x in range(n):
            if in_S[x]:
                for d, li in point_to_lines[x]:
                    gid = line_global_id[(d, li)]
                    line_count[gid] += 1
        
        satisfied = np.zeros(n, dtype=np.int32)
        for d in range(num_dirs):
            for li in range(len(lines[d])):
                gid = line_global_id[(d, li)]
                cnt = line_count[gid]
                if cnt == 0:
                    # All pts satisfied
                    for pt in global_line_points[gid]:
                        satisfied[pt] += 1
                elif cnt == 1:
                    # Find the single point
                    for pt in global_line_points[gid]:
                        if in_S[pt]:
                            satisfied[pt] += 1
                            break
                # else: no one satisfied
        
        if np.all(satisfied >= 1):
            pts = [all_points[i] for i in range(n) if not in_S[i]]
        else:
            pts = all_points

    if not pts:
        pts = all_points

    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
