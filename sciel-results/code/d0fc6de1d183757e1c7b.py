# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using Simulated Annealing on the complement set S.
    
    The Nikodym property: for every point x in F_49^2, there exists a direction d such
    that the line L(x,d) contains no points of S other than possibly x.
    
    We maximize |S| (the complement of N). We use Simulated Annealing with add/remove moves.
    We track the number of blocked points (points with no safe direction).
    Energy = M * num_blocked - |S|. We minimize Energy.
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    p = 7
    k = 3  # alpha^2 = 3
    
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    num_dirs = 50
    
    # Precompute lines
    direction_lines = []
    
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    for ma in range(p):
        for mb in range(p):
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[pt][dir] = line_idx
    point_line = [[0] * num_dirs for _ in range(n)]
    for dir_idx in range(num_dirs):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx

    # line_members[pt][dir] = list of other points on the same line
    # This is used for the "impact" of adding/removing a point
    # Actually, we just need to iterate over the line for a given pt and dir.
    # We can use direction_lines[dir][point_line[pt][dir]]
    
    # Greedy Initialization
    def greedy_initial():
        in_S = [False] * n
        # safe_count[q] = number of directions d where count[q][d] == 0
        # Initially all counts are 0, so safe_count is 50
        safe_count = [num_dirs] * n
        counts = [[0] * num_dirs for _ in range(n)]
        
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            # Check if adding idx is valid (doesn't block anyone)
            # Simulate adding idx
            new_blocked = 0
            changes = []
            valid = True
            
            for d in range(num_dirs):
                li = point_line[idx][d]
                line = direction_lines[d][li]
                for q in line:
                    if q == idx:
                        continue
                    if counts[q][d] == 0:
                        # This direction was safe for q, now it's not
                        changes.append((q, d))
                        safe_count[q] -= 1
                        if safe_count[q] == 0:
                            new_blocked += 1
                            # We don't break immediately, we need to rollback all changes
                            # But if we know it's invalid, we can stop checking
                            # Actually, we just need to know if it's valid.
                            # If new_blocked > 0, it's invalid.
                            pass
            
            if new_blocked == 0:
                in_S[idx] = True
                # Apply changes (they are already applied in the simulation above? No, I modified safe_count directly)
                # I need to be careful. I modified safe_count and counts in the simulation.
                # I should revert if invalid.
                # Let's rewrite the simulation to not modify state until confirmed.
                
                # Actually, the code above modified `safe_count` and `counts`?
                # No, I didn't update `counts` in the loop above. I only checked `counts[q][d] == 0`.
                # But I did `safe_count[q] -= 1`. This is a modification.
                # I need to revert these modifications if invalid.
                
                # Let's do a clean simulation.
                
            else:
                # Revert changes
                for q, d in changes:
                    safe_count[q] += 1
                    # counts[q][d] was not modified in the check loop, so no need to revert counts.
                # in_S[idx] remains False
                
        return in_S, counts, safe_count

    # The greedy above is a bit messy with state modification. Let's write a cleaner one.
    def greedy_initial_clean():
        in_S = [False] * n
        counts = [[0] * num_dirs for _ in range(n)]
        safe_count = [num_dirs] * n
        
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            # Check validity
            is_valid = True
            # We need to know which (q, d) pairs would become unsafe
            # A pair (q, d) becomes unsafe if counts[q][d] == 0
            # If it becomes unsafe, safe_count[q] decreases.
            # If safe_count[q] becomes 0, q is blocked.
            
            # We can just count how many q would become blocked.
            # We don't need to track which specific d becomes unsafe for rollback if we revert,
            # we just need to revert the safe_count changes.
            
            newly_blocked = 0
            affected = []
            
            for d in range(num_dirs):
                li = point_line[idx][d]
                line = direction_lines[d][li]
                for q in line:
                    if q == idx:
                        continue
                    if counts[q][d] == 0:
                        affected.append((q, d))
                        safe_count[q] -= 1
                        if safe_count[q] == 0:
                            newly_blocked += 1
            
            if newly_blocked == 0:
                in_S[idx] = True
                # Update counts
                for q, d in affected:
                    counts[q][d] = 1
                # safe_count is already updated
            else:
                # Revert safe_count
                for q, d in affected:
                    safe_count[q] += 1
                # in_S[idx] stays False
                # counts are not updated
            
        return in_S, counts, safe_count

    best_S = None
    best_size = 0
    best_counts = None
    best_safe_count = None
    best_num_blocked = 0
    
    num_restarts = 100
    
    M = 10000  # Penalty for blocked points
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        in_S, counts, safe_count = greedy_initial_clean()
        current_size = sum(in_S)
        num_blocked = sum(1 for sc in safe_count if sc == 0)
        
        if num_blocked == 0:
            if current_size > best_size:
                best_size = current_size
                best_S = in_S[:]
                best_counts = [row[:] for row in counts]
                best_safe_count = safe_count[:]
                best_num_blocked = 0
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        max_iters = 20000
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            if T < T_min:
                break
            
            if random.random() < 0.5:
                # Add a point not in S
                candidates = [i for i in range(n) if not in_S[i]]
                if not candidates:
                    continue
                pt = random.choice(candidates)
                
                # Calculate delta
                delta_blocked = 0
                affected = []
                for d in range(num_dirs):
                    li = point_line[pt][d]
                    line = direction_lines[d][li]
                    for q in line:
                        if q == pt:
                            continue
                        if counts[q][d] == 0:
                            affected.append((q, d))
                            safe_count[q] -= 1
                            if safe_count[q] == 0:
                                delta_blocked += 1
                
                delta_E = M * delta_blocked - 1
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Accept
                    in_S[pt] = True
                    current_size += 1
                    for q, d in affected:
                        counts[q][d] = 1
                    num_blocked += delta_blocked
                else:
                    # Reject, revert
                    for q, d in affected:
                        safe_count[q] += 1
                    
            else:
                # Remove a point from S
                candidates = [i for i in range(n) if in_S[i]]
                if not candidates:
                    continue
                pt = random.choice(candidates)
                
                # Calculate delta
                delta_blocked = 0
                affected = []
                for d in range(num_dirs):
                    li = point_line[pt][d]
                    line = direction_lines[d][li]
                    for q in line:
                        if q == pt:
                            continue
                        if counts[q][d] == 1:
                            # This point was the only one blocking q in direction d
                            # If we remove it, q gains this direction
                            affected.append((q, d))
                            safe_count[q] += 1
                            if safe_count[q] == 1:
                                # q was blocked, now unblocked
                                delta_blocked -= 1 # num_blocked decreases
                
                delta_E = M * delta_blocked + 1 # |S| decreases by 1, so -|S| increases by 1
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Accept
                    in_S[pt] = False
                    current_size -= 1
                    for q, d in affected:
                        counts[q][d] = 0
                    num_blocked += delta_blocked
                else:
                    # Reject, revert
                    for q, d in affected:
                        safe_count[q] -= 1
            
            T *= cooling
            
            if num_blocked == 0 and current_size > best_size:
                best_size = current_size
                best_S = in_S[:]
                best_counts = [row[:] for row in counts]
                best_safe_count = safe_count[:]
                best_num_blocked = 0
    
    # Fallback
    if best_S is None:
        best_S = [False] * n
    
    # Build result: N = All \ S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
