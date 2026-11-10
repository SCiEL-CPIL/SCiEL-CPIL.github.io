# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Strategy:
    1. Greedy construction of the complement set S (points NOT in N).
    2. Local Search: Try to swap points in S with points not in S to increase |S|.
       Specifically, if a point q cannot be added because it blocks the last safe direction
       of a point r, we try to remove a point p in S that is on a different line through r,
       thereby freeing up another safe direction for r, which then allows q to be added.
       We perform such swaps if they result in a net gain or maintain size while potentially
       unlocking further gains.
    """
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin
    
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
    
    # Precompute line_membership: line_membership[dir][line_idx] = list of points
    # Already have direction_lines[dir][line_idx]
    
    # safe_mask[q]: bit d is 1 if direction d is safe for q.
    # Initially all 1s: (1 << 50) - 1.
    
    best_S = [False] * n
    best_size = 0
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Randomize order
        order = list(range(n))
        random.shuffle(order)
        
        in_S = [False] * n
        safe_mask = [(1 << 50) - 1] * n
        current_size = 0
        
        # Phase 1: Greedy Addition
        for idx in order:
            # Check if we can add idx to S
            can_add = True
            for d in range(50):
                li = point_line[idx][d]
                line = direction_lines[d][li]
                for q in line:
                    if q == idx:
                        continue
                    # If q has exactly one safe direction, and it is d, then adding idx blocks it.
                    if safe_mask[q] == (1 << d):
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                # Add idx to S
                in_S[idx] = True
                current_size += 1
                
                # Update safe masks
                for d in range(50):
                    li = point_line[idx][d]
                    line = direction_lines[d][li]
                    for q in line:
                        if q == idx:
                            continue
                        if (safe_mask[q] >> d) & 1:
                            safe_mask[q] &= ~(1 << d)
        
        # Phase 2: Local Search (Swaps)
        # We try to add points that are not in S.
        # If a point q cannot be added, it's because some r has only one safe direction d,
        # and q is on L(r, d).
        # To add q, we need to free up another direction d' for r.
        # We can do this by removing all points in S that are on L(r, d') (excluding r).
        # Let R = {p in S | p in L(r, d') and p != r}.
        # If we remove R, r becomes safe in direction d'. Then we can add q.
        # Net change: -|R| + 1.
        # We only want to do this if -|R| + 1 >= 0, i.e., |R| <= 1.
        # If |R| == 0, we could have added q directly (contradiction).
        # So we look for cases where |R| == 1.
        # This is a 1-for-1 swap: remove p, add q.
        
        # To make this efficient, we maintain a list of "blocked" points and "blockers".
        # For each q not in S, find the reason it's blocked.
        # If blocked by r in direction d, we need to remove a point p in S on L(r, d') for some d' != d.
        
        # Let's iterate a few times, trying to find beneficial or neutral swaps.
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Find candidates for addition
            candidates = [i for i in range(n) if not in_S[i]]
            random.shuffle(candidates)
            
            for q in candidates:
                if time.time() > deadline - 10:
                    break
                
                # Check if q can be added
                can_add = True
                blocking_r = -1
                blocking_d = -1
                for d in range(50):
                    li = point_line[q][d]
                    line = direction_lines[d][li]
                    for r_pt in line:
                        if r_pt == q:
                            continue
                        if safe_mask[r_pt] == (1 << d):
                            can_add = False
                            blocking_r = r_pt
                            blocking_d = d
                            break
                    if not can_add:
                        break
                
                if can_add:
                    # Add q
                    in_S[q] = True
                    current_size += 1
                    for d in range(50):
                        li = point_line[q][d]
                        line = direction_lines[d][li]
                        for r_pt in line:
                            if r_pt == q:
                                continue
                            if (safe_mask[r_pt] >> d) & 1:
                                safe_mask[r_pt] &= ~(1 << d)
                    improved = True
                else:
                    # Try to find a swap
                    # We are blocked by r_pt in direction blocking_d.
                    # We need to find a direction d' for r_pt such that 
                    # the number of points in S on L(r_pt, d') (excluding r_pt) is 1.
                    # Let that point be p.
                    # If we remove p, r_pt becomes safe in d'. Then we can add q.
                    # Check if such a swap exists.
                    
                    found_swap = False
                    for d_prime in range(50):
                        if d_prime == blocking_d:
                            continue
                        
                        li_prime = point_line[blocking_r][d_prime]
                        line_prime = direction_lines[d_prime][li_prime]
                        
                        # Count points in S on this line (excluding blocking_r)
                        pts_in_S_on_line = []
                        for p_pt in line_prime:
                            if p_pt == blocking_r:
                                continue
                            if in_S[p_pt]:
                                pts_in_S_on_line.append(p_pt)
                        
                        if len(pts_in_S_on_line) == 1:
                            p_pt = pts_in_S_on_line[0]
                            # Perform swap: remove p_pt, add q
                            # 1. Remove p_pt from S
                            in_S[p_pt] = False
                            current_size -= 1
                            
                            # Update safe masks: p_pt is no longer in S, so it unblocks directions.
                            # For any point u, if p_pt was on L(u, d_u) and d_u was blocked by p_pt,
                            # d_u might become safe again.
                            # Specifically, for any u, if (safe_mask[u] & ~(1<<d_u)) == 0 and p_pt in L(u, d_u),
                            # then safe_mask[u] |= (1 << d_u).
                            # This is expensive to do globally.
                            # However, we only care about whether the swap is valid.
                            # Since we are removing a point from S, it can only help validity.
                            # So we just need to ensure that adding q doesn't break anything else.
                            # But wait, we removed p_pt. This might make other points safe,
                            # which is good. It doesn't make any point unsafe.
                            # So the only risk is adding q.
                            # We already know q was blocked only by blocking_r in blocking_d.
                            # After removing p_pt, blocking_r has a new safe direction d_prime.
                            # So blocking_r is no longer in the "only one safe direction" state.
                            # Are there other points that q blocks?
                            # We checked that q is blocked by blocking_r.
                            # It might also be blocked by other points?
                            # The check `if safe_mask[r_pt] == (1 << d)` finds the FIRST blocker.
                            # If there are multiple blockers, we need to handle them all.
                            # For simplicity, let's assume we only handle single blockers.
                            # If there are multiple, this swap might not work.
                            
                            # Let's re-verify if q can be added after removing p_pt.
                            # We need to simulate the removal of p_pt to update safe_masks?
                            # Or just check if any point r' has only one safe direction d'
                            # and q is on L(r', d').
                            
                            # To be safe, let's actually update the safe masks for the swap.
                            # This is O(n * 50) per swap, which is too slow if we do many.
                            # But we only do it if we find a candidate.
                            
                            # Let's try a simpler check:
                            # Does q block any point r' that has only one safe direction d'
                            # where d' is NOT affected by the removal of p_pt?
                            # Actually, removing p_pt only ADDS safe directions.
                            # So if q was blocked by r' in d', and d' is still the only safe direction
                            # for r' after removing p_pt, then q still blocks r'.
                            # p_pt removal helps r' only if p_pt was on L(r', d'') for some d''.
                            # If p_pt was on L(r', d'), it doesn't help r' in direction d'.
                            
                            # So, if q was blocked by multiple points, we need to remove blockers for ALL of them.
                            # This swap only removes one blocker.
                            # So this swap is only valid if q is blocked by EXACTLY one point.
                            
                            # Let's check if q is blocked by exactly one point.
                            num_blockers = 0
                            for d_check in range(50):
                                li_check = point_line[q][d_check]
                                line_check = direction_lines[d_check][li_check]
                                for r_check in line_check:
                                    if r_check == q:
                                        continue
                                    if safe_mask[r_check] == (1 << d_check):
                                        num_blockers += 1
                                        if num_blockers > 1:
                                            break
                                if num_blockers > 1:
                                    break
                            
                            if num_blockers == 1:
                                # Proceed with swap
                                # Update safe masks for removal of p_pt
                                for d_upd in range(50):
                                    li_upd = point_line[p_pt][d_upd]
                                    line_upd = direction_lines[d_upd][li_upd]
                                    for u in line_upd:
                                        if u == p_pt:
                                            continue
                                        # If u had d_upd blocked by p_pt, it might be safe now.
                                        # Check if d_upd was the only missing bit?
                                        # We don't track "missing" bits, we track "safe" bits.
                                        # safe_mask[u] has bit d_upd = 0 if blocked.
                                        # If we remove p_pt, bit d_upd might become 1.
                                        # It becomes 1 if NO other point in S is on L(u, d_upd).
                                        # This check is expensive.
                                        
                                        # Approximation: Just set the bit to 1?
                                        # No, that's incorrect.
                                        
                                        # Let's just re-evaluate safe_mask for u?
                                        # Too slow.
                                        
                                        # Alternative: Don't update safe_mask immediately.
                                        # Instead, just check if the new state is valid.
                                        # But we need to update in_S.
                                        
                                        # Let's try a different approach:
                                        # Just perform the swap and then verify the whole set?
                                        # Too slow.
                                        
                                        # Let's assume the swap is valid if num_blockers == 1.
                                        # This is a heuristic.
                                        
                                        pass
                            
                            if num_blockers == 1:
                                # Commit the swap
                                in_S[q] = True
                                current_size += 1
                                
                                # We must update safe_mask correctly.
                                # Removal of p_pt:
                                for d_upd in range(50):
                                    li_upd = point_line[p_pt][d_upd]
                                    line_upd = direction_lines[d_upd][li_upd]
                                    for u in line_upd:
                                        if u == p_pt:
                                            continue
                                        # Check if d_upd is now safe for u
                                        # d_upd is safe if no point in S is on L(u, d_upd) except u.
                                        # We know p_pt was on L(u, d_upd).
                                        # If p_pt was the ONLY point in S on L(u, d_upd), then d_upd becomes safe.
                                        
                                        # Count points in S on L(u, d_upd)
                                        count_s = 0
                                        for v in line_upd:
                                            if v == u:
                                                continue
                                            if in_S[v]:
                                                count_s += 1
                                                if count_s > 0:
                                                    break
                                        
                                        if count_s == 0:
                                            safe_mask[u] |= (1 << d_upd)
                                
                                # Addition of q:
                                for d_add in range(50):
                                    li_add = point_line[q][d_add]
                                    line_add = direction_lines[d_add][li_add]
                                    for u in line_add:
                                        if u == q:
                                            continue
                                        if (safe_mask[u] >> d_add) & 1:
                                            safe_mask[u] &= ~(1 << d_add)
                                
                                improved = True
                                found_swap = True
                                break
                        
                        if found_swap:
                            break
                    
                    if found_swap:
                        break
        
        if current_size > best_size:
            best_size = current_size
            best_S = in_S[:]
            
            # Early stop if we reach a very good size?
            if best_size > 200:
                pass

    # Build result N = complement of S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    if not pts:
        pts = all_points
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
