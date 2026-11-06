import itertools
import random
import sys

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles relationships
# pair_to_qs: dict mapping (idx_i, idx_j) to set of idx_k such that {i,j,k} is isosceles
pair_to_qs = {}

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        p_i = POINTS[i]
        p_j = POINTS[j]
        d_ij = dist2(p_i, p_j)
        
        blocked = set()
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            p_k = POINTS[k]
            d_ik = dist2(p_i, p_k)
            d_jk = dist2(p_j, p_k)
            
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                blocked.add(k)
        
        pair_to_qs[(i, j)] = blocked
        pair_to_qs[(j, i)] = blocked # Symmetric, though we usually access with sorted keys

def construct():
    S = set()
    blocked_count = [0] * NUM_POINTS
    
    def is_unblocked(q):
        return q not in S and blocked_count[q] == 0

    def add_point(p):
        S.add(p)
        for r in S:
            if r == p:
                continue
            key = (p, r) if p < r else (r, p)
            for q in pair_to_qs[key]:
                if q != p and q not in S:
                    blocked_count[q] += 1

    def remove_point(p):
        S.remove(p)
        for r in S:
            key = (p, r) if p < r else (r, p)
            for q in pair_to_qs[key]:
                if q != p and q not in S:
                    blocked_count[q] -= 1

    # Greedy Construction
    while True:
        unblocked = [q for q in range(NUM_POINTS) if is_unblocked(q)]
        if not unblocked:
            break
        
        # Select best candidate
        best_p = None
        best_score = -1
        
        for p in unblocked:
            # Calculate how many unblocked points p will block
            new_blocked = 0
            for r in S:
                key = (p, r) if p < r else (r, p)
                for q in pair_to_qs[key]:
                    if q != p and is_unblocked(q):
                        new_blocked += 1
            if new_blocked > best_score:
                best_score = new_blocked
                best_p = p
        
        if best_p is None:
            break
        add_point(best_p)

    # Local Search: Try to shrink S
    # We try to remove a point p. If S\{p} is not maximal, we greedily add points to fix it.
    # If the resulting size is smaller, we accept.
    improved = True
    while improved:
        improved = False
        # Shuffle to break symmetry
        s_list = list(S)
        random.shuffle(s_list)
        
        for p in s_list:
            if p not in S:
                continue
            
            # Save state
            old_S = set(S)
            old_blocked = blocked_count[:]
            
            # Remove p
            remove_point(p)
            
            # Check if maximal
            unblocked = [q for q in range(NUM_POINTS) if is_unblocked(q)]
            
            if not unblocked:
                # S\{p} is maximal and independent. We reduced size by 1.
                # Accept
                improved = True
                break
            else:
                # S\{p} is not maximal. We must add points.
                # Greedily add points to restore maximality
                temp_S_size = len(S)
                # We want to see if we can restore maximality with < 1 added points?
                # Impossible, we need at least 1 if unblocked is not empty.
                # So size will be at least len(S) + 1.
                # Original size was len(S) + 1.
                # New size will be >= len(S) + 1.
                # So we cannot improve size by removing p if maximality requires adding back >= 1 point.
                # Wait.
                # If we remove p, size decreases by 1.
                # If we add k points, size increases by k.
                # Net change: k - 1.
                # We need k - 1 < 0 => k < 1 => k = 0.
                # So we ONLY accept removal if S\{p} is already maximal.
                # If S\{p} is not maximal, we MUST add points.
                # The minimum number of points to add is 1 (if one point blocks all others).
                # So size becomes (len(S)+1) - 1 + 1 = len(S)+1. No improvement.
                # Unless... adding one point allows us to remove another point later?
                # That's a 2-opt move.
                
                # For now, only accept if k=0.
                # Restore state
                S.clear()
                S.update(old_S)
                for i in range(NUM_POINTS):
                    blocked_count[i] = old_blocked[i]
                # We need to re-add p? No, S was restored to old_S which included p.
                # But blocked_count was restored.
                # We just need to continue.
                continue

    return [POINTS[q] for q in S]
