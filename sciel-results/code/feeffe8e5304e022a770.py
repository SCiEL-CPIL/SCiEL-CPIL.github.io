import time
import random
import numpy as np

N = 16
NUM_PTS = N * N
FULL_MASK = (1 << NUM_PTS) - 1

# Precompute geometry
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute Iso masks
# Iso[i][j] is a bitmask of k such that triangle i,j,k is isosceles
Iso = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    for j in range(i + 1, NUM_PTS):
        d_ij = dist[i][j]
        mask = 0
        for k in range(NUM_PTS):
            if k == i or k == j:
                continue
            d_ik = dist[i][k]
            d_jk = dist[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        Iso[i][j] = mask
        Iso[j][i] = mask

def get_blocked_mask(S_list):
    """Return the union of Iso masks for all pairs in S_list."""
    mask = 0
    n = len(S_list)
    for i in range(n):
        pi = S_list[i]
        for j in range(i + 1, n):
            pj = S_list[j]
            mask |= Iso[pi][pj]
    return mask

def is_valid(S_list):
    """Check if no three points in S_list form an isosceles triangle."""
    if len(S_list) < 3:
        return True
    blocked = get_blocked_mask(S_list)
    S_mask = 0
    for p in S_list:
        S_mask |= (1 << p)
    return (S_mask & blocked) == 0

def is_maximal(S_list):
    """Check if no point outside S_list can be added."""
    if not is_valid(S_list):
        return False
    blocked = get_blocked_mask(S_list)
    S_mask = 0
    for p in S_list:
        S_mask |= (1 << p)
    return (S_mask | blocked) == FULL_MASK

def construct():
    DEADLINE = time.time() + 290
    best_set = None
    best_len = float('inf')
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        
        S = []
        
        for p in order:
            # Check if p can be added
            can_add = True
            # Optimization: check against current S
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if (Iso[S[i]][S[j]] >> p) & 1:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                S.append(p)
                
        # Ensure maximal by adding any remaining valid points
        # Actually, the greedy loop above adds points if they don't conflict with existing pairs.
        # But does it ensure maximality?
        # If a point p is not added, it means it conflicts with some pair in S.
        # So it is blocked.
        # Thus, the resulting set is maximal.
        return S

    # Run multiple greedy restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > DEADLINE:
            break
        S = greedy_construct(i)
        
        if len(S) < best_len:
            best_len = len(S)
            best_set = S.copy()
            
    # Local Search on the best set found
    if best_set is not None:
        S = best_set
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            n = len(S)
            # Try removals
            for i in range(n):
                S_trial = S[:i] + S[i+1:]
                if is_maximal(S_trial):
                    S = S_trial
                    improved = True
                    break
            
            if not improved:
                # Try swaps
                S_mask = 0
                for p in S:
                    S_mask |= (1 << p)
                
                # Randomly pick a point to remove
                if S:
                    i = random.randint(0, len(S) - 1)
                    q = S[i]
                    S_without_q = S[:i] + S[i+1:]
                    
                    # Find candidates to add
                    # Candidates must not be in S and must not be blocked by S_without_q
                    # And the new set must be valid
                    # And the new set must be maximal
                    
                    # To be maximal, the new set must block all points not in it.
                    # This is hard to check incrementally.
                    # Just try a few random candidates.
                    candidates = [p for p in range(NUM_PTS) if not (S_mask >> p) & 1]
                    random.shuffle(candidates)
                    
                    found = False
                    for p in candidates[:20]:
                        S_swap = S_without_q + [p]
                        if is_valid(S_swap) and is_maximal(S_swap):
                            S = S_swap
                            improved = True
                            found = True
                            break
                            
                    if not found:
                         # Try removing a different point or stopping
                         pass

    if best_set is None:
        best_set = greedy_construct(0)
    
    return [POINTS[i] for i in best_set]
