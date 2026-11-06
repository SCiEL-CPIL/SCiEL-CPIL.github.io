import itertools
import random
import time
import sys

N = 16
GRID_SIZE = N * N
TIME_LIMIT = 300
START_TIME = time.time()

# Precomputation
# Points are 0..255
# dist2[i, j]
# block_mask[i, j] is an integer bitset of points x such that {i, j, x} is isosceles

def precompute():
    points = [(x, y) for y in range(N) for x in range(N)]
    dist2 = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        xi, yi = points[i]
        for j in range(GRID_SIZE):
            xj, yj = points[j]
            dist2[i][j] = (xi-xj)**2 + (yi-yj)**2
    
    block_mask = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        for j in range(i+1, GRID_SIZE):
            mask = 0
            d_ij = dist2[i][j]
            for x in range(GRID_SIZE):
                if x == i or x == j:
                    continue
                d_ix = dist2[i][x]
                d_jx = dist2[j][x]
                if d_ix == d_jx or d_ix == d_ij or d_jx == d_ij:
                    mask |= (1 << x)
            block_mask[i][j] = mask
            block_mask[j][i] = mask
    return points, dist2, block_mask

POINTS, DIST2, BLOCK_MASK = precompute()

def is_valid_set(S):
    # S is a set of indices
    # Check no-isosceles
    for i in S:
        for j in S:
            if i == j: continue
            # Check if i is blocked by pair (j, k) for any k in S?
            # Actually, just check pairs
    # More efficient:
    for i in S:
        for j in S:
            if i == j: continue
            # If i is in block_mask[j][k] for any k in S, k!=i, then invalid
            # But block_mask[j][k] includes i if {j,k,i} is isosceles.
            # We need to ensure no such k exists.
            # This is O(|S|^3) if done naively.
            # With bitsets:
            # For a fixed i, we want to know if there exist j, k in S, j!=i, k!=i, j!=k such that i is in block_mask[j][k].
            # This is still hard to do with a single bitset operation for all j,k.
            # However, during construction/reduction, we can maintain validity.
            pass
    return True

def greedy_construct():
    S = set()
    order = list(range(GRID_SIZE))
    random.shuffle(order)
    for p in order:
        # Check if p can be added to S
        valid = True
        # Check against all pairs in S
        # If S is small, this is fast.
        # For each q in S, we need to ensure {p, q, r} is not isosceles for any r in S.
        # Equivalent: p is NOT in block_mask[q][r] for any q,r in S, q!=p, r!=p.
        # Since p is not in S yet, q,r are distinct from p.
        for q in S:
            # If p is in block_mask[q][r] for any r in S, r!=q, then invalid.
            # We can iterate r
            for r in S:
                if r == q: continue
                if (BLOCK_MASK[q][r] >> p) & 1:
                    valid = False
                    break
            if not valid: break
        if valid:
            S.add(p)
    return S

def is_maximal(S):
    # Check if all x not in S are blocked by some pair in S
    # blocked_by[x] = number of pairs in S that block x
    # If blocked_by[x] == 0 for any x not in S, not maximal.
    # We can compute this on the fly or maintain it.
    # For simplicity in the reduction step, we will check specifically.
    
    # For each x not in S:
    #   Is there a pair (a,b) in S such that x is in block_mask[a][b]?
    for x in range(GRID_SIZE):
        if x in S: continue
        blocked = False
        # Check all pairs in S
        # Optimization: If S is large, this is slow.
        # But S is expected to be small (< 30).
        # Pairs in S: ~400.
        # 256 * 400 = 100k ops. Fast enough.
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                if (BLOCK_MASK[a][b] >> x) & 1:
                    blocked = True
                    break
            if blocked: break
        if not blocked:
            return False
    return True

def try_remove(S, p):
    # Check if S \ {p} is maximal
    # 1. p must be blocked by S \ {p}
    # 2. All x in grid \ (S \ {p}) must be blocked by S \ {p}
    #    Note: grid \ (S \ {p}) includes p and (grid \ S).
    
    S_new = S - {p}
    s_list = list(S_new)
    
    # Check p
    p_blocked = False
    for i in range(len(s_list)):
        a = s_list[i]
        for j in range(i+1, len(s_list)):
            b = s_list[j]
            if (BLOCK_MASK[a][b] >> p) & 1:
                p_blocked = True
                break
        if p_blocked: break
    if not p_blocked:
        return None # Cannot remove p

    # Check others
    for x in range(GRID_SIZE):
        if x in S_new: continue
        # x is either p (already checked) or x not in S
        if x == p: continue
        blocked = False
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                if (BLOCK_MASK[a][b] >> x) & 1:
                    blocked = True
                    break
            if blocked: break
        if not blocked:
            return None # Cannot remove p

    return S_new

def reduce_set(S):
    # Local search: try to remove points
    improved = True
    while improved:
        improved = False
        # Try removing each point in random order
        p_list = list(S)
        random.shuffle(p_list)
        for p in p_list:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return S
            S_new = try_remove(S, p)
            if S_new is not None:
                S = S_new
                improved = True
                break # Restart loop with new S
    return S

def construct():
    best_S = set()
    best_size = GRID_SIZE + 1
    
    # Run multiple restarts
    while time.time() - START_TIME < TIME_LIMIT - 10:
        # Greedy start
        S = greedy_construct()
        
        # Reduce
        S = reduce_set(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = set(S)
            
    return [POINTS[i] for i in best_S]

# Fallback if construct is too slow or fails
def construct():
    try:
        return _construct_impl()
    except Exception as e:
        # Return a simple greedy set as fallback
        S = set()
        for p in range(GRID_SIZE):
            valid = True
            for q in S:
                for r in S:
                    if r == q: continue
                    if (BLOCK_MASK[q][r] >> p) & 1:
                        valid = False
                        break
                if not valid: break
            if valid:
                S.add(p)
        return [POINTS[i] for i in S]

def _construct_impl():
    best_S = set()
    best_size = GRID_SIZE + 1
    
    # Run multiple restarts
    while time.time() - START_TIME < TIME_LIMIT - 10:
        # Greedy start
        S = greedy_construct()
        
        # Reduce
        S = reduce_set(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = set(S)
            
    if not best_S:
        # Fallback
        S = greedy_construct()
        best_S = S
        
    return [POINTS[i] for i in best_S]
