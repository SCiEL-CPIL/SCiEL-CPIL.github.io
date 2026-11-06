import time
import random
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

# New structural change: Set-cover based construction
# Instead of greedy add, we view it as covering the grid with blocked sets.
# But we need to ensure the set is independent (no isosceles).
# This is a hypergraph independent set problem.
# Let's try a different heuristic: Maximize the "blocking power" per point added.

def greedy_construct_set_cover():
    S = set()
    remaining = set(range(GRID_SIZE))
    
    while remaining:
        best_p = None
        best_gain = -1
        
        for p in remaining:
            # Check if p can be added
            valid = True
            for q in S:
                for r in S:
                    if r == q: continue
                    if (BLOCK_MASK[q][r] >> p) & 1:
                        valid = False
                        break
                if not valid: break
            
            if not valid:
                continue
            
            # Calculate gain: how many points in remaining does p block?
            # A point x is blocked if there exists q, r in S U {p} such that x is in block_mask[q][r].
            # This is expensive to calculate exactly.
            # Approximation: count how many points in remaining are in block_mask[p][q] for any q in S.
            # Or simply: how many points in remaining are "close" to p in some metric?
            # Let's stick to a simpler heuristic: prefer points that block many others.
            # We can precompute for each p, the set of points it blocks with each q.
            # For now, let's just use random order but with a bias towards points that have high degree in the "blocking graph".
            # Degree of p = number of q such that p blocks q or q blocks p?
            # Actually, let's just try to maximize the size of the blocked set.
            
            # For each candidate p, calculate the number of points in remaining that would be blocked by adding p.
            # This requires checking for each x in remaining if x is blocked by S U {p}.
            # This is O(|remaining| * |S|^2) per candidate p.
            # Total O(|remaining|^2 * |S|^2).
            # |remaining| ~ 256, |S| ~ 10. 256^2 * 100 = 6.5M. Too slow per iteration.
            
            # Simpler heuristic:
            # Count how many points in remaining are in the union of block_mask[p][q] for q in S.
            # This is O(|remaining| * |S|) bitset ops.
            
            blocked_count = 0
            for x in remaining:
                if x == p: continue
                # Check if x is blocked by p with some q in S
                is_blocked = False
                for q in S:
                    if (BLOCK_MASK[p][q] >> x) & 1:
                        is_blocked = True
                        break
                if is_blocked:
                    blocked_count += 1
            
            if blocked_count > best_gain:
                best_gain = blocked_count
                best_p = p
        
        if best_p is None:
            # No point can be added? Should not happen if we start empty.
            # But maybe all remaining are blocked by S?
            # If so, we are done.
            break
        
        S.add(best_p)
        remaining.remove(best_p)
        
        # Remove points that are now blocked? No, we want to keep them in remaining to check if they can be added.
        # But if a point x is blocked by S, it cannot be added.
        # So we can remove blocked points from remaining to speed up.
        new_remaining = set()
        for x in remaining:
            # Check if x is blocked by S
            is_blocked = False
            for q in S:
                for r in S:
                    if q == r: continue
                    if (BLOCK_MASK[q][r] >> x) & 1:
                        is_blocked = True
                        break
                if is_blocked: break
            if not is_blocked:
                new_remaining.add(x)
        remaining = new_remaining
        
        if not remaining:
            break
            
    return S

def _construct_impl_new():
    best_S = set()
    best_size = GRID_SIZE + 1
    
    # Run multiple restarts
    while time.time() - START_TIME < TIME_LIMIT - 10:
        # Greedy start with set-cover heuristic
        S = greedy_construct_set_cover()
        
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

# Override construct to use the new method
def construct():
    try:
        return _construct_impl_new()
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

BET: {mechanism: "Replaces random greedy with a set-cover heuristic that prioritizes points blocking many others, potentially finding smaller maximal sets faster.", q: 0.6, delta: [0.0, 0.5]}
