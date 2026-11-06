import time
import random
import sys

N = 16
GRID_SIZE = N * N
TIME_LIMIT = 325.0
START_TIME = time.time()

# Precomputation
# Points are 0..255
# dist2[i, j]
# block_mask[i, j] is an integer bitset of points x such that {i, j, x} is isosceles

def precompute():
    points = [(x, y) for y in range(N) for x in range(N)]
    # dist2[i][j]
    dist2 = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        xi, yi = points[i]
        for j in range(GRID_SIZE):
            xj, yj = points[j]
            dist2[i][j] = (xi-xj)**2 + (yi-yj)**2
    
    # block_mask[i][j]
    # We only need i < j for storage, but access should be symmetric or we store both.
    # Storing both is 256*256 integers, which is fine.
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
ALL_POINTS_MASK = (1 << GRID_SIZE) - 1

def popcount(x):
    return bin(x).count('1')

def greedy_construct(rng):
    S = set()
    blocked_mask = 0
    
    while True:
        # Find addable points: not in S, not in blocked_mask
        # addable_mask = ALL_POINTS_MASK & ~blocked_mask & ~bitmask(S)
        s_mask = 0
        for s in S:
            s_mask |= (1 << s)
        
        addable_mask = ALL_POINTS_MASK & ~blocked_mask & ~s_mask
        
        if addable_mask == 0:
            break
        
        # Get list of addable indices
        addable_indices = []
        m = addable_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            addable_indices.append(idx)
            m ^= lsb
        
        best_p = None
        best_gain = -1
        best_new_blocked = 0
        
        for p in addable_indices:
            # Calculate new blocked mask if we add p
            new_blocked = blocked_mask
            for s in S:
                new_blocked |= BLOCK_MASK[p][s]
            
            # Gain is number of newly blocked points
            # We don't strictly need popcount if we just compare masks, 
            # but gain helps break ties or guide search.
            # Actually, we just want to maximize the coverage.
            # Since we add p, p itself is not blocked (it's in S now), 
            # but other points might be.
            # The condition for maximality is that ALL points are blocked or in S.
            # So we want to maximize the union of blocked points.
            
            # Simple heuristic: maximize popcount of new_blocked
            gain = popcount(new_blocked)
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
                best_new_blocked = new_blocked
        
        if best_p is None:
            break
            
        S.add(best_p)
        blocked_mask = best_new_blocked
        
    return S

def is_maximal(S):
    if not S:
        # Empty set is maximal only if grid is empty, which is false
        return False
        
    s_mask = 0
    for s in S:
        s_mask |= (1 << s)
        
    blocked_mask = 0
    s_list = list(S)
    for i in range(len(s_list)):
        for j in range(i+1, len(s_list)):
            blocked_mask |= BLOCK_MASK[s_list[i]][s_list[j]]
            
    # Maximal if every point is either in S or blocked
    return (blocked_mask | s_mask) == ALL_POINTS_MASK

def reduce_set(S, rng):
    S = set(S)
    improved = True
    while improved:
        improved = False
        s_list = list(S)
        rng.shuffle(s_list)
        
        for p in s_list:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return S
                
            S_new = S - {p}
            
            if not S_new:
                # Cannot remove last point if grid is not empty
                continue
                
            if is_maximal(S_new):
                S = S_new
                improved = True
                break # Restart with new S
                
    return S

def _construct_impl():
    best_S = set()
    best_size = GRID_SIZE + 1
    rng = random.Random(42)
    
    # Run multiple restarts
    while time.time() - START_TIME < TIME_LIMIT - 10:
        # Greedy start
        S = greedy_construct(rng)
        
        # Reduce
        S = reduce_set(S, rng)
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = set(S)
            
            # If we found a very small set, we can try harder to reduce it
            if best_size <= 8:
                # Do some local search or more reductions
                for _ in range(10):
                    if time.time() - START_TIME > TIME_LIMIT - 5:
                        break
                    S_trial = reduce_set(best_S, rng)
                    if len(S_trial) < best_size:
                        best_size = len(S_trial)
                        best_S = S_trial

    if not best_S:
        # Fallback: just return a greedy set
        S = greedy_construct(rng)
        best_S = S
        
    return [POINTS[i] for i in best_S]

def construct():
    try:
        return _construct_impl()
    except Exception as e:
        # Return a simple greedy set as fallback
        S = set()
        blocked_mask = 0
        rng = random.Random(0)
        
        while True:
            s_mask = 0
            for s in S:
                s_mask |= (1 << s)
            addable_mask = ALL_POINTS_MASK & ~blocked_mask & ~s_mask
            if addable_mask == 0:
                break
            
            # Pick random addable
            m = addable_mask
            # Just pick the first one for simplicity in fallback
            lsb = m & -m
            p = lsb.bit_length() - 1
            
            S.add(p)
            new_blocked = blocked_mask
            for s in S:
                if s != p:
                    new_blocked |= BLOCK_MASK[p][s]
            blocked_mask = new_blocked
            
        return [POINTS[i] for i in S]

# Keep the original function name and signature
def construct():
    return _construct_impl()
