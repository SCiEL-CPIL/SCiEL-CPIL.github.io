import random
import time

N = 9
MAX_VERTEX = 1 << N

# EVOLVE-BLOCK-START

# Precompute neighbor masks
_neighborhood_masks = [0] * MAX_VERTEX
for _v in range(MAX_VERTEX):
    _mask = 0
    for _b in range(N):
        _mask |= 1 << (_v ^ (1 << _b))
    _neighborhood_masks[_v] = _mask

def _greedy_construct_optimized(start=0, time_limit=5.0):
    """
    Optimized greedy construction using bitmask for forbidden vertices.
    """
    path = [start]
    used = 1 << start
    blocked_mask = 0  # Neighbors of path[:-1]
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        cur = path[-1]
        # Candidates: neighbors of cur, not used, not blocked
        cand_mask = _neighborhood_masks[cur] & ~used & ~blocked_mask
        
        if cand_mask == 0:
            break
            
        candidates = []
        m = cand_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
            
        if not candidates:
            break
            
        best_cand = None
        best_score = -1
        
        for v in candidates:
            # If we move to v, new blocked_mask = blocked_mask | neighbors(cur)
            new_blocked = blocked_mask | _neighborhood_masks[cur]
            new_used = used | (1 << v)
            
            # Next candidates: neighbors of v, not in new_used, not in new_blocked
            next_cand_mask = _neighborhood_masks[v] & ~new_used & ~new_blocked
            score = next_cand_mask.bit_count()
            
            # Add random noise to break ties and explore different paths
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_cand = v
        
        if best_cand is None:
            break
            
        path.append(best_cand)
        used |= (1 << best_cand)
        blocked_mask = blocked_mask | _neighborhood_masks[cur]
        
    return path

def _backtracking_construct(start=0, max_depth=20, time_limit=5.0):
    """
    Limited depth-first search with backtracking.
    """
    best_path = []
    start_time = time.time()
    
    def dfs(path, used, blocked_mask, depth):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = path[:]
            
        if depth >= max_depth:
            return False
            
        if time.time() - start_time > time_limit:
            return False
            
        cur = path[-1]
        cand_mask = _neighborhood_masks[cur] & ~used & ~blocked_mask
        
        if cand_mask == 0:
            return False
            
        candidates = []
        m = cand_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
            
        random.shuffle(candidates)
        
        for v in candidates:
            new_used = used | (1 << v)
            new_blocked = blocked_mask | _neighborhood_masks[cur]
            path.append(v)
            if dfs(path, new_used, new_blocked, depth + 1):
                return True
            path.pop()
            
        return False
        
    dfs([start], 1 << start, 0, 0)
    return best_path

def construct():
    """
    Construct the longest snake in Q_9 found by various strategies.
    Returns a list of distinct integers representing the vertices in order.
    """
    best_path = []
    start_time = time.time()
    total_time_limit = 110.0  # Leave 10s buffer
    
    random.seed(42)
    
    # Strategy 1: Optimized Greedy with various starts
    starts = [0, 1, 2, 511, 510, 255, 256, 128, 64, 32, 16, 8, 4, 2]
    for _ in range(10):
        starts.append(random.randint(0, MAX_VERTEX - 1))
        
    for start in starts:
        if time.time() - start_time > total_time_limit:
            break
        path = _greedy_construct_optimized(start, time_limit=2.0)
        if len(path) > len(best_path):
            best_path = path
            
    # Strategy 2: Backtracking with limited depth
    bt_starts = [0, 1, 2, 511, 255, 256]
    for start in bt_starts:
        if time.time() - start_time > total_time_limit:
            break
        path = _backtracking_construct(start, max_depth=30, time_limit=5.0)
        if len(path) > len(best_path):
            best_path = path
            
    # Strategy 3: More greedy runs with randomization
    runs = 0
    while time.time() - start_time < total_time_limit:
        start = random.randint(0, MAX_VERTEX - 1)
        path = _greedy_construct_optimized(start, time_limit=1.0)
        if len(path) > len(best_path):
            best_path = path
        runs += 1
        if runs > 2000:
            break

    return best_path

# EVOLVE-BLOCK-END
