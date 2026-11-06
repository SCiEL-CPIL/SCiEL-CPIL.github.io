"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START
def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:  # not a power of 2
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _greedy_extend(path, used, bit_order, deadline):
    """Try to extend path greedily using Warnsdorff's rule heuristic."""
    cur = path[-1]
    candidates = []
    
    for b in bit_order:
        v = cur ^ (1 << b)
        if v in used:
            continue
        # Check against all previous vertices (except the one we're extending from)
        ok = True
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                ok = False
                break
        if ok:
            candidates.append(v)
    
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Warnsdorff's rule: choose the candidate with the fewest valid next moves
    # This helps avoid dead ends by forcing the path into tight spots early.
    best_v = None
    min_score = float('inf')
    
    for v in candidates:
        score = 0
        for b2 in range(N):
            v2 = v ^ (1 << b2)
            if v2 in used or v2 == cur:
                continue
            # Check v2 against all path vertices except cur
            ok2 = True
            for p in path[:-1]:
                if (v2 ^ p).bit_count() == 1:
                    ok2 = False
                    break
            if ok2:
                score += 1
        if score < min_score:
            min_score = score
            best_v = v
            
    return best_v

def _local_search(path, used, deadline):
    """Try to improve the snake by extending ends.
    Returns improved path if found, else original."""
    if time.time() > deadline:
        return path
    
    best = path[:]
    best_len = len(path)
    
    # Try extending both ends multiple times
    max_iters = 500
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to extend the end
        cur = path[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                path.append(v)
                used.add(v)
                extended = True
                break
        
        # Try to extend the beginning
        cur = path[0]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for p in path[1:]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                path.insert(0, v)
                used.add(v)
                extended = True
                break
        
        if not extended:
            break
            
        if len(path) > best_len:
            best = path[:]
            best_len = len(path)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_path = []
    best_len = 0
    
    # Multiple restarts with different random bit orders
    random.seed(42)
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Generate a random bit order for this attempt
        bit_order = list(range(N))
        random.shuffle(bit_order)
        
        # Start with a random vertex
        start_vertex = random.randint(0, TOTAL_VERTICES - 1)
        path = [start_vertex]
        used = {start_vertex}
        
        # Greedy extension with Warnsdorff heuristic
        while time.time() < deadline:
            nxt = _greedy_extend(path, used, bit_order, deadline)
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        
        # Local search improvement
        if time.time() < deadline - 2:
            path = _local_search(path, used, deadline)
        
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
            
        # Safety break if we are very close to deadline
        if time.time() > deadline - 5:
            break
    
    # Final validation
    if best_path and _is_valid_snake(best_path):
        return best_path
    
    # Fallback: return the greedy result
    path, used = [0], {0}
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)
# EVOLVE-BLOCK-END
