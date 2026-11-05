import time
import random

N = 9
MAX_TIME = 115.0  # Safety margin before 120s/150s kill

def _is_valid_extension(v, path, used):
    if v in used:
        return False
    # Check non-consecutive vertices
    # path[-1] is current, so check path[:-1]
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def _count_valid_neighbors(v, path, used):
    count = 0
    for b in range(N):
        u = v ^ (1 << b)
        if _is_valid_extension(u, path, used):
            count += 1
    return count

def _greedy_search(seed_offset=0):
    random.seed(seed_offset)
    path = [0]
    used = {0}
    current = 0
    
    while True:
        valid_neighbors = []
        for b in range(N):
            v = current ^ (1 << b)
            if _is_valid_extension(v, path, used):
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            break
            
        # Lookahead: score each neighbor by how many options it leaves
        scored_neighbors = []
        for v in valid_neighbors:
            # Temporarily add v to used/path to check neighbors? 
            # No, just count valid neighbors of v given current path+used+v
            # We can approximate by counting valid neighbors of v in current state, 
            # but we must exclude 'current' which is already in used.
            # Actually, _is_valid_extension checks against path[:-1] and used.
            # If we move to v, the new current is v.
            # The new valid neighbors of v are those u where u is not in used+{v} and dist(u, p)>=2 for p in path+[v] (excluding v itself? no, path is the history).
            # Let's just count valid neighbors of v in the current state, 
            # but we must ensure we don't count 'current' again (it's in used).
            
            # Efficient check:
            # For each neighbor u of v:
            # 1. u not in used and u != v
            # 2. dist(u, p) >= 2 for all p in path[:-1] (which is path before current)
            #    AND dist(u, current) >= 2 (since current becomes non-consecutive to u? No, u is neighbor of v, v is neighbor of current. 
            #    u and current are distance 2 or 0. If u==current, dist 0. If u!=current, dist 2. 
            #    So u is valid wrt current if u != current.
            #    But we also need dist(u, p) >= 2 for all p in path[:-2] (ancestors of current).
            
            # This is getting complicated to optimize perfectly. 
            # Let's just use the simple count: how many valid moves from v if we were there?
            # We can simulate adding v to path and used, then count, then revert? 
            # Or just count valid neighbors of v that are not in used and not current.
            # The constraint "dist(u, p) >= 2 for all p in path[:-1]" is the hard part.
            
            # Let's stick to a simpler heuristic: 
            # Prefer bits that are "less used" or just random among valid ones?
            # The lookahead of depth 1 is: count valid neighbors of v.
            # Let's implement a helper that counts valid neighbors of v given the current path/used state, 
            # assuming v is the new current.
            
            # Valid neighbors of v are u such that:
            # 1. u not in used
            # 2. u != v
            # 3. (u ^ p).bit_count() >= 2 for all p in path[:-1] 
            #    (Note: path[:-1] contains current. So dist(u, current) must be >= 2. 
            #     Since u is neighbor of v and v is neighbor of current, dist(u, current) is 0 or 2.
            #     So we just need u != current.)
            # 4. (u ^ p).bit_count() >= 2 for all p in path[:-2] (ancestors).
            
            # So for each u in neighbors(v):
            # if u in used or u == current: continue
            # check dist(u, p) for p in path[:-2]
            
            count = 0
            for b2 in range(N):
                u = v ^ (1 << b2)
                if u in used or u == current:
                    continue
                # Check ancestors
                valid = True
                for p in path[:-2]: # path[:-1] is current, so ancestors are path[:-2]
                    if (u ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    count += 1
            scored_neighbors.append((count, v))
            
        # Sort by count descending, then randomize ties
        # To avoid sorting overhead, just pick max with random tie-breaker
        max_count = max(c for c, v in scored_neighbors)
        candidates = [v for c, v in scored_neighbors if c == max_count]
        next_v = random.choice(candidates)
        
        path.append(next_v)
        used.add(next_v)
        current = next_v
        
    return path

def construct():
    start_time = time.time()
    best_path = []
    
    # First, run the deterministic seed greedy to have a baseline
    # The seed greedy in the prompt uses lowest bit flip.
    # Let's run a standard greedy first to get a good initial best.
    # We can just run _greedy_search with a fixed seed or the original logic.
    # Actually, _greedy_search with random seed is fine.
    # Let's run one quick deterministic greedy to ensure we have a valid path.
    
    # Original greedy logic for baseline
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    best_path = path
    
    # Now run randomized searches
    i = 0
    while time.time() - start_time < MAX_TIME:
        p = _greedy_search(i)
        if len(p) > len(best_path):
            best_path = p
            # Optional: print or log, but not needed
        i += 1
        # If we hit a very long path, we might want to slow down or stop?
        # No, keep trying to improve.
        
    return best_path
