import time
import random

N = 9
MAX_TIME = 115.0  # Safety margin before 120s/150s kill

# Precompute bit counts for all possible XOR results (0 to 2^9 - 1)
_BIT_COUNT = [i.bit_count() for i in range(1 << N)]
# Precompute neighbors for all 512 vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(1 << N)]

def _can_extend(v, path, used):
    """Check if v can be appended to path."""
    if v in used:
        return False
    # Check non-consecutive vertices: all in path except the last one (current)
    # v is a neighbor of path[-1] (current), so dist(v, path[-1]) == 1, which is fine for consecutive.
    # But we must ensure dist(v, p) >= 2 for all p in path[:-1].
    for p in path[:-1]:
        if _BIT_COUNT[v ^ p] == 1:
            return False
    return True

def _greedy_search(seed_offset=0, depth_limit=1000):
    """
    Greedy search with lookahead.
    Returns a list of vertices forming a valid snake.
    """
    rng = random.Random(seed_offset)
    path = [0]
    used = {0}
    current = 0
    
    while len(path) < depth_limit:
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if _can_extend(v, path, used):
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            break
            
        # Lookahead: score each neighbor by how many valid neighbors it has
        scored_neighbors = []
        for v in valid_neighbors:
            # Count valid neighbors of v, assuming v is added to path
            # New path: path + [v]
            # New used: used + {v}
            # We need to count u in neighbors(v) such that:
            # 1. u not in used and u != v (v is in used now)
            # 2. dist(u, p) >= 2 for all p in path (since path is (path + [v])[:-1])
            
            count = 0
            for u in _NEIGHBORS[v]:
                if u in used:
                    continue
                # Check against all p in path
                valid = True
                for p in path:
                    if _BIT_COUNT[u ^ p] == 1:
                        valid = False
                        break
                if valid:
                    count += 1
            scored_neighbors.append((count, v))
            
        # Sort by count descending, pick randomly from top ties
        # To avoid full sort, find max count and collect ties
        max_count = max(c for c, v in scored_neighbors)
        candidates = [v for c, v in scored_neighbors if c == max_count]
        next_v = rng.choice(candidates)
        
        path.append(next_v)
        used.add(next_v)
        current = next_v
        
    return path

def _beam_search(seed_offset=0, beam_width=5, depth_limit=200):
    """
    Beam search to find longer snakes.
    """
    rng = random.Random(seed_offset)
    # Each beam entry is (path, used_set)
    beams = [([0], {0})]
    
    for depth in range(depth_limit):
        new_beams = []
        for path, used in beams:
            current = path[-1]
            for v in _NEIGHBORS[current]:
                if _can_extend(v, path, used):
                    new_path = path + [v]
                    new_used = used | {v}
                    new_beams.append((new_path, new_used))
                    
        if not new_beams:
            break
            
        # Score each new beam by its length (all same length) and then by potential
        # Sort by length descending, then by random tiebreaker
        # Actually all have same length at this depth, so we need another criterion
        # Let's use the number of valid extensions as a heuristic
        scored_beams = []
        for path, used in new_beams:
            current = path[-1]
            count = 0
            for v in _NEIGHBORS[current]:
                if v not in used:
                    # Quick check: can v extend?
                    can = True
                    for p in path[:-1]:
                        if _BIT_COUNT[v ^ p] == 1:
                            can = False
                            break
                    if can:
                        count += 1
            scored_beams.append((count, len(path), path, used))
            
        # Sort by count descending, then by length descending
        scored_beams.sort(key=lambda x: (x[0], x[1]), reverse=True)
        beams = [(path, used) for count, length, path, used in scored_beams[:beam_width]]
        
        if not beams:
            break
            
        # Keep only the longest paths
        max_len = max(len(p) for p, u in beams)
        beams = [(p, u) for p, u in beams if len(p) == max_len]
        
        if not beams:
            break
            
    if not beams:
        return [0]
    
    # Return the longest beam
    best = max(beams, key=lambda x: len(x[0]))
    return best[0]

def construct():
    start_time = time.time()
    best_path = [0]
    best_len = 1
    
    # Initial deterministic greedy (lowest bit)
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for v in _NEIGHBORS[cur]:
            if v not in used and all(_BIT_COUNT[v ^ p] != 1 for p in path[:-1]):
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    best_path = path
    best_len = len(best_path)
    
    # Randomized greedy searches
    i = 0
    while time.time() - start_time < MAX_TIME:
        p = _greedy_search(i)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        i += 1
        
        # Occasionally try beam search
        if i % 10 == 0 and time.time() - start_time < MAX_TIME - 5:
            bp = _beam_search(i, beam_width=3)
            if len(bp) > best_len:
                best_path = bp
                best_len = len(bp)
        
    return best_path
