import time
import random

N = 9
MAX_TIME = 115.0  # Safety margin before 120s/150s kill

# Precompute bit counts for all possible XOR results (0 to 2^9 - 1)
_BIT_COUNT = [i.bit_count() for i in range(1 << N)]
# Precompute neighbors for all 512 vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(1 << N)]
# Precompute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
_NEIGHBOR_MASK = [0] * (1 << N)
for v in range(1 << N):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASK[v] = mask

def _greedy_search(seed_offset=0, depth_limit=1000):
    """
    Optimized Greedy search using Forbidden Mask (bitmask).
    """
    rng = random.Random(seed_offset)
    path = [0]
    # Forbidden mask as a bitmask: bit i is set if vertex i is forbidden
    forbidden = 1  # vertex 0 is forbidden
    
    current = 0
    
    while len(path) < depth_limit:
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            break
            
        # Lookahead: score each neighbor by how many valid neighbors it has
        scored_neighbors = []
        for v in valid_neighbors:
            # If we pick v, the new path is path + [v].
            # The new forbidden set for the NEXT step will be:
            # Forbidden_new = Forbidden_old | _NEIGHBOR_MASK[current] | (1 << v)
            
            # We want to count u in Neighbors(v) such that u is NOT in Forbidden_new.
            # Valid candidates: bits in _NEIGHBOR_MASK[v] that are NOT in Forbidden_new
            forbidden_new = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            valid_mask = _NEIGHBOR_MASK[v] & ~forbidden_new
            count = valid_mask.bit_count()
            scored_neighbors.append((count, v))
            
        if not scored_neighbors:
            break

        # Sort by count descending, pick randomly from top ties
        max_count = max(c for c, v in scored_neighbors)
        candidates = [v for c, v in scored_neighbors if c == max_count]
        next_v = rng.choice(candidates)
        
        path.append(next_v)
        
        # Update Forbidden Mask
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << next_v)
        
        current = next_v
        
    return path

def _beam_search(seed_offset=0, beam_width=20, depth_limit=200):
    """
    Optimized Beam Search using Forbidden Mask (bitmask).
    """
    rng = random.Random(seed_offset)
    # Beam entry: (path, forbidden_mask, current)
    
    beams = [([0], 1, 0)]  # forbidden=1 means vertex 0 is forbidden
    
    for depth in range(depth_limit):
        new_beams = []
        for path, forbidden, current in beams:
            for v in _NEIGHBORS[current]:
                if not (forbidden >> v) & 1:
                    # Create new state
                    new_path = path + [v]
                    new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
                    
                    new_beams.append((new_path, new_forbidden, v))
                    
        if not new_beams:
            break
            
        # Score and prune
        # Score: number of valid extensions (lookahead 1)
        scored_beams = []
        for path, forbidden, current in new_beams:
            # Valid neighbors of current: bits in _NEIGHBOR_MASK[current] not in forbidden
            valid_mask = _NEIGHBOR_MASK[current] & ~forbidden
            count = valid_mask.bit_count()
            scored_beams.append((count, path, forbidden, current))
            
        # Sort by count
        scored_beams.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top beam_width
        beams = [(p, f, c) for _, p, f, c in scored_beams[:beam_width]]
        
        if not beams:
            break
            
    if not beams:
        return [0]
    
    best = max(beams, key=lambda x: len(x[0]))
    return best[0]

def construct():
    start_time = time.time()
    best_path = [0]
    best_len = 1
    
    # Initial deterministic greedy
    path = [0]
    forbidden = 1  # vertex 0 forbidden
    current = 0
    while True:
        nxt = None
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        # Update forbidden
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << nxt)
        current = nxt
        
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
        if i % 5 == 0 and time.time() - start_time < MAX_TIME - 10:
            bp = _beam_search(i, beam_width=20)
            if len(bp) > best_len:
                best_path = bp
                best_len = len(bp)
        
    return best_path
