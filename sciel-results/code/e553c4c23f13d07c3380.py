"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TIME_LIMIT = 110.0  # safety margin below 120s evaluator limit
SEED_LENGTH = 104

# EVOLVE-BLOCK-START
def _is_valid_extension(v, path, used, exclusion):
    """Check if v can be appended to path.
    - v not in used
    - v differs from all non-consecutive vertices by >= 2 bits
    We maintain an exclusion set: vertices that are at Hamming distance 1 from any
    non-terminal vertex in the path (i.e., would violate induced property if added).
    Actually, the constraint is: for any non-consecutive pair, Hamming distance >= 2.
    Consecutive vertices differ by exactly 1 bit.
    
    So when we add v after cur=path[-1], we need:
    1. v not in used
    2. For all p in path[:-1] (all except the last, since v is consecutive to path[-1]):
       popcount(v ^ p) != 1  (must be >= 2, and since v != p, it's >= 2)
    
    We can maintain a set 'exclusion' of all vertices u such that there exists some
    p in path[:-1] with popcount(u ^ p) == 1. Then v is valid iff v not in used and v not in exclusion.
    
    When we extend path by v, the new "non-terminal" vertices become path[:-1] + [path[-1]]
    (the old terminal becomes non-terminal). So we need to add to exclusion all neighbors
    of the old terminal (path[-1]) that aren't already there.
    """
    if v in used or v in exclusion:
        return False
    return True

def _greedy_search(rng, time_budget):
    """Run one greedy snake construction with randomization."""
    path = [0]
    used = {0}
    # exclusion: set of vertices at Hamming distance 1 from any non-terminal vertex
    # Initially path = [0], no non-terminal vertices, so exclusion = empty
    exclusion = set()
    
    while True:
        cur = path[-1]
        # Find valid next vertices
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in exclusion:
                candidates.append(v)
        
        if not candidates:
            return path
        
        # Pick one candidate (randomized)
        v = rng.choice(candidates)
        path.append(v)
        used.add(v)
        
        # Update exclusion: the old terminal (cur) becomes non-terminal
        # Add all neighbors of cur to exclusion
        for b in range(N):
            nb = cur ^ (1 << b)
            exclusion.add(nb)
        
        # Time check
        if time.time() > time_budget:
            return path

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = None
    best_len = 0
    
    rng = random.Random(42)
    
    # First run: deterministic greedy (original) to get baseline
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
            break
        path.append(nxt)
        used.add(nxt)
    best_path = path
    best_len = len(path)
    
    # Now run randomized greedy searches
    # Each search is fast; we can run many
    search_count = 0
    while time.time() < deadline - 2.0:  # 2s safety margin
        p = _greedy_search(rng, deadline - 1.0)
        search_count += 1
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    return best_path
# EVOLVE-BLOCK-END
