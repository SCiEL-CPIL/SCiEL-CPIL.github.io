"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9

# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 150.0  # wall-clock seconds before evaluator kills us

def _greedy_randomized(start_vertex, bit_order, excluded, used, deadline):
    """
    Greedy snake extension with O(1) exclusion-set maintenance.
    Returns (path, excluded, used).
    """
    path = [start_vertex]
    cur = start_vertex
    # Precompute neighbors for speed
    while True:
        if time.time() > deadline:
            break
        found = False
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used or v in excluded:
                continue
            # Accept this vertex
            path.append(v)
            used.add(v)
            # Exclude all neighbors of v (they'd be at distance 1 from v,
            # and v is not adjacent to the next vertex we'll add)
            for nb in range(N):
                excl = v ^ (1 << nb)
                if excl != cur:  # don't exclude the vertex we came from
                    excluded.add(excl)
            cur = v
            found = True
            break
        if not found:
            break
    return path, excluded, used

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - 5.0  # 5s safety margin
    best_path = None
    best_len = 0
    
    rng = random.Random(42)
    
    # First: run the original deterministic greedy to establish a baseline
    # using the fast exclusion-set method
    bit_order = list(range(N))
    excluded = set()
    used = set()
    path, excluded, used = _greedy_randomized(0, bit_order, excluded, used, deadline)
    best_path = path
    best_len = len(path)
    
    # Now do many randomized restarts
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Random start vertex
        start = rng.randint(0, (1 << N) - 1)
        # Random bit order
        bo = list(range(N))
        rng.shuffle(bo)
        
        excluded = set()
        used = set()
        path, excluded, used = _greedy_randomized(start, bo, excluded, used, deadline)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    return best_path
# EVOLVE-BLOCK-END

