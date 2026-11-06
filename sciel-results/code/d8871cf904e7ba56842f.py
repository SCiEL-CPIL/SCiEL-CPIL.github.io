"""Snake-in-the-box in Q_9: maximize induced path length."""
import time
import random

N = 9
V = 1 << N  # 512 vertices

# Precompute neighbors for each vertex
NEIGHBORS = [[] for _ in range(V)]
for v in range(V):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# Precompute bit_count for XOR differences up to V
# Actually we can use int.bit_count() which is fast in Python 3.8+


def _is_valid_extension(path, used, candidate):
    """Check if adding candidate to path keeps it an induced path.
    candidate must differ from path[-1] by exactly 1 bit (guaranteed by caller).
    Must differ from all other path vertices by >= 2 bits.
    """
    for i in range(len(path) - 1):
        if (candidate ^ path[i]).bit_count() == 1:
            return False
    return True


def _greedy_construct(start=0, bit_order=None):
    """Greedy snake construction with given bit ordering."""
    if bit_order is None:
        bit_order = list(range(N))
    path = [start]
    used = {start}
    while True:
        cur = path[-1]
        nxt = None
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _is_valid_extension(path, used, v):
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    return path


def _greedy_with_random_bits(start=0, seed=None):
    """Greedy with randomized bit order."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    bit_order = list(range(N))
    rng.shuffle(bit_order)
    return _greedy_construct(start, bit_order)


def _backtrack_extend(path, used, depth_limit=0, deadline=None):
    """Try to extend path further using limited backtracking.
    Returns the longest path found (extending the input path).
    """
    best_path = list(path)
    best_len = len(path)
    
    def _extend(cur_path, cur_used):
        nonlocal best_path, best_len
        if deadline and time.time() > deadline:
            return
        cur = cur_path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in cur_used:
                continue
            if _is_valid_extension(cur_path, cur_used, v):
                candidates.append(v)
        
        if not candidates:
            return
        
        # Try each candidate
        for v in candidates:
            new_path = cur_path + [v]
            new_used = cur_used | {v}
            if len(new_path) > best_len:
                best_path = list(new_path)
                best_len = len(new_path)
            _extend(new_path, new_used)
    
    _extend(path, used)
    return best_path


def _local_search(path, deadline=None):
    """Try to improve path by inserting vertices or rearranging.
    For now, try to extend from both ends.
    """
    if deadline and time.time() > deadline:
        return path
    
    best = list(path)
    
    # Try extending from the end
    improved = True
    while improved:
        improved = False
        cur = best[-1]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in set(best):
                continue
            if _is_valid_extension(best, set(best), v):
                best.append(v)
                improved = True
                break
    
    # Try extending from the beginning
    improved = True
    while improved:
        improved = False
        cur = best[0]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in set(best):
                continue
            # Check induced: v must differ from all except best[1] by >= 2 bits
            ok = True
            for i in range(1, len(best)):
                if (v ^ best[i]).bit_count() == 1:
                    ok = False
                    break
            if ok:
                best.insert(0, v)
                improved = True
                break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_path = []
    best_len = 0
    
    # Phase 1: Try multiple greedy constructions with different bit orders and starts
    # Use a variety of starting vertices and bit orderings
    seen_starts = set()
    
    # First, the standard greedy (deterministic)
    p = _greedy_construct(0)
    if len(p) > best_len:
        best_path = p
        best_len = len(p)
    
    # Try different bit orderings
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        start = rng.randint(0, V - 1)
        p = _greedy_construct(start, bit_order)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
    
    # Phase 2: Local search / extension on the best path found
    if time.time() < deadline - 5.0:
        best_path = _local_search(best_path, deadline)
        best_len = len(best_path)
    
    # Phase 3: Try backtracking extension on the best path
    if time.time() < deadline - 10.0:
        used = set(best_path)
        extended = _backtrack_extend(best_path, used, deadline=deadline - 2.0)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
    
    # Phase 4: More randomized greedy tries if time remains
    if time.time() < deadline - 5.0:
        for trial in range(500):
            if time.time() > deadline:
                break
            bit_order = list(range(N))
            rng.shuffle(bit_order)
            start = rng.randint(0, V - 1)
            p = _greedy_construct(start, bit_order)
            if len(p) > best_len:
                best_path = p
                best_len = len(p)
                # Quick local search on new best
                if time.time() < deadline - 5.0:
                    best_path = _local_search(best_path, deadline)
                    best_len = len(best_path)
    
    # Final check: ensure validity
    # Verify the path is a valid induced path
    assert len(best_path) >= 2, "Path too short"
    used_set = set(best_path)
    assert len(used_set) == len(best_path), "Duplicate vertices in path"
    for i in range(len(best_path) - 1):
        assert (best_path[i] ^ best_path[i + 1]).bit_count() == 1, f"Consecutive vertices differ by more than 1 bit at index {i}"
    for i in range(len(best_path)):
        for j in range(i + 2, len(best_path)):
            assert (best_path[i] ^ best_path[j]).bit_count() >= 2, f"Non-consecutive vertices differ by 1 bit at indices {i} and {j}"
    
    return best_path
