import time
import random
import bisect

def _compute_sums(candidate):
    """Compute all nonempty consecutive-block sums for a sorted candidate."""
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            sums.add(s)
    return sums

def _is_valid_candidate(candidate):
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build(order):
    selected = []
    for x in order:
        candidate = selected + [x]
        candidate.sort()
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected

def _try_insert(selected, x):
    """Try to insert x into sorted selected. Returns new_selected or None."""
    pos = bisect.bisect_left(selected, x)
    if pos < len(selected) and selected[pos] == x:
        return None
    new_selected = selected[:pos] + [x] + selected[pos:]
    if _is_valid_candidate(new_selected):
        return new_selected
    return None

def _local_search_insert(selected, available, deadline):
    """Try to improve sequence by inserting available numbers at any position."""
    current = selected[:]
    available_set = set(available)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        avail_list = list(available_set)
        random.shuffle(avail_list)
        for x in avail_list:
            if x in current:
                continue
            new_seq = _try_insert(current, x)
            if new_seq is not None:
                current = new_seq
                available_set.discard(x)
                improved = True
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    
    # 1. Deterministic Baselines
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    # 2. Random Restarts with Insertion-Based Local Search
    rng = random.Random(42)
    base_range = list(range(1, 257))
    
    while time.time() < deadline:
        # Try different random orders
        rng.shuffle(base_range)
        seq = _greedy_build(base_range)
        
        if len(seq) > len(best_seq):
            best_seq = seq
            
        # Local search: try to insert more elements into the greedy result
        if len(seq) < 256 and time.time() < deadline - 1:
            available = set(range(1, 257)) - set(seq)
            improved_seq = _local_search_insert(seq, available, deadline)
            if len(improved_seq) > len(best_seq):
                best_seq = improved_seq
        
        # Also periodically try local search on the current best
        if len(best_seq) < 256 and time.time() < deadline - 1:
            available = set(range(1, 257)) - set(best_seq)
            improved_seq = _local_search_insert(best_seq, available, deadline)
            if len(improved_seq) > len(best_seq):
                best_seq = improved_seq
        
        # Early exit for very long sequences
        if len(best_seq) >= 22:
            break
    
    return best_seq
