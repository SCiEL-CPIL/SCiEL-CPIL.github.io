import time
import random

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

def construct():
    start_time = time.time()
    # Safety margin: 55s limit, 60s hard limit, 90s kill
    deadline = start_time + 55.0
    
    # 1. Deterministic Baselines
    best_seq = []
    
    # Ascending
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    # Descending
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    # 2. Random Restarts
    # Use a fixed seed for reproducibility
    rng = random.Random(42)
    base_range = list(range(1, 257))
    
    while time.time() < deadline:
        rng.shuffle(base_range)
        seq = _greedy_build(base_range)
        if len(seq) > len(best_seq):
            best_seq = seq
            
    return best_seq
# EVOLVE-BLOCK-END
