"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _max_sidon_subset_size(arr):
    """Find the largest subset of arr where all pair sums (i<=j) are unique."""
    n = len(arr)
    # Try subsets from largest to smallest
    for size in range(n, 0, -1):
        found = False
        for subset in itertools.combinations(range(n), size):
            sums = set()
            ok = True
            for i in range(size):
                for j in range(i, size):
                    s = arr[subset[i]] + arr[subset[j]]
                    if s in sums:
                        ok = False
                        break
                    sums.add(s)
                if not ok:
                    break
            if ok:
                found = True
                break
        if found:
            return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    # Try several candidate sets designed to maximize additive collisions
    candidates = [
        # Arithmetic progression - many sum collisions
        list(range(0, 24, 2)),  # 0,2,4,...,22
        # Two clusters
        [0, 1, 2, 3, 4, 5, 122, 123, 124, 125, 126, 127],
        # Modulo structure
        [i * 10 + 5 for i in range(12)],  # 5,15,25,...,115
        # Dense cluster
        list(range(12)),  # 0-11
        # Sparse with structure
        [0, 1, 3, 7, 15, 31, 63, 94, 95, 97, 101, 111],
        # All even in range
        list(range(0, 128, 11))[:12],
        # Mixed clusters
        [0, 1, 2, 10, 11, 12, 116, 117, 118, 126, 127, 128],
        # Fibonacci-like
        [0, 1, 2, 4, 8, 16, 32, 64, 65, 66, 68, 72],
    ]
    
    best_set = None
    best_size = 99
    
    for cand in candidates:
        if time.time() > deadline:
            break
        # Validate
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        size = _max_sidon_subset_size(cand)
        if size < best_size:
            best_size = size
            best_set = cand
            if best_size <= 4:  # can't do much better
                break
    
    # Try more random structured sets if time permits
    if time.time() < deadline - 10 and best_size > 5:
        import random
        random.seed(42)
        for _ in range(500):
            if time.time() > deadline:
                break
            # Random set with clustering
            cluster1 = random.sample(range(0, 40), 6)
            cluster2 = random.sample(range(88, 129), 6)
            cand = sorted(cluster1 + cluster2)
            if len(set(cand)) != 12:
                continue
            size = _max_sidon_subset_size(cand)
            if size < best_size:
                best_size = size
                best_set = cand
                if best_size <= 4:
                    break
    
    return best_set if best_set else list(range(12))
# EVOLVE-BLOCK-END
