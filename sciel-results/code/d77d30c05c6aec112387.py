"""Hybrid Greedy + Local Search construction for the ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import heapq
import itertools
import numpy as np
import time
import random

# Precomputation cache for the specific task
_Q = 3
_N = 8
_RADIUS = 1
_TOTAL = _Q ** _N
_WEIGHT = _Q ** np.arange(_N - 1, -1, -1)
_WORDS = None
_NEIGHBORS = None

def _init_precomputation(q, n, radius):
    global _WORDS, _NEIGHBORS, _TOTAL, _Q, _N, _WEIGHT
    if q != 3 or n != 8 or radius != 1:
        return False
    
    _Q = q
    _N = n
    _TOTAL = q ** n
    _WEIGHT = q ** np.arange(n - 1, -1, -1)
    
    # Generate all words
    _WORDS = np.empty((_TOTAL, n), dtype=np.int64)
    for i in range(_TOTAL):
        v = i
        for j in range(n - 1, -1, -1):
            _WORDS[i, j] = v % q
            v //= q
            
    # Precompute neighbors
    _NEIGHBORS = []
    for i in range(_TOTAL):
        w = _WORDS[i].copy()
        nb = [i]
        for j in range(n):
            old = w[j]
            for a in range(q):
                if a == old:
                    continue
                w[j] = a
                nb.append(int(w.dot(_WEIGHT)))
                w[j] = old
        _NEIGHBORS.append(nb)
    return True

def _greedy_init(q, n, radius):
    """Fast greedy initialization."""
    if not _init_precomputation(q, n, radius):
        # Fallback for other params
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    uncovered = np.ones(_TOTAL, dtype=bool)
    remaining = _TOTAL
    codewords = []
    
    covered_count = np.array([len(nb) for nb in _NEIGHBORS], dtype=np.int64)
    heap = [(-covered_count[i], i) for i in range(_TOTAL)]
    heapq.heapify(heap)
    
    while remaining > 0:
        while True:
            neg_key, i = heapq.heappop(heap)
            if -neg_key == covered_count[i]:
                break
        codewords.append(_WORDS[i].tolist())
        for nb in _NEIGHBORS[i]:
            if uncovered[nb]:
                uncovered[nb] = False
                remaining -= 1
                for j in _NEIGHBORS[nb]:
                    covered_count[j] -= 1
                    if remaining > 0:
                        heapq.heappush(heap, (-covered_count[j], j))
                        
    return codewords

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    _init_precomputation(q, n, radius)
    
    # 1. Greedy Initialization
    codewords = _greedy_init(q, n, radius)
    C = set(tuple(c) for c in codewords)
    
    # 2. Local Search: Remove redundant codewords
    cover_count = np.zeros(_TOTAL, dtype=np.int64)
    C_dict = {}
    for c in C:
        idx = int(np.array(c, dtype=np.int64).dot(_WEIGHT))
        C_dict[c] = idx
        for nb in _NEIGHBORS[idx]:
            cover_count[nb] += 1
            
    start_time = time.time()
    timeout = 110.0
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        # Find a removable codeword
        c_to_remove = None
        c_idx = None
        for c in list(C):
            idx = C_dict[c]
            nb_list = _NEIGHBORS[idx]
            if np.min(cover_count[nb_list]) > 1:
                c_to_remove = c
                c_idx = idx
                break
        
        if c_to_remove is not None:
            C.remove(c_to_remove)
            del C_dict[c_to_remove]
            for nb in _NEIGHBORS[c_idx]:
                cover_count[nb] -= 1
            improved = True
            continue
        else:
            # No single removal possible.
            # Try a random swap to escape local optimum?
            # For now, break to ensure validity and speed.
            break

    return [list(c) for c in C]
# EVOLVE-BLOCK-END
