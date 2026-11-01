"""Perturbation-based Local Search for ternary covering code K_3(8,1)."""
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
_COVERERS = None # Reverse neighbors: word -> list of codewords covering it

def _init_precomputation(q, n, radius):
    global _WORDS, _NEIGHBORS, _COVERERS, _TOTAL, _Q, _N, _WEIGHT
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
    # Initialize coverers as empty lists
    _COVERERS = [[] for _ in range(_TOTAL)]
    
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
        # Add i to coverers of all its neighbors
        for nb_idx in nb:
            _COVERERS[nb_idx].append(i)
            
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
    C_set = set(tuple(c) for c in codewords)
    
    # Data structures for fast updates
    cover_count = np.zeros(_TOTAL, dtype=np.int64)
    C_dict = {} # codeword tuple -> index
    for c in C_set:
        idx = int(np.array(c, dtype=np.int64).dot(_WEIGHT))
        C_dict[c] = idx
        for nb in _NEIGHBORS[idx]:
            cover_count[nb] += 1
            
    best_size = len(C_set)
    best_C = set(C_set)
    
    start_time = time.time()
    timeout = 110.0
    
    # 2. Local Search Loop
    while (time.time() - start_time) < timeout:
        # Phase 1: Exhaustive Removal (find local minimum)
        changed = True
        while changed:
            changed = False
            # Shuffle order to avoid bias
            items = list(C_set)
            random.shuffle(items)
            for c in items:
                if c not in C_set: continue
                idx = C_dict[c]
                nb_list = _NEIGHBORS[idx]
                # Check if any neighbor has cover count 1
                # If min(cover_count[nb_list]) > 1, we can remove
                if np.min(cover_count[nb_list]) > 1:
                    # Remove c
                    C_set.remove(c)
                    del C_dict[c]
                    for nb in nb_list:
                        cover_count[nb] -= 1
                    changed = True
                    if len(C_set) < best_size:
                        best_size = len(C_set)
                        best_C = set(C_set)
        
        # Phase 2: Perturbation (if we have time)
        if len(C_set) == 0: break
        if (time.time() - start_time) > timeout - 5.0: break
        
        # Randomly remove a codeword to create a hole, then repair
        # This allows escaping local minima where no single removal is possible
        # but a swap might be.
        
        # Pick a random codeword to remove
        c_remove = random.choice(list(C_set))
        idx_remove = C_dict[c_remove]
        
        # Remove it (state becomes invalid)
        C_set.remove(c_remove)
        del C_dict[c_remove]
        
        # Identify newly uncovered words
        uncovered = np.zeros(_TOTAL, dtype=bool)
        for nb in _NEIGHBORS[idx_remove]:
            cover_count[nb] -= 1
            if cover_count[nb] == 0:
                uncovered[nb] = True
                
        # Repair: Greedily add codewords to cover 'uncovered'
        # We use the COVERERS precomputation to find candidates efficiently
        while np.any(uncovered):
            # Pick a random uncovered word
            u = np.random.choice(np.where(uncovered)[0])
            # Pick a random coverer for u that is NOT in C_set
            # Note: Coverers might include words already in C_set?
            # If a coverer is in C_set, its removal would have decreased cover_count.
            # Since u is uncovered, no codeword in C_set covers it.
            # So all coverers of u are NOT in C_set.
            candidates = _COVERERS[u]
            if not candidates:
                # Should not happen for radius 1
                break
            c_add = random.choice(candidates)
            
            # Add c_add to C_set
            C_set.add(tuple(_WORDS[c_add]))
            C_dict[tuple(_WORDS[c_add])] = c_add
            
            # Update cover counts
            for nb in _NEIGHBORS[c_add]:
                cover_count[nb] += 1
                if cover_count[nb] > 0:
                    uncovered[nb] = False
                    
        # After repair, we might have added many words.
        # The next iteration of Phase 1 will try to remove redundant ones.
        
    return [list(c) for c in best_C]
# EVOLVE-BLOCK-END
