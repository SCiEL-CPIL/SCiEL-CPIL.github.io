import time
import random
import math

_cache = {}
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _generate_all_vectors(n):
    if n == 0:
        return [()]
    result = []
    def _gen(prefix, depth):
        if depth == n:
            result.append(tuple(prefix))
            return
        for v in range(3):
            prefix.append(v)
            _gen(prefix, depth+1)
            prefix.pop()
    _gen([], 0)
    return result

def _greedy_size(order, n):
    added = set()
    blocked = set()
    size = 0
    for v in order:
        if v in blocked:
            continue
        # v is allowed
        # We need to block future vectors that would form AP with v and any a in added
        # Pairs (v, a) for a in added
        # Also, does v form AP with itself? No, distinct.
        # So we iterate over current added (before adding v)
        # To avoid copying, we can iterate over added, but we must not use v in the loop
        # since v is not in added yet.
        for a in added:
            c = tuple((-a[i] - v[i]) % 3 for i in range(n))
            blocked.add(c)
        
        added.add(v)
        size += 1
    return size

def _find_best_priorities(n, time_limit=280):
    if n in _cache:
        return _cache[n]
    
    all_vecs = _generate_all_vectors(n)
    N = len(all_vecs)
    rng = random.Random(42)
    
    best_order = None
    best_size = 0
    
    start_time = time.time()
    
    # Heuristic Initializations
    # 1. Random
    # 2. Weight sort
    # 3. Zero-heavy?
    
    inits = []
    # Random
    r_order = all_vecs[:]
    rng.shuffle(r_order)
    inits.append(r_order)
    
    # Weight sort (sum of coords)
    w_order = sorted(all_vecs, key=lambda v: sum(v))
    inits.append(w_order)
    
    # Weight sort descending
    w_order_desc = sorted(all_vecs, key=lambda v: sum(v), reverse=True)
    inits.append(w_order_desc)
    
    # "Sparse" sort (number of non-zeros)
    s_order = sorted(all_vecs, key=lambda v: sum(1 for x in v if x != 0))
    inits.append(s_order)
    
    for init_order in inits:
        if time.time() - start_time > time_limit:
            break
            
        current_order = init_order[:]
        current_size = _greedy_size(current_order, n)
        
        if current_size > best_size:
            best_size = current_size
            best_order = current_order[:]
            
        # Local Search: Hill Climbing with Segment Reversal
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Try a few random moves
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                i = rng.randint(0, N - 1)
                j = rng.randint(0, N - 1)
                if i > j:
                    i, j = j, i
                if i == j:
                    continue
                
                # Reverse segment i to j
                # current_order[i:j+1] = reversed(current_order[i:j+1])
                segment = current_order[i:j+1]
                current_order[i:j+1] = segment[::-1]
                
                new_size = _greedy_size(current_order, n)
                
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                    if new_size > best_size:
                        best_size = new_size
                        best_order = current_order[:]
                else:
                    # Revert
                    current_order[i:j+1] = segment
                
                if improved:
                    break # Restart the "for" loop to try more moves from new state?
                    # Or just continue to next move.
                    # Usually, accept and continue.
        
    # Map order to priority
    # priority(vector) should be such that higher priority = earlier in order
    # index 0 is highest priority.
    # So priority(v) = N - index(v)
    priority_map = {}
    for idx, v in enumerate(best_order):
        priority_map[v] = float(N - idx)
        
    _cache[n] = priority_map
    return priority_map

def priority(vector, n):
    if n not in _cache:
        _find_best_priorities(n, time_limit=280)
    
    scores = _cache[n]
    return float(scores[vector])
