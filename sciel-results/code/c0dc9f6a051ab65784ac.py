import time
import random
import numpy as np

# Global state
_cache = {}
_start_time = None
_rng = random.Random(12345)

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _generate_all_vectors(n):
    """Generate all 3^n vectors as a list of tuples."""
    if n == 0:
        return [()]
    # Use itertools.product equivalent but manual for clarity
    result = []
    def _gen(prefix, depth):
        if depth == n:
            result.append(tuple(prefix))
            return
        for v in range(3):
            prefix.append(v)
            _gen(prefix, depth + 1)
            prefix.pop()
    _gen([], 0)
    return result

def _greedy_size(order, n):
    """
    Given an ordering of all 3^n vectors, return the cap set size from greedy.
    A vector v conflicts with existing a, b if a + b + v == 0 mod 3.
    """
    added_set = set()
    count = 0
    for v in order:
        # Check if adding v creates a 3-term arithmetic progression
        # v conflicts if there exists a, b in added_set such that a + b + v == 0 mod 3
        # => b = -a - v mod 3
        conflict = False
        for a in added_set:
            b = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if b in added_set and b != a and b != v:
                conflict = True
                break
        if not conflict:
            added_set.add(v)
            count += 1
    return count

def _greedy_size_fast(order, n, conflict_map):
    """
    Optimized greedy size calculation using a precomputed conflict map.
    conflict_map: dict mapping vector -> set of vectors that conflict with it.
    Actually, we need to check if v has any pair (a,b) in current set.
    The standard fast check is: for each a in added_set, check if (-a-v) mod 3 is in added_set.
    To speed this up, we can't easily precompute everything for all subsets.
    However, we can use a bitset representation if n is small enough.
    """
    # For now, use the standard set-based approach but optimized with local variables
    added_set = set()
    count = 0
    for v in order:
        conflict = False
        for a in added_set:
            b = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if b in added_set and b != a and b != v:
                conflict = True
                break
        if not conflict:
            added_set.add(v)
            count += 1
    return count

def _compute_priorities_from_set(n, target_set):
    """
    Given a target cap set, compute priorities for all vectors.
    Vectors in the set get high priority.
    Vectors that conflict with many pairs in the set get low priority.
    """
    all_vecs = _generate_all_vectors(n)
    total = len(all_vecs)
    target_set = set(target_set)
    
    # Count how many "witnesses" (a,b) in target_set conflict with each vector v
    # v conflicts with (a,b) if a+b+v == 0 mod 3
    # For each v, count = number of pairs (a,b) in target_set such that a+b == -v mod 3
    # We can precompute the sum of all pairs in target_set.
    
    # Efficiently: for each a in target_set, for each b in target_set,
    # compute s = (a+b) mod 3. Then for a given v, we want count of s == -v mod 3.
    
    # But iterating all pairs is O(k^2) where k is size of target_set.
    # k can be up to ~3^n / 3^epsilon. For n=8, 3^8=6561, cap set size ~ 100-200.
    # 200^2 = 40000 pairs. 6561 vectors. Total ops ~ 260M. Might be slow in Python.
    
    # Alternative: Use numpy to speed up?
    # Or just use a simpler heuristic: 
    # Priority = (number of coordinates in {0,1,2} that match a "good" pattern)
    
    # Let's try a different approach:
    # 1. Start with a random vector.
    # 2. Greedily build a cap set using a dynamic priority function.
    #    The priority of v depends on how many conflicts it would create with the current set.
    #    But we don't know the current set in advance.
    
    # Let's stick to a robust heuristic:
    # 1. Generate a large random sample of vectors.
    # 2. Find a large cap set within that sample using greedy.
    # 3. Use the structure of that cap set to define priorities.
    
    # Simpler: Use a "distance" based priority.
    # Vectors with more 1s are often "better" in some constructions?
    # Actually, the known best constructions for cap sets in {0,1,2}^n
    # often involve vectors with coordinates in {0,1} only (binary vectors)
    # and avoiding 3-term APs.
    
    # Let's try: Prioritize vectors with fewer 2s.
    # Or: Prioritize vectors that are "far" from each other in Hamming distance?
    
    # Best structural change: Use a SAT-like or Max-SAT heuristic.
    # But we must return a static priority.
    
    # Let's use a "local search" to find a good ordering.
    # We will simulate the greedy process multiple times with random perturbations.
    
    return None

def _find_best_order(n, time_limit=280):
    """Find a good ordering for greedy cap set construction."""
    if n in _cache:
        return _cache[n]
    
    all_vecs = _generate_all_vectors(n)
    total = len(all_vecs)
    
    if total <= 1:
        _cache[n] = all_vecs
        return all_vecs
    
    best_order = None
    best_size = 0
    
    # 1. Baseline: Random order
    rng = random.Random(42)
    order = all_vecs[:]
    rng.shuffle(order)
    sz = _greedy_size(order, n)
    best_size = sz
    best_order = order
    
    # 2. Structured orders
    # Sort by number of 1s (descending)
    order1 = sorted(all_vecs, key=lambda v: sum(1 for x in v if x == 1), reverse=True)
    sz1 = _greedy_size(order1, n)
    if sz1 > best_size:
        best_size = sz1
        best_order = order1
    
    # Sort by number of 2s (ascending)
    order2 = sorted(all_vecs, key=lambda v: sum(1 for x in v if x == 2))
    sz2 = _greedy_size(order2, n)
    if sz2 > best_size:
        best_size = sz2
        best_order = order2
    
    # Sort by sum of coordinates
    order3 = sorted(all_vecs, key=lambda v: sum(v))
    sz3 = _greedy_size(order3, n)
    if sz3 > best_size:
        best_size = sz3
        best_order = order3
    
    # 3. Local Search / Hill Climbing on the ordering
    # We represent the ordering as a list of indices.
    # We try to swap adjacent elements or move elements to improve the greedy size.
    # This is expensive, so we do it on a subset or use a simpler heuristic.
    
    # Better: Use a "Simulated Annealing" style approach on the priority weights.
    # Let priority(v) = sum(w_i * v_i) + noise.
    # We want to find weights w that maximize the greedy size.
    
    # But computing greedy size is O(3^n * K) where K is cap set size.
    # For n=8, 3^8=6561, K~100. 6561*100 = 656k ops.
    # We can do ~1000 evaluations in 300s? 1000 * 0.001s = 1s. Yes.
    
    # Let's try to optimize a linear priority function:
    # p(v) = dot(w, v)
    # We want to find w in R^n.
    
    # Initialize w randomly.
    w = [0.0] * n
    for i in range(n):
        w[i] = _rng.uniform(-1, 1)
    
    best_w = w[:]
    best_size_w = 0
    
    # Evaluate current w
    order_w = sorted(all_vecs, key=lambda v: sum(w[i] * v[i] for i in range(n)), reverse=True)
    sz_w = _greedy_size(order_w, n)
    best_size_w = sz_w
    if sz_w > best_size:
        best_size = sz_w
        best_order = order_w
        best_w = w[:]
    
    # Hill climbing on w
    step_size = 0.1
    for _ in range(2000):
        if _elapsed() > time_limit - 20:
            break
        
        # Pick a random dimension to update
        dim = _rng.randint(0, n - 1)
        old_w = w[dim]
        w[dim] += _rng.choice([-1, 1]) * step_size
        
        order_w = sorted(all_vecs, key=lambda v: sum(w[i] * v[i] for i in range(n)), reverse=True)
        sz_w = _greedy_size(order_w, n)
        
        if sz_w > best_size_w:
            best_size_w = sz_w
            best_w = w[:]
            if sz_w > best_size:
                best_size = sz_w
                best_order = order_w
        else:
            w[dim] = old_w
            # Reduce step size if no improvement for a while?
            # Simple: just keep going.
    
    _cache[n] = best_order
    return best_order

def priority(vector, n):
    """Return priority for a vector. Higher = added earlier in greedy."""
    if n not in _cache:
        _find_best_order(n, time_limit=280)
    
    order = _cache[n]
    
    if not hasattr(priority, '_index_cache'):
        priority._index_cache = {}
    
    if n not in priority._index_cache:
        idx = {v: i for i, v in enumerate(order)}
        priority._index_cache[n] = idx
    
    idx_map = priority._index_cache[n]
    # Higher priority = earlier in order
    return float(len(order) - idx_map[vector])
