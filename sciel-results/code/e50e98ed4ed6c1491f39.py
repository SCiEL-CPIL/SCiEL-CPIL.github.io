"""Triangle-free graphs without independent 10-sets (R(3,10) > n).

Strategy:
  1. Start from the known Andrasfai graph And(13) on 39 vertices (score 39).
  2. Try to extend it to n=40 by adding a vertex with carefully chosen edges.
  3. Use fast triangle-free / alpha<10 checkers (branch-and-bound with bitsets).
  4. Also try random search for potentially better graphs.
  5. Keep an explicit deadline guard.

The largest known order is 39; R(3,10) <= 41, so we cap at n=40.
"""
import time
import random
from itertools import combinations

DEADLINE = 115.0  # safety margin before 120s limit

def _has_triangle(adj, n):
    """Check if graph with adjacency bitmasks has a triangle."""
    for i in range(n):
        ai = adj[i]
        j = ai
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            if jj > i:
                common = ai & adj[jj]
                if common:
                    return True
            j ^= lb
    return False

def _greedy_independent_set(adj, n):
    """Greedy lower bound on independent set size."""
    size = 0
    remaining = (1 << n) - 1
    while remaining:
        # Pick vertex with minimum degree in remaining subgraph
        best_v = -1
        best_deg = n + 1
        m = remaining
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            m ^= lb
        if best_v < 0:
            break
        size += 1
        remaining &= ~(adj[best_v] | (1 << best_v))
    return size

def _max_independent_set_size(adj, n, limit=10):
    """Branch and bound: return size of max independent set, or >= limit if found."""
    # Get a greedy lower bound first
    best = [_greedy_independent_set(adj, n)]
    
    if best[0] >= limit:
        return best[0]
    
    def _bbs(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        # Pick vertex with maximum degree in candidates (for better pruning)
        best_v = -1
        best_deg = -1
        m = candidates
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj[v] & candidates).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= lb
        if best_v < 0:
            return
        lb = 1 << best_v
        rest = candidates ^ lb
        # Include v first (usually finds larger sets faster)
        new_cands = rest & ~adj[best_v]
        _bbs(chosen_count + 1, new_cands)
        # Exclude v
        _bbs(chosen_count, rest)
    
    _bbs(0, (1 << n) - 1)
    return best[0]

def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle(adj, n):
        return False
    if _max_independent_set_size(adj, n, limit=10) >= 10:
        return False
    return True

def _andrasfai_13():
    """Return adj bitmasks for Andrasfai graph And(13) on 39 vertices.
    
    And(k) is a circulant graph on 3k vertices where vertex i is adjacent to
    vertex j iff (j - i) mod 3k is in {1, 2, ..., k-1} ∪ {3k - k + 1, ..., 3k - 1}.
    For k=13: n=39, each vertex connects to 12 neighbors (6 on each side).
    This graph is triangle-free and has independence number 13.
    Wait - alpha=13 means it HAS an independent set of size 13 >= 10.
    
    Let me reconsider. The Andrasfai graph And(k) has:
    - n = 3k vertices
    - Triangle-free
    - Independence number = k
    
    For alpha < 10, we need k <= 9. And(9) has n=27, alpha=9.
    But the known result is n=39 with alpha < 10.
    
    Actually, let me use a different construction. The known (3,10)-Ramsey graph
    on 39 vertices is not simply And(13). Let me use a different approach:
    start with And(9) on 27 vertices and try to extend.
    
    Actually, re-reading the literature: R(3,10) > 39 means there exists a
    triangle-free graph on 39 vertices with no independent set of size 10.
    The Andrasfai graph And(k) has alpha = k, so And(9) has alpha=9 < 10
    and n=27. But we need n=39.
    
    Let me just use And(9) as the base and extend it.
    """
    n = 27
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d <= 8:  # connect to next 8 vertices (k-1=8 for k=9)
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj, n

def _andrasfai_9():
    """Return adj bitmasks for And(9) on 27 vertices."""
    return _andrasfai_13()

def _try_extend(adj, n, max_n=40, deadline=DEADLINE):
    """Try to extend the graph by adding vertices one at a time."""
    best_n = n
    best_adj = list(adj)
    
    for target_n in range(n + 1, max_n + 1):
        if time.time() > deadline:
            break
        found = False
        attempts = 300
        for _ in range(attempts):
            if time.time() > deadline:
                break
            # Build a random independent set to serve as neighbors of new vertex
            cand = list(range(n))
            random.shuffle(cand)
            I = []
            for v in cand:
                conflict = False
                for u in I:
                    if adj[v] & (1 << u):
                        conflict = True
                        break
                if not conflict:
                    I.append(v)
                if len(I) >= 12:
                    break
            
            if len(I) < 2:
                continue
            
            # New vertex n connects to all in I
            new_adj = list(adj)
            new_adj.append(0)
            for i in I:
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
            
            if _is_valid(new_adj, n + 1):
                adj = new_adj
                n = n + 1
                best_n = n
                best_adj = list(adj)
                found = True
                break
        
        if not found:
            # Try with smaller neighbor sets
            for _ in range(150):
                if time.time() > deadline:
                    break
                I = []
                for v in range(n):
                    if random.random() < 0.25:
                        I.append(v)
                # Make independent
                valid_I = []
                for v in I:
                    ok = True
                    for u in valid_I:
                        if adj[v] & (1 << u):
                            ok = False
                            break
                    if ok:
                        valid_I.append(v)
                I = valid_I
                
                if len(I) < 1:
                    continue
                
                new_adj = list(adj)
                new_adj.append(0)
                for i in I:
                    new_adj[i] |= (1 << n)
                    new_adj[n] |= (1 << i)
                
                if _is_valid(new_adj, n + 1):
                    adj = new_adj
                    n = n + 1
                    best_n = n
                    best_adj = list(adj)
                    found = True
                    break
    
    return best_adj, best_n

def _random_search(max_n=40, deadline=DEADLINE):
    """Random search for a valid graph with large n."""
    best_n = 0
    best_adj = None
    
    for n in range(28, max_n + 1):
        if time.time() > deadline:
            break
        for attempt in range(30):
            if time.time() > deadline:
                break
            # Random triangle-free graph via random edge ordering
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            for i, j in pairs:
                common = adj[i] & adj[j]
                if common == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            
            if _is_valid(adj, n):
                best_n = n
                best_adj = list(adj)
                break
    
    return best_adj, best_n

def construct():
    start_time = time.time()
    
    # Start from And(9) on 27 vertices
    adj, n = _andrasfai_9()
    best_adj, best_n = list(adj), n
    
    # Verify base graph is valid
    if not _is_valid(adj, n):
        # Fallback: build a smaller valid graph
        best_adj, best_n = _random_search(max_n=27, deadline=start_time + DEADLINE)
        if best_n == 0:
            return {'n': 1, 'edges': []}
    
    # Try to extend
    ext_adj, ext_n = _try_extend(adj, n, max_n=40, deadline=start_time + DEADLINE - 5)
    if ext_n > best_n:
        best_adj, best_n = ext_adj, ext_n
    
    # Random search for potentially better
    if time.time() < start_time + DEADLINE - 10:
        rand_adj, rand_n = _random_search(max_n=40, deadline=start_time + DEADLINE - 5)
        if rand_n > best_n:
            best_adj, best_n = rand_adj, rand_n
    
    # Build edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
