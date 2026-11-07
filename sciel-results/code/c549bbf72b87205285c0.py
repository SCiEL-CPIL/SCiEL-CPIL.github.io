import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}

# Precompute all potential edges (non-adjacent, non-distance-2)
_CANDIDATES = []
for u in range(_N):
    for v in range(u + 1, _N):
        if (u, v) in _BASE_SET:
            continue
        if abs(u - v) == 2:
            continue
        _CANDIDATES.append((u, v))

def _build_adj(added):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _is_triangle_free(added):
    # Check if adding 'added' to base creates a triangle
    # We check incrementally in the main loop, but this is for validation
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        if adj[a] & adj[b]:
            return False
        adj[a].add(b)
        adj[b].add(a)
    return True

def _diameter(added):
    adj = _build_adj(added)
    best = 0
    for s in range(_N):
        dist = [-1] * _N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 4: # Early exit if we care only about <=4
                        # But we need exact diameter for comparison
                        pass 
                    dq.append(v)
        if -1 in dist:
            return 10**9
        m = max(dist)
        if m > best:
            best = m
    return best

def _valid(added):
    if not _is_triangle_free(added):
        return False
    d = _diameter(added)
    return d <= P['diameter']

def _shares_neighbor(added, u, v):
    # Check if u and v share a neighbor in the graph formed by BASE + added
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    return bool(adj[u] & adj[v])

def _construct_greedy(seed):
    rng = random.Random(seed)
    added = []
    added_set = set()
    cur_diam = _diameter(added)
    
    while cur_diam > P['diameter']:
        cands = []
        for (u, v) in _CANDIDATES:
            if (u, v) in added_set:
                continue
            if _shares_neighbor(added, u, v):
                continue
            cands.append((u, v))
        
        if not cands:
            return None
            
        # Evaluate candidates
        best_edge = None
        best_d = 10**9
        # To save time, we can sample if cands is large, but N=24 is small
        # Max cands is ~276 - 23 - 22 = 231. This is small enough to check all.
        for (u, v) in cands:
            trial = added + [(u, v)]
            # _shares_neighbor already checked triangle-freeness for this specific edge
            # But we must ensure the *new* edge doesn't create a triangle with *other* added edges
            # The _shares_neighbor check is sufficient for triangle-freeness of the set
            # because a triangle is formed by 3 edges. If we add (u,v), a triangle exists
            # iff there is a w such that (u,w) and (w,v) are in the graph.
            # This is exactly what _shares_neighbor checks.
            
            d = _diameter(trial)
            if d < best_d:
                best_d = d
                best_edge = (u, v)
            elif d == best_d:
                # Tie-breaking: prefer edges that are "longer" in the path?
                # Or just random
                if rng.random() < 0.5:
                    best_edge = (u, v)

        if best_edge is None:
            return None
            
        added.append(best_edge)
        added_set.add(best_edge)
        cur_diam = best_d
        
    return added

def _try_remove(added, seed):
    rng = random.Random(seed)
    cur = list(added)
    improved = True
    while improved:
        improved = False
        # Shuffle to break symmetry
        indices = list(range(len(cur)))
        rng.shuffle(indices)
        for i in indices:
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
    return cur

def construct():
    deadline = time.time() + 55.0
    best = None
    
    for seed in range(100):
        if time.time() >= deadline:
            break
        g = _construct_greedy(seed)
        if g is not None and _valid(g):
            reduced = _try_remove(g, seed)
            if best is None or len(reduced) < len(best):
                best = reduced
            if best is not None and len(best) <= 5:
                break
                
    if best is None:
        # Fallback: return a valid solution if found, else empty (invalid)
        # But we must return a valid one.
        # The greedy should always find one.
        return []
        
    return [[int(u), int(v)] for (u, v) in best]
