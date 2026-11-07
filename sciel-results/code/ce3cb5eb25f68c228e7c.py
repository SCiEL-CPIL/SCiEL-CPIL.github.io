"""Combined program: search for a minimal triangle-free diameter-4 augmentation of the path."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}

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

def _try_remove(added, seed):
    rng = random.Random(seed)
    cur = list(added)
    improved = True
    while improved:
        improved = False
        # Shuffle to avoid getting stuck in a specific order
        indices = list(range(len(cur)))
        rng.shuffle(indices)
        for i in indices:
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
    return cur

def _try_swap(added, seed, max_swaps=20):
    rng = random.Random(seed)
    cur = list(added)
    # Generate candidate edges
    all_cands = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u, v) in _BASE_SET:
                continue
            all_cands.append((u, v))
    
    improved = True
    swaps_made = 0
    while improved and swaps_made < max_swaps:
        improved = False
        indices = list(range(len(cur)))
        rng.shuffle(indices)
        for i in indices:
            # Try to remove cur[i] and add a random candidate
            # We only swap if it allows us to remove MORE edges later?
            # Simpler: just try to find a smaller set.
            # Since we already did removal, we are at a local minimum for removal.
            # Swap might help escape local minima if we allow adding an edge to remove two?
            # No, swap keeps count same. We need to reduce count.
            # So swap is not directly useful for minimization unless it enables a subsequent removal.
            # Let's just stick to removal for now, but maybe try adding an edge and then removing two?
            # That's complex.
            pass
    return cur

def _construct_random(seed):
    rng = random.Random(seed)
    # Generate a random set of edges
    # Ensure it's likely to be valid
    # Start with a known good structure (e.g., step 8) and perturb
    base_edges = []
    step = rng.choice([6, 7, 8, 9, 10])
    for i in range(_N):
        j = i + step
        if j < _N:
            base_edges.append((i, j))
    
    # Add some random edges
    for _ in range(rng.randint(0, 5)):
        u = rng.randint(0, _N-1)
        v = rng.randint(0, _N-1)
        if u != v and (u, v) not in _BASE_SET and (v, u) not in _BASE_SET:
            if (u, v) not in base_edges and (v, u) not in base_edges:
                base_edges.append((u, v))
                
    return base_edges

def construct():
    deadline = time.time() + 85.0  # Safety margin
    best = None
    best_len = float('inf')
    
    # 1. Try structured patterns
    for step in range(5, 24):
        edges = []
        for i in range(_N):
            j = i + step
            if j < _N:
                edges.append((i, j))
        if _valid(edges):
            reduced = _try_remove(edges, seed=step)
            if len(reduced) < best_len:
                best_len = len(reduced)
                best = reduced
            if best_len <= 8:
                break # Good enough?
                
    # 2. Try random starts
    seed = 0
    while time.time() < deadline:
        # Generate a few random starts
        for _ in range(10):
            start = _construct_random(seed)
            seed += 1
            if _valid(start):
                reduced = _try_remove(start, seed)
                if len(reduced) < best_len:
                    best_len = len(reduced)
                    best = reduced
            # Also try starting from empty and greedy adding?
            # Greedy adding is slow.
            
        if best_len <= 8:
            break

    if best is None:
        # Fallback: The full bipartite graph is always valid
        # But we need to return a list.
        # The seed program returns ~121 edges.
        # We should return a valid small set if we found one, else fallback.
        # Since we always find one in step 1 (step 8 gives 16, reducible?), 
        # best should not be None.
        # If it is, return the full bipartite set.
        return [[u, v] for u in range(_N) for v in range(u + 1, _N)
                if (u - v) % 2 and (u, v) not in _BASE_SET]
        
    return [[int(u), int(v)] for (u, v) in best]
# EVOLVE-BLOCK-END
