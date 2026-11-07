"""Rewritten program: search for a minimal triangle-free diameter-4 augmentation of the path."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}

# Precompute bipartition: even/odd indices
_EVEN = [i for i in range(_N) if i % 2 == 0]
_ODD = [i for i in range(_N) if i % 2 == 1]

# All valid candidate edges (bipartite, non-base)
_CANDIDATES = []
for u in _EVEN:
    for v in _ODD:
        if (u, v) not in _BASE_SET and (v, u) not in _BASE_SET:
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

def _diameter_from_adj(adj):
    """Compute diameter from adjacency list. Returns 10**9 if disconnected."""
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

def _diameter(added):
    adj = _build_adj(added)
    return _diameter_from_adj(adj)

def _valid(added):
    d = _diameter(added)
    return d <= P['diameter']

def _construct_greedy(seed):
    """Greedy: iteratively add the edge that most reduces diameter, among bipartite candidates."""
    rng = random.Random(seed)
    added = []
    added_set = set()
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    cur = _diameter_from_adj(adj)
    while cur > P['diameter']:
        # Gather available candidates (not yet added)
        cands = [(u, v) for (u, v) in _CANDIDATES if (u, v) not in added_set]
        if not cands:
            return None
        # Evaluate each candidate
        best_edge = None
        best_d = 10**9
        # Shuffle for tie-breaking diversity
        rng.shuffle(cands)
        for (u, v) in cands:
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            d = _diameter_from_adj(adj)
            # Remove edge
            adj[u].discard(v)
            adj[v].discard(u)
            if d < best_d:
                best_d = d
                best_edge = (u, v)
            if best_d <= P['diameter']:
                # Can't do better than target
                break
        if best_edge is None or best_d == 10**9:
            return None
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append(best_edge)
        added_set.add(best_edge)
        cur = best_d
    return added

def _try_remove(added, seed):
    """Try to remove edges one at a time while maintaining validity."""
    rng = random.Random(seed)
    cur = list(added)
    improved = True
    while improved:
        improved = False
        # Shuffle order for diversity
        order = list(range(len(cur)))
        rng.shuffle(order)
        for i in order:
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
    return cur

def _local_search(added, seed, deadline):
    """Try edge swaps: remove one, add another, to reduce count or maintain."""
    rng = random.Random(seed)
    cur = list(added)
    cur_set = set(cur)
    best = cur
    best_len = len(cur)
    while time.time() < deadline:
        # Try removing an edge
        order = list(range(len(cur)))
        rng.shuffle(order)
        found = False
        for i in order:
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                cur_set.discard(cur[i] if i < len(cur) else None)
                # Rebuild set properly
                cur_set = set(cur)
                best = list(cur)
                best_len = len(cur)
                found = True
                break
        if not found:
            # Try swap: remove edge i, add candidate edge j
            swap_order = list(range(len(cur)))
            rng.shuffle(swap_order)
            for i in swap_order:
                if time.time() >= deadline:
                    break
                # Remove edge i
                trial = cur[:i] + cur[i+1:]
                # Try adding a random candidate
                cands = [c for c in _CANDIDATES if c not in set(trial)]
                if not cands:
                    continue
                # Try a few random candidates
                for _ in range(20):
                    c = rng.choice(cands)
                    trial2 = trial + [c]
                    if _valid(trial2):
                        cur = trial2
                        cur_set = set(cur)
                        best = list(cur)
                        best_len = len(cur)
                        found = True
                        break
                if found:
                    break
        if not found:
            break
    return best

def construct():
    deadline = time.time() + 50.0  # safety margin under 90s
    best = None
    best_len = 10**9
    
    # Phase 1: Greedy construction with multiple seeds
    seed = 0
    while time.time() < deadline:
        g = _construct_greedy(seed)
        if g is not None and _valid(g):
            reduced = _try_remove(g, seed)
            if len(reduced) < best_len:
                best = reduced
                best_len = len(reduced)
        seed += 1
        if best_len <= 8:
            break
    
    # Phase 2: Local search to improve
    if best is not None:
        phase2_deadline = min(deadline, time.time() + 20.0)
        ls_result = _local_search(best, 0, phase2_deadline)
        if len(ls_result) < best_len:
            best = ls_result
            best_len = len(best)
    
    # Phase 3: More greedy seeds if time remains
    if time.time() < deadline:
        seed = 100
        while time.time() < deadline:
            g = _construct_greedy(seed)
            if g is not None and _valid(g):
                reduced = _try_remove(g, seed)
                if len(reduced) < best_len:
                    best = reduced
                    best_len = len(reduced)
            seed += 1
            if best_len <= 8:
                break
    
    if best is None:
        # Fallback: bipartite complete graph minus base edges
        return [[u, v] for u, v in _CANDIDATES]
    return [[int(u), int(v)] for (u, v) in best]
# EVOLVE-BLOCK-END
