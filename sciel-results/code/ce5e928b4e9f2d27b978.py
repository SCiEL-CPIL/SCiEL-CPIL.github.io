import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
PATH_EDGES = P['edges']
PATH_SET = frozenset(frozenset((u, v)) for u, v in PATH_EDGES)

# Precompute path adjacency for triangle check
PATH_ADJ = [set() for _ in range(N)]
for u, v in PATH_EDGES:
    PATH_ADJ[u].add(v)
    PATH_ADJ[v].add(u)

def _build_adj(added):
    adj = [set() for _ in range(N)]
    for u, v in PATH_EDGES:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(added):
    # Check: for each vertex, its added-neighbor set must not contain adjacent path vertices
    added_neighbors = [set() for _ in range(N)]
    for u, v in added:
        added_neighbors[u].add(v)
        added_neighbors[v].add(u)
    for v in range(N):
        nbrs = added_neighbors[v]
        for a in nbrs:
            for b in nbrs:
                if a != b and b in PATH_ADJ[a]:
                    return False
    return True

def _diameter_le4(added):
    adj = _build_adj(added)
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= 4:
                continue
            nd = dist[u] + 1
            for v in adj[u]:
                if dist[v] == -1:
                    if nd > 4:
                        return False
                    dist[v] = nd
                    q.append(v)
        if -1 in dist:
            return False
    return True

def _valid(added):
    if not _is_triangle_free(added):
        return False
    if not _diameter_le4(added):
        return False
    return True

def _can_add(added_set, added_list, u, v):
    if u == v:
        return False
    key = frozenset((u, v))
    if key in PATH_SET:
        return False
    if key in added_set:
        return False
    # Check triangle-free: adding (u,v) must not create a triangle
    # Triangle formed if: there exists w such that (u,w) in added and w adjacent to v on path
    # OR (v,w) in added and w adjacent to u on path
    added_neighbors_u = set()
    added_neighbors_v = set()
    for a, b in added_list:
        if a == u:
            added_neighbors_u.add(b)
        elif b == u:
            added_neighbors_u.add(a)
        if a == v:
            added_neighbors_v.add(b)
        elif b == v:
            added_neighbors_v.add(a)
    # Check: for any w in added_neighbors_u, if w is path-adjacent to v -> triangle (u,v,w)
    for w in added_neighbors_u:
        if v in PATH_ADJ[w]:
            return False
    for w in added_neighbors_v:
        if u in PATH_ADJ[w]:
            return False
    return True

def _reduce(added, deadline):
    """Remove redundant edges."""
    added = list(added)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(added)):
            trial = added[:i] + added[i+1:]
            if _valid(trial):
                added = trial
                improved = True
                break
    return added

def _local_search(added, deadline):
    """Hill-climbing with edge swaps."""
    cur = list(added)
    best = list(cur)
    best_len = len(cur)
    
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N) if frozenset((u,v)) not in PATH_SET]
    
    no_improve = 0
    while time.time() < deadline - 2:
        cur = _reduce(cur, deadline)
        if len(cur) < best_len:
            best = list(cur)
            best_len = len(cur)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 20:
                break
        
        improved = False
        cur_set = set(frozenset((u,v)) for u,v in cur)
        cur_list = list(cur)
        
        # Try removing one edge and adding another
        random.shuffle(cur_list)
        for e in cur_list:
            if time.time() > deadline - 2:
                break
            e_set = frozenset(e)
            remaining = [x for x in cur_list if x != e]
            remaining_set = set(frozenset((u,v)) for u,v in remaining)
            
            candidates = [p for p in all_pairs if frozenset(p) not in remaining_set]
            random.shuffle(candidates)
            
            for f in candidates[:50]:
                if time.time() > deadline - 2:
                    break
                trial = remaining + [f]
                if _valid(trial):
                    trial = _reduce(trial, deadline)
                    if len(trial) < len(cur):
                        cur = trial
                        cur_set = set(frozenset((u,v)) for u,v in cur)
                        cur_list = list(cur)
                        if len(cur) < best_len:
                            best = list(cur)
                            best_len = len(cur)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try adding a random edge
            candidates = [p for p in all_pairs if frozenset(p) not in cur_set]
            if not candidates:
                break
            for _ in range(30):
                if time.time() > deadline - 2:
                    break
                f = random.choice(candidates)
                trial = cur + [f]
                if _valid(trial):
                    trial = _reduce(trial, deadline)
                    if len(trial) < len(cur):
                        cur = trial
                        cur_set = set(frozenset((u,v)) for u,v in cur)
                        if len(cur) < best_len:
                            best = list(cur)
                            best_len = len(cur)
                        break
    
    return best

def _solve_from_seed(seed, deadline):
    added = list(seed)
    # Complete: add edges to fix diameter
    while time.time() < deadline - 5:
        if _valid(added):
            break
        adj = _build_adj(added)
        # Find farthest pair
        best_dist = -1
        best_pair = None
        for s in range(N):
            dist = [-1] * N
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= 5:
                    continue
                nd = dist[u] + 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = nd
                        q.append(v)
            for t in range(N):
                if dist[t] > best_dist:
                    best_dist = dist[t]
                    best_pair = (s, t)
        
        if best_pair is None:
            break
        
        a, b = best_pair
        added_set = set(frozenset((u,v)) for u,v in added)
        added_list = list(added)
        
        # Try adding (a,b)
        if _can_add(added_set, added_list, a, b):
            added.append((a, b))
        else:
            # Try intermediate vertices
            found = False
            for v in range(N):
                if v != a and v != b and _can_add(added_set, added_list, a, v):
                    added.append((a, v))
                    found = True
                    break
                if v != a and v != b and _can_add(added_set, added_list, v, b):
                    added.append((v, b))
                    found = True
                    break
            if not found:
                # Try any valid edge
                for (u, v) in all_pairs_global:
                    if _can_add(added_set, added_list, u, v):
                        added.append((u, v))
                        break
                else:
                    break
    
    if not _valid(added):
        return None
    
    added = _reduce(added, deadline)
    added = _local_search(added, deadline)
    
    if _valid(added):
        return added
    return None

all_pairs_global = [(u, v) for u in range(N) for v in range(u+1, N) if frozenset((u,v)) not in PATH_SET]

def _make_seeds():
    seeds = []
    # Empty
    seeds.append([])
    # Hub-based seeds
    for hub in [6, 11, 12, 17]:
        added = []
        for v in range(N):
            if v != hub and abs(v - hub) >= 2:
                added.append((hub, v))
        seeds.append(added)
    # Sparse hub
    seeds.append([(11, 0), (11, 2), (11, 4), (11, 6), (11, 8), (11, 14), (11, 16), (11, 18), (11, 20), (11, 22)])
    seeds.append([(11, 1), (11, 3), (11, 5), (11, 7), (11, 9), (11, 13), (11, 15), (11, 17), (11, 19), (11, 21), (11, 23)])
    # Two-hub
    seeds.append([(8, 0), (8, 2), (8, 4), (8, 6), (8, 10), (8, 12), (8, 14), (8, 16), (8, 18), (8, 20), (8, 22),
                  (15, 1), (15, 3), (15, 5), (15, 7), (15, 9), (15, 11), (15, 13), (15, 17), (15, 19), (15, 21), (15, 23)])
    # Random
    for _ in range(5):
        added = []
        for _ in range(15):
            u, v = random.sample(range(N), 2)
            if frozenset((u,v)) not in PATH_SET:
                added.append((min(u,v), max(u,v)))
        seeds.append(added)
    return seeds

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin before 90s kill
    best = None
    best_len = float('inf')
    
    seeds = _make_seeds()
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        sol = _solve_from_seed(seed, deadline)
        if sol is not None and len(sol) < best_len:
            best = sol
            best_len = len(sol)
            if best_len <= 8:
                break
    
    if best is None:
        # Fallback
        fb = [(11, 0), (11, 2), (11, 4), (11, 6), (11, 8), (11, 14), (11, 16), (11, 18), (11, 20), (11, 22)]
        if _valid(fb):
            best = fb
        else:
            best = _solve_from_seed([], deadline)
    
    # Final polish
    if time.time() < deadline - 5:
        polished = _local_search(best, deadline)
        if _valid(polished) and len(polished) < len(best):
            best = polished
    
    return [[int(u), int(v)] for (u, v) in best]
