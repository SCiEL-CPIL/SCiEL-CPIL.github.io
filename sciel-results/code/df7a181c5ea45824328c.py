import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
PATH_EDGES = P['edges']
PATH_SET = frozenset((min(u, v), max(u, v)) for u, v in PATH_EDGES)
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in PATH_SET]
ALL_PAIRS_SET = set(ALL_PAIRS)

def _build_adj(added):
    adj = [[] for _ in range(N)]
    for u, v in PATH_EDGES:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _is_triangle_free(added):
    adj = _build_adj(added)
    for u in range(N):
        neighbors = adj[u]
        n = len(neighbors)
        for i in range(n):
            a = neighbors[i]
            for j in range(i + 1, n):
                b = neighbors[j]
                # Check if a and b are adjacent
                # Using a set for quick lookup
                if b in adj[a]:
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
                    dist[v] = nd
                    q.append(v)
        for d in dist:
            if d == -1:
                return False
    return True

def _valid(added):
    if not _is_triangle_free(added):
        return False
    if not _diameter_le4(added):
        return False
    return True

def _can_add(added_set, adj, u, v):
    if u == v:
        return False
    k = (min(u, v), max(u, v))
    if k in PATH_SET:
        return False
    if k in added_set:
        return False
    # Triangle check: u-v edge would create triangle if any common neighbor
    # Common neighbors of u and v
    # For speed, we can just check intersection of adj[u] and adj[v]
    # But we also need to ensure u and v are not already connected (handled above)
    # And that adding u-v doesn't create triangle with any w
    # Triangle (u, v, w) exists if w is neighbor of both u and v.
    # So check if adj[u] intersects adj[v].
    su = set(adj[u])
    sv = set(adj[v])
    if su & sv:
        return False
    return True

def _complete(added, deadline):
    added = set(added)
    while time.time() < deadline:
        adj = _build_adj(added)
        if _is_triangle_free(added) and _diameter_le4(added):
            return added
        
        # Find farthest pair
        best_d = -1
        a, b = -1, -1
        for s in range(N):
            dist = [-1] * N
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                nd = dist[u] + 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = nd
                        q.append(v)
            for t in range(N):
                if dist[t] > best_d:
                    best_d = dist[t]
                    a, b = s, t
        
        # Try to add edge (a, b)
        cand = None
        if _can_add(added, adj, a, b):
            cand = (a, b)
        else:
            # Try to add edge from a to some other vertex
            for v in range(N):
                if v != a and _can_add(added, adj, a, v):
                    cand = (a, v)
                    break
        if cand is None:
            # Try any edge
            for u, v in ALL_PAIRS:
                if _can_add(added, adj, u, v):
                    cand = (u, v)
                    break
        if cand is None:
            return added
        
        added.add(cand)
    return added

def _reduce(added, deadline):
    added = set(added)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in list(added):
            trial = added - {e}
            if _valid(trial):
                added = trial
                improved = True
                break
    return added

def _local_search(added, deadline):
    cur = set(added)
    best = set(cur)
    no_improve = 0
    while time.time() < deadline:
        cur = _reduce(cur, deadline)
        if len(cur) < len(best):
            best = set(cur)
            no_improve = 0
        
        improved = False
        e_list = list(cur)
        random.shuffle(e_list)
        
        for e in e_list:
            if time.time() > deadline:
                break
            
            # Try to swap e with some other edge f
            # Or just remove e and see if we can fix with fewer edges?
            # Standard 1-swap: remove e, add f.
            # But we want to minimize edges.
            # If we remove e, we might lose validity.
            # If we add f, we might gain validity but increase count.
            # So we want to find f such that (cur - {e}) + {f} is valid and len is same?
            # No, that doesn't help minimize.
            # We want to find a set S such that len(S) < len(cur) and S is valid.
            # _reduce handles removal.
            # Local search with swaps can help escape local minima in the removal landscape.
            
            # Let's try: remove e. If invalid, try adding one edge f to make it valid.
            # If (cur - {e}) + {f} is valid, then we haven't reduced size.
            # But maybe from there we can remove more?
            
            # Actually, let's stick to simple reduction and random restarts or kicks.
            
            # Kick: add a random edge, then reduce.
            # If the reduced set is smaller than cur, accept.
            
            cands = [p for p in ALL_PAIRS if p not in cur]
            if not cands:
                break
            
            kicked = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                f = random.choice(cands)
                trial = cur | {f}
                trial = _reduce(trial, deadline)
                if len(trial) < len(cur):
                    cur = trial
                    if len(cur) < len(best):
                        best = set(cur)
                    no_improve = 0
                    kicked = True
                    break
            
            if kicked:
                improved = True
                break
        
        if not improved:
            no_improve += 1
            if no_improve >= 5:
                break
        
    return best

def _solve_from(seed, deadline):
    added = _complete(set(seed), deadline)
    if not _valid(added):
        return None
    added = _reduce(added, deadline)
    added = _local_search(added, deadline)
    if _valid(added):
        return added
    return None

def _make_seeds():
    seeds = []
    seeds.append([])
    
    # Hub seeds
    for hub in range(N):
        added = []
        for v in range(N):
            if v != hub and abs(v - hub) >= 2:
                added.append((hub, v))
        seeds.append(added)
        
    # Specific hub 11 (center)
    seeds.append([(11, c) for c in [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]])
    
    # Chord seeds
    for k in [4, 5, 6, 7, 8]:
        added = []
        pos = 0
        while pos + k <= N - 1:
            added.append((pos, pos + k))
            pos += k
        if pos < N - 1:
            added.append((pos, N - 1))
        seeds.append(added)
        
    return seeds

def construct():
    t0 = time.time()
    # Deadline guard: 90s wall clock. Use 85s to be safe.
    deadline = t0 + 85
    best = None
    
    seeds = _make_seeds()
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        sol = _solve_from(seed, deadline)
        if sol is not None and (best is None or len(sol) < len(best)):
            best = sol
            if len(best) <= 4: # Theoretical lower bound is likely around 5-6, 4 is very good
                break
            
    if best is None:
        # Fallback: try to build from scratch with a simple heuristic
        # Or just use the hub seed which is known to work
        fb = [(11, c) for c in [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]]
        if _valid(set(fb)):
            best = set(fb)
        else:
            # Last resort
            best = set()
            for u, v in ALL_PAIRS:
                if time.time() > deadline - 1:
                    break
                trial = best | {(u, v)}
                if _valid(trial):
                    best = trial
                    if len(best) <= 10:
                        break
    
    # Final polish
    if time.time() < deadline - 1:
        polished = _local_search(best, deadline)
        if _valid(polished) and len(polished) < len(best):
            best = polished

    return [[int(u), int(v)] for (u, v) in best]
