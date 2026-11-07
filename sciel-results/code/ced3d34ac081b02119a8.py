import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
PATH_EDGES = P['edges']
PATH_SET = set((min(u, v), max(u, v)) for u, v in PATH_EDGES)
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in PATH_SET]

def _build(added):
    adj = [[] for _ in range(N)]
    mat = [[False] * N for _ in range(N)]
    for u, v in PATH_EDGES:
        adj[u].append(v); adj[v].append(u)
        mat[u][v] = mat[v][u] = True
    for u, v in added:
        adj[u].append(v); adj[v].append(u)
        mat[u][v] = mat[v][u] = True
    return adj, mat

def _tri_free(adj, mat):
    for u in range(N):
        au = adj[u]
        for i in range(len(au)):
            a = au[i]
            rowa = mat[a]
            for j in range(i + 1, len(au)):
                b = au[j]
                if rowa[b]:
                    return False
    return True

def _diam_le4(adj):
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
        for d in dist:
            if d == -1:
                return False
    return True

def _valid_set(added):
    adj, mat = _build(added)
    if not _tri_free(adj, mat):
        return False
    return _diam_le4(adj)

def _local_search(initial, target_size, deadline):
    current = set(initial)
    best = set(initial)
    if _valid_set(current):
        best = set(current)
    
    # Try to reduce size if possible, or just find any valid set of target_size
    # We prioritize finding a valid set of size <= target_size
    
    while time.time() < deadline:
        if _valid_set(current) and len(current) < len(best):
            best = set(current)
            
        # If we have a valid solution, try to reduce it
        if _valid_set(current):
            # Try removing an edge
            removed = False
            for e in list(current):
                trial = current - {e}
                if _valid_set(trial):
                    current = trial
                    removed = True
                    break
            if removed:
                continue
        
        # If invalid or cannot reduce, try to fix/improve
        # Strategy: Randomly add/remove/swap edges to try to reach validity
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add' and len(current) < target_size + 2:
            candidates = [p for p in ALL_PAIRS if p not in current]
            if candidates:
                e = random.choice(candidates)
                current = current | {e}
        elif move_type == 'remove' and len(current) > 0:
            e = random.choice(list(current))
            current = current - {e}
        elif move_type == 'swap' and len(current) > 0:
            e_old = random.choice(list(current))
            candidates = [p for p in ALL_PAIRS if p not in current]
            if candidates:
                e_new = random.choice(candidates)
                current = (current - {e_old}) | {e_new}
        else:
            # Fallback: Random restart-ish
            if random.random() < 0.1:
                current = set(random.sample(ALL_PAIRS, min(target_size, len(ALL_PAIRS))))
                
        # Hard limit on size
        if len(current) > target_size + 2:
             # Remove random edge
             e = random.choice(list(current))
             current = current - {e}

    return best

def _make_seeds(target_size):
    seeds = []
    # Seed 1: Random
    seeds.append(set(random.sample(ALL_PAIRS, min(target_size, len(ALL_PAIRS)))))
    
    # Seed 2: Structured "Step"
    # Try to cover the range
    step = N // (target_size + 1)
    added = []
    for i in range(1, target_size + 1):
        u = i * step
        v = (i + 1) * step if (i + 1) * step < N else N - 1
        if u != v and (min(u,v), max(u,v)) not in PATH_SET:
            added.append((min(u,v), max(u,v)))
    # Ensure size is close
    while len(added) < target_size:
        e = random.choice(ALL_PAIRS)
        if e not in added:
            added.append(e)
    seeds.append(set(added[:target_size]))
    
    # Seed 3: Hub based
    hub = N // 2
    added = []
    for v in range(N):
        if v != hub and abs(v - hub) >= 2 and (min(hub,v), max(hub,v)) not in PATH_SET:
            added.append((min(hub,v), max(hub,v)))
            if len(added) >= target_size:
                break
    seeds.append(set(added))
    
    return seeds

def construct():
    t0 = time.time()
    deadline = t0 + 80
    
    best_solution = None
    best_size = 100
    
    # Try to find solution with 7 edges first
    target_sizes = [7, 8, 9, 10]
    
    for target in target_sizes:
        if time.time() > deadline - 5:
            break
            
        seeds = _make_seeds(target)
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            sol = _local_search(seed, target, deadline)
            if sol is not None and _valid_set(sol):
                if len(sol) < best_size:
                    best_size = len(sol)
                    best_solution = sol
                    if best_size == 7: # Can't get better than 7 if 7 is possible? Maybe 6?
                         # Check if 6 is possible?
                         pass
                         
    # If no solution found, fallback to a known valid one or empty
    if best_solution is None:
        # Try a brute force small search for 7 edges
        # Or just return a valid 8-edge solution
        # Let's try to construct a valid 8-edge solution manually if search fails
        # But search should find it.
        
        # Fallback: Just return empty list? No, must be valid.
        # The path itself is valid (triangle free, connected, diameter 23).
        # But we need diameter <= 4.
        # So we MUST find a valid set.
        
        # If search failed, it's a problem.
        # Let's force a search with a longer deadline for 8 edges
        if time.time() < deadline:
             sol = _local_search(set(random.sample(ALL_PAIRS, 8)), 8, deadline)
             if sol and _valid_set(sol):
                 best_solution = sol
                 
    if best_solution is None:
        # Absolute fallback: Return a set that is likely valid
        # This might fail validation, but better than nothing?
        # No, "Any broken rule makes the whole result invalid".
        # We need to be sure.
        
        # Let's try a specific known good structure for 8 edges
        # (0,4), (4,8), (8,12), (12,16), (16,20), (20,23) -> 6 edges, diam 8.
        # Add (0, 10), (10, 20)?
        
        # If we reach here, we have failed to find a valid solution.
        # Return empty list? No.
        # Return a set of edges that we are confident is valid.
        
        # Let's try:
        # (0, 5), (5, 10), (10, 15), (15, 20), (20, 23), (0, 15), (10, 23)?
        
        # To be safe, if no solution found, we return a set of 10 edges 
        # that we verify one last time.
        
        fallback = set()
        for i in range(0, N, 4):
            for j in range(i+4, N, 4):
                if (i, j) not in PATH_SET:
                    fallback.add((i, j))
                    if len(fallback) >= 10:
                        break
            if len(fallback) >= 10:
                break
        
        if _valid_set(fallback):
            best_solution = fallback
        else:
            # Last resort: Random search with everything left
            while time.time() < deadline:
                sol = _local_search(set(random.sample(ALL_PAIRS, 10)), 10, deadline)
                if sol and _valid_set(sol):
                    best_solution = sol
                    break
                    
    if best_solution is None:
        return [] # Should not happen
        
    return [[int(u), int(v)] for (u, v) in best_solution]

