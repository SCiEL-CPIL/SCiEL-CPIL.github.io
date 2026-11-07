import time
import random
from collections import deque

# Task Parameters
N = 24
PATH_EDGES = [[i, i + 1] for i in range(N - 1)]
PATH_SET = set((min(u, v), max(u, v)) for u, v in PATH_EDGES)
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N) if (u, v) not in PATH_SET]

def _build(added):
    """Builds adjacency list and matrix for path + added edges."""
    adj = [[] for _ in range(N)]
    mat = [[False] * N for _ in range(N)]
    for u, v in PATH_EDGES:
        adj[u].append(v)
        adj[v].append(u)
        mat[u][v] = mat[v][u] = True
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
        mat[u][v] = mat[v][u] = True
    return adj, mat

def _is_triangle_free(adj, mat):
    """Checks if graph is triangle-free."""
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

def _diameter(adj):
    """Calculates diameter. Returns infinity if disconnected."""
    max_dist = 0
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    q.append(v)
        if -1 in dist:
            return float('inf')
    return max_dist

def _is_valid(added):
    """Checks if added edges result in triangle-free, connected, diameter <= 4 graph."""
    adj, mat = _build(added)
    if not _is_triangle_free(adj, mat):
        return False
    diam = _diameter(adj)
    return diam <= 4

def _reduce(added, deadline):
    """Removes edges if possible while maintaining validity."""
    added = set(added)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in list(added):
            trial = added - {e}
            if _is_valid(trial):
                added = trial
                improved = True
                break
    return added

def _local_search(added, deadline):
    """Local search to minimize edge count."""
    cur = set(added)
    best = set(cur)
    no_improve = 0
    # Precompute valid candidates for speed? No, check on the fly is fast enough for N=24.
    
    while time.time() < deadline:
        cur = _reduce(cur, deadline)
        if len(cur) < len(best):
            best = set(cur)
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > 20:
            break
            
        # Try swaps: remove e, add f
        e_list = list(cur)
        random.shuffle(e_list)
        moved = False
        for e in e_list:
            if time.time() > deadline:
                break
            # Find valid f to add
            # We want to find an f such that (cur - {e}) U {f} is valid
            # Or just try to add f to cur, then reduce, and see if we get smaller?
            # Swap is better: try all f not in cur.
            cands = [p for p in ALL_PAIRS if p not in cur]
            random.shuffle(cands)
            for f in cands:
                trial = (cur - {e}) | {f}
                if _is_valid(trial):
                    cur = trial
                    moved = True
                    break
            if moved:
                break
                
        if not moved:
            # Try kicking: add random f, then reduce
            cands = [p for p in ALL_PAIRS if p not in cur]
            if not cands:
                break
            kicked = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                f = random.choice(cands)
                trial = cur | {f}
                trial = _reduce(trial, deadline)
                if len(trial) < len(cur):
                    cur = trial
                    if len(cur) < len(best):
                        best = set(cur)
                    kicked = True
                    break
            if not kicked and no_improve > 5:
                # Restart with a different seed if stuck
                break

    return best

def _greedy_construct(deadline):
    """Greedy construction: add edges that reduce diameter most."""
    added = set()
    adj, mat = _build(added)
    current_diam = _diameter(adj)
    
    while current_diam > 4 and time.time() < deadline:
        # Find pair with max distance
        # Actually, just find any valid edge that reduces diameter
        best_edge = None
        best_diam = current_diam + 1
        
        # Sample candidates to speed up, or iterate all
        # N=24, ALL_PAIRS ~ 276. Iterating all is fine.
        for u, v in ALL_PAIRS:
            if (u, v) in added or (u, v) in PATH_SET:
                continue
            # Check triangle-free for this edge
            su = set(adj[u])
            sv = set(adj[v])
            if su & sv:
                continue
            
            trial = added | {(u, v)}
            t_adj, t_mat = _build(trial)
            if not _is_triangle_free(t_adj, t_mat):
                continue
            
            t_diam = _diameter(t_adj)
            if t_diam < best_diam:
                best_diam = t_diam
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        added.add(best_edge)
        adj, mat = _build(added)
        current_diam = best_diam
        
    return _reduce(added, deadline)

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin for 90s limit
    
    best_sol = None
    
    # 1. Try Greedy
    sol = _greedy_construct(deadline - 10)
    if sol is not None and _is_valid(sol):
        best_sol = sol
        
    # 2. Try specific heuristics (Hub-based)
    # Hub at 11
    hub = 11
    added = set()
    for v in range(N):
        if v != hub:
            added.add((min(hub, v), max(hub, v)))
    sol = _reduce(added, deadline - 10)
    if _is_valid(sol) and (best_sol is None or len(sol) < len(best_sol)):
        best_sol = sol
        
    # Hub at 12
    hub = 12
    added = set()
    for v in range(N):
        if v != hub:
            added.add((min(hub, v), max(hub, v)))
    sol = _reduce(added, deadline - 10)
    if _is_valid(sol) and (best_sol is None or len(sol) < len(best_sol)):
        best_sol = sol
        
    # 3. Random Restart Local Search
    for i in range(50):
        if time.time() > deadline - 5:
            break
            
        # Generate random initial set
        k_init = random.randint(3, 10)
        init_set = set(random.sample(ALL_PAIRS, min(k_init, len(ALL_PAIRS))))
        
        # Reduce it
        cur = _reduce(init_set, deadline - 5)
        
        # Local Search
        polished = _local_search(cur, deadline - 5)
        
        if _is_valid(polished) and (best_sol is None or len(polished) < len(best_sol)):
            best_sol = polished
            
    # 4. If we found a small solution, try to prove optimality or improve further
    if best_sol is not None and len(best_sol) <= 8:
        # Aggressive local search on the best solution
        final = _local_search(best_sol, deadline - 2)
        if _is_valid(final) and len(final) < len(best_sol):
            best_sol = final

    if best_sol is None:
        # Fallback: Should not happen, but just in case
        # Return a valid large solution
        added = set()
        for v in range(N):
            if v != 11:
                added.add((min(11, v), max(11, v)))
        best_sol = _reduce(added, deadline)

    # Ensure return type is correct: list of lists of ints
    result = [[int(u), int(v)] for u, v in best_sol]
    return result

# Run construct if executed directly for testing? No, evaluator calls construct().
