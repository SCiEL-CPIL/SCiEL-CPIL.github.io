"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = 24

def _path_edges_set():
    s = set()
    for u, v in P['edges']:
        s.add((min(u,v), max(u,v)))
    return s

PATH_SET = _path_edges_set()

def _all_non_path_edges():
    """All candidate edges not in the path."""
    out = []
    for u in range(N):
        for v in range(u+1, N):
            if (u, v) not in PATH_SET:
                out.append((u, v))
    return out

ALL_NONPATH = _all_non_path_edges()

def _build_adj(added):
    adj = [[] for _ in range(N)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_diameter(adj):
    maxd = 0
    for s in range(N):
        dist = [-1]*N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= maxd:
                pass
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if dist[v] > maxd:
                        maxd = dist[v]
                    q.append(v)
        if -1 in dist:
            return -1
    return maxd

def _triangle_free(adj):
    adj_set = [set(a) for a in adj]
    for u in range(N):
        for v in adj[u]:
            for w in adj[u]:
                if v < w and w in adj_set[v]:
                    return False
    return True

def _valid(added):
    # no duplicates / no path edges
    es = set(PATH_SET)
    for u, v in added:
        k = (min(u,v), max(u,v))
        if k in es:
            return False
        es.add(k)
    adj = _build_adj(added)
    if not _triangle_free(adj):
        return False
    d = _bfs_diameter(adj)
    return d >= 0 and d <= 4

def _quick_valid(added):
    es = set(PATH_SET)
    for u, v in added:
        k = (min(u,v), max(u,v))
        if k in es:
            return False
        es.add(k)
    adj = _build_adj(added)
    adj_set = [set(a) for a in adj]
    for u in range(N):
        for v in adj[u]:
            for w in adj[u]:
                if v < w and w in adj_set[v]:
                    return False
    # diameter via BFS with early stop
    for s in range(N):
        dist = [-1]*N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du > 4:
                return False
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if dist[v] > 4:
                        return False
                    q.append(v)
        if -1 in dist:
            return False
    return True

def _initial_solution():
    # 10-edge star at vertex 12 to odd vertices on each side
    return [[12,1],[12,3],[12,5],[12,7],[12,9],
            [12,15],[12,17],[12,19],[12,21],[12,23]]

def _greedy_add(start_edges, max_time=30):
    """Add edges greedily to reduce diameter, staying triangle-free and valid."""
    start_time = time.time()
    cur = [list(e) for e in start_edges]
    cur_set = set((min(u,v),max(u,v)) for u,v in cur)
    while time.time() - start_time < max_time:
        adj = _build_adj(cur)
        d = _bfs_diameter(adj)
        if d <= 4:
            break
        # find farthest pair
        far = None
        far_d = -1
        for s in range(N):
            dist = [-1]*N
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u]+1
                        q.append(v)
            for t in range(N):
                if dist[t] > far_d:
                    far_d = dist[t]
                    far = (s, t)
        if far is None:
            break
        s, t = far
        # try adding edges that shorten dist(s,t)
        candidates = []
        for (u, v) in ALL_NONPATH:
            if (u,v) in cur_set:
                continue
            # quick triangle check: adding (u,v) must not create a triangle
            # i.e., no common neighbor of u and v in current adj
            adj_set = [set(a) for a in adj]
            common = adj_set[u] & adj_set[v]
            if common:
                continue
            candidates.append((u, v))
        if not candidates:
            break
        # try each candidate, keep the one that most reduces diameter
        best_e = None
        best_d = far_d
        for (u, v) in candidates:
            trial = cur + [[u, v]]
            adj2 = _build_adj(trial)
            d2 = _bfs_diameter(adj2)
            if d2 >= 0 and d2 < best_d:
                best_d = d2
                best_e = [u, v]
        if best_e is None:
            break
        cur.append(best_e)
        cur_set.add((min(best_e[0], best_e[1]), max(best_e[0], best_e[1])))
    return cur

def _greedy_removal(start_edges, max_time=40):
    """Start from a valid set; repeatedly try removing any edge that keeps validity."""
    start_time = time.time()
    cur = [list(e) for e in start_edges]
    if not _quick_valid(cur):
        return start_edges
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for i in range(len(cur)):
            if time.time() - start_time > max_time:
                break
            trial = cur[:i] + cur[i+1:]
            if _quick_valid(trial):
                cur = trial
                improved = True
                break
    return cur

def _random_restart_greedy(restarts=50, max_time=50):
    """Randomized greedy: pick random candidate edge orderings and add/remove."""
    start_time = time.time()
    import random
    rng = random.Random(12345)
    best = None
    best_len = 999
    # candidate pool
    pool = list(ALL_NONPATH)
    for it in range(restarts):
        if time.time() - start_time > max_time:
            break
        cur = []
        cur_set = set()
        # shuffle pool
        shuffled = list(pool)
        rng.shuffle(shuffled)
        # build a connected-ish starting point: start empty, add edges to reduce diameter
        # First ensure connectivity is inherent (path is already there), so we can start empty
        # and just add edges to get diameter <= 4
        for (u, v) in shuffled:
            if time.time() - start_time > max_time:
                break
            trial = cur + [[u, v]]
            if _quick_valid(trial):
                cur = trial
                cur_set.add((u, v))
                if len(cur) >= 12:
                    break
        # Now try to trim
        cur = _greedy_removal(cur, max_time=min(5, max_time - (time.time() - start_time)))
        if _quick_valid(cur) and len(cur) < best_len:
            best = [list(e) for e in cur]
            best_len = len(cur)
            if best_len <= 6:
                break
    return best

def _local_search_polish(added_edges, max_time=20):
    """Try to reduce edge count via 1-for-0 and 1-for-1 swaps, preserving validity."""
    start_time = time.time()
    cur = [list(e) for e in added_edges]
    if not _quick_valid(cur):
        return added_edges
    best = cur
    best_len = len(cur)
    while time.time() - start_time < max_time:
        improved = False
        # removal
        for i in range(len(cur)):
            if time.time() - start_time > max_time:
                break
            trial = cur[:i] + cur[i+1:]
            if _quick_valid(trial):
                cur = trial
                improved = True
                if len(cur) < best_len:
                    best = cur
                    best_len = len(cur)
                break
        if improved:
            continue
        # swap: remove one, add one different
        for i in range(len(cur)):
            if time.time() - start_time > max_time:
                break
            removed = cur[i]
            base = cur[:i] + cur[i+1:]
            base_set = set((min(u,v),max(u,v)) for u,v in base)
            base_set |= PATH_SET
            for (u, v) in ALL_NONPATH:
                if time.time() - start_time > max_time:
                    break
                if (u, v) in base_set:
                    continue
                trial = base + [[u, v]]
                if _quick_valid(trial):
                    cur = trial
                    improved = True
                    if len(cur) < best_len:
                        best = cur
                        best_len = len(cur)
                    break
            if improved:
                break
        if not improved:
            break
    return best

def _try_small_exhaustive(target_len, max_time=20):
    """Try to find a valid solution of exactly target_len edges via greedy set cover style."""
    start_time = time.time()
    # Use a simple randomized local search to find small solutions
    import random
    rng = random.Random(98765)
    best = None
    best_len = target_len + 1
    for it in range(300):
        if time.time() - start_time > max_time:
            break
        # random start: pick target_len random edges
        cur = rng.sample(ALL_NONPATH, k=target_len)
        cur = [list(e) for e in cur]
        if not _quick_valid(cur):
            # try to fix: remove invalid edges then add
            keep = []
            for e in cur:
                trial = keep + [e]
                if _quick_valid(trial):
                    keep = trial
            cur = keep
            # pad
            for (u, v) in ALL_NONPATH:
                if len(cur) >= target_len:
                    break
                if (min(u,v),max(u,v)) in set((min(a,b),max(a,b)) for a,b in cur):
                    continue
                trial = cur + [[u, v]]
                if _quick_valid(trial):
                    cur = trial
        # polish
        cur = _greedy_removal(cur, max_time=min(2, max_time - (time.time() - start_time)))
        if _quick_valid(cur) and len(cur) < best_len:
            best = cur
            best_len = len(cur)
            if best_len <= 5:
                break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit

    # Start with known 10-edge solution
    solution = _initial_solution()

    # Greedy removal to trim the initial solution
    if time.time() < deadline:
        solution = _greedy_removal(solution, max_time=min(10, deadline - time.time()))

    # Randomized restart greedy to find smaller solutions
    if time.time() < deadline:
        remaining = min(35, deadline - time.time())
        if remaining > 0:
            rnd = _random_restart_greedy(restarts=60, max_time=remaining)
            if rnd is not None and len(rnd) < len(solution):
                solution = rnd

    # Try to find even smaller via targeted search
    if time.time() < deadline:
        remaining = min(20, deadline - time.time())
        if remaining > 0 and len(solution) > 6:
            small = _try_small_exhaustive(target_len=max(5, len(solution)-2), max_time=remaining)
            if small is not None and len(small) < len(solution):
                solution = small

    # Local search polish
    if time.time() < deadline:
        remaining = min(15, deadline - time.time())
        if remaining > 0:
            solution = _local_search_polish(solution, max_time=remaining)

    # Final greedy removal pass
    if time.time() < deadline:
        remaining = min(5, deadline - time.time())
        if remaining > 0:
            solution = _greedy_removal(solution, max_time=remaining)

    # Verify final solution
    if not _quick_valid(solution):
        solution = _initial_solution()

    # Ensure return type is correct
    result = []
    for e in solution:
        result.append([int(e[0]), int(e[1])])

    return result
# EVOLVE-BLOCK-END
