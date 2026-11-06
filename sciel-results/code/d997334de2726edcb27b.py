"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _hopcroft_karp(n, edges):
    """
    Finds maximum matching in the bipartite double cover of the graph.
    Returns matching size.
    """
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    pair_u = [-1] * n
    pair_v = [-1] * n
    INF = float('inf')
    dist = [0] * n
    
    def bfs():
        queue = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = pair_v[v]
                if w != -1 and dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
                elif w == -1:
                    found = True
        return found

    def dfs(u):
        for v in adj[u]:
            w = pair_v[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching_size = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching_size += 1
                    
    return matching_size

def _greedy_decomposition(n, edges, rng):
    """
    Greedy path decomposition with random tie-breaking.
    """
    remaining_edges = set()
    for e in edges:
        remaining_edges.add((min(e[0], e[1]), max(e[0], e[1])))
        
    rem_adj = defaultdict(set)
    for (u, v) in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    paths = []
    
    while remaining_edges:
        # Find starting edge
        # Heuristic: Pick edge with max sum of degrees, random tie-break
        best_edge = None
        best_score = -1
        candidates = []
        for e in remaining_edges:
            u, v = e
            score = len(rem_adj[u]) + len(rem_adj[v])
            if score > best_score:
                best_score = score
                candidates = [e]
            elif score == best_score:
                candidates.append(e)
        
        if not candidates:
            break
            
        start_edge = rng.choice(candidates)
        u, v = start_edge
        
        # Initialize path
        path = [u, v]
        visited = {u, v}
        
        # Remove start edge
        remaining_edges.discard(start_edge)
        rem_adj[u].discard(v)
        rem_adj[v].discard(u)
        
        # Extend to the right (from v)
        current = v
        while True:
            neighbors = [nb for nb in rem_adj[current] if nb not in visited]
            if not neighbors:
                break
            
            # Pick neighbor with max degree, random tie-break
            best_nb = None
            best_deg = -1
            cand_nbs = []
            for nb in neighbors:
                deg = len(rem_adj[nb])
                if deg > best_deg:
                    best_deg = deg
                    cand_nbs = [nb]
                elif deg == best_deg:
                    cand_nbs.append(nb)
            
            next_v = rng.choice(cand_nbs)
            path.append(next_v)
            visited.add(next_v)
            
            e = (min(current, next_v), max(current, next_v))
            remaining_edges.discard(e)
            rem_adj[current].discard(next_v)
            rem_adj[next_v].discard(current)
            
            current = next_v
        
        # Extend to the left (from u)
        current = u
        while True:
            neighbors = [nb for nb in rem_adj[current] if nb not in visited]
            if not neighbors:
                break
            
            best_nb = None
            best_deg = -1
            cand_nbs = []
            for nb in neighbors:
                deg = len(rem_adj[nb])
                if deg > best_deg:
                    best_deg = deg
                    cand_nbs = [nb]
                elif deg == best_deg:
                    cand_nbs.append(nb)
            
            next_v = rng.choice(cand_nbs)
            path.insert(0, next_v)
            visited.add(next_v)
            
            e = (min(current, next_v), max(current, next_v))
            remaining_edges.discard(e)
            rem_adj[current].discard(next_v)
            rem_adj[next_v].discard(current)
            
            current = next_v
            
        paths.append(path)
        
    return paths

def _solve_graph(n, p, seed, time_budget=30.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []
    
    # Calculate lower bound
    max_matching = _hopcroft_karp(n, edges)
    lower_bound = n - 1 - max_matching
    if lower_bound < 1:
        lower_bound = 1
        
    best_paths = None
    best_count = float('inf')
    
    start_time = time.time()
    rng = random.Random(seed + 100)
    
    # Randomized restarts
    attempts = 0
    while time.time() - start_time < time_budget:
        paths = _greedy_decomposition(n, edges, rng)
        count = len(paths)
        
        if count < best_count:
            best_count = count
            best_paths = paths
            
        if best_count == lower_bound:
            break
            
        attempts += 1
        # If we've tried enough, reduce frequency or stop
        if attempts > 200:
            break
            
    if best_paths is None:
        # Fallback: single edges
        best_paths = [[u, v] for (u, v) in edges]
        
    return best_paths

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            # Not enough time left, use fallback
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            # Allocate time budget
            remaining_time = deadline - elapsed - 10
            time_budget = max(1.0, min(40.0, remaining_time / 3))
            paths = _solve_graph(*spec, time_budget=time_budget)
            results.append(paths)
    
    return results
