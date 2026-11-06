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

def _get_odd_vertices(edges):
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return [v for v, d in deg.items() if d % 2 != 0]

def _hopcroft_karp(left, right, adj):
    """
    Hopcroft-Karp algorithm for maximum bipartite matching.
    left: list of left nodes
    right: list of right nodes
    adj: dict mapping left node to list of right nodes
    Returns: number of matches
    """
    if not left or not right:
        return 0
    
    # Map nodes to indices for easier array handling
    left_idx = {u: i for i, u in enumerate(left)}
    right_idx = {v: j for j, v in enumerate(right)}
    
    # Build adjacency in terms of indices
    adj_idx = defaultdict(list)
    for u in left:
        for v in adj[u]:
            if v in right_idx:
                adj_idx[left_idx[u]].append(right_idx[v])
                
    # matchL[i] = j means left i is matched to right j
    # matchR[j] = i means right j is matched to left i
    matchL = [-1] * len(left)
    matchR = [-1] * len(right)
    
    def bfs():
        # dist[i] = distance from left node i to the nearest free right node
        dist = [0] * len(left)
        queue = deque()
        for u in range(len(left)):
            if matchL[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
                
        found = False
        while queue:
            u = queue.popleft()
            for v in adj_idx[u]:
                mu = matchR[v]
                if mu == -1:
                    found = True
                elif dist[mu] == float('inf'):
                    dist[mu] = dist[u] + 1
                    queue.append(mu)
        return found
    
    def dfs(u):
        for v in adj_idx[u]:
            mu = matchR[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                matchL[u] = v
                matchR[v] = u
                return True
        dist[u] = float('inf')
        return False
        
    matching = 0
    while bfs():
        for u in range(len(left)):
            if matchL[u] == -1:
                if dfs(u):
                    matching += 1
                    
    return matching

def _greedy_decomposition(n, edges, rng, time_budget, start_time):
    """
    Greedy path decomposition using randomized restarts.
    """
    m = len(edges)
    if m == 0:
        return []
    
    best_paths = None
    best_count = float('inf')
    
    # Try multiple randomized restarts
    max_attempts = 30
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
            
        # Shuffle edges to randomize the order they are considered
        shuffled_edges = list(edges)
        rng.shuffle(shuffled_edges)
        
        used_edges = set()
        paths = []
        
        # Build adjacency for quick lookup
        adj = defaultdict(set)
        for u, v in shuffled_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # We will greedily pick a start node, find a long path, remove edges.
        # To encourage covering odd vertices as endpoints, we can prioritize them.
        # However, a simple strategy is to just pick a random start node from remaining edges.
        
        remaining_edges = set(shuffled_edges)
        
        while remaining_edges:
            # Pick a random edge to start
            u, v = rng.choice(list(remaining_edges))
            
            # Decide direction
            if rng.random() < 0.5:
                start, end = u, v
            else:
                start, end = v, u
                
            # Try to extend the path from start
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Find neighbors of current in remaining_edges
                neighbors = [nb for nb in adj[current] if nb not in visited and (min(current, nb), max(current, nb)) in remaining_edges]
                
                if not neighbors:
                    break
                
                # Pick a random neighbor
                next_node = rng.choice(neighbors)
                path.append(next_node)
                visited.add(next_node)
                
                # Remove edge
                e = (min(current, next_node), max(current, next_node))
                remaining_edges.discard(e)
                
                current = next_node
                
                # Stop if we hit the initial end node? No, we want to extend as far as possible.
                # But wait, if we are doing a simple greedy walk, we might loop?
                # No, we check `visited`. So it's a simple path.
                
            paths.append(path)
            
            # If path length is 1, it means we couldn't extend, which shouldn't happen if we picked an edge.
            # But if start==end? No, u!=v.
            
        count = len(paths)
        if count < best_count:
            best_count = count
            best_paths = [list(p) for p in paths]
            
        # Optional: Break if we reached a good bound
        # Lower bound is max(1, odd_count/2)
        odd_verts = _get_odd_vertices(edges)
        lower_bound = max(1, len(odd_verts) // 2)
        if best_count <= lower_bound + 2:
             # We are close to optimal, maybe break early to save time?
             # But more attempts might find lower_bound.
             pass
             
    if best_paths is None:
        # Fallback: single edges
        best_paths = [[u, v] for (u, v) in edges]
        
    return best_paths

def _beam_search_longest_path(n, remaining_edges, start_node, beam_width=50, max_depth=50):
    """
    Finds a long simple path starting from start_node using beam search.
    Returns the path as a list of vertices.
    """
    if not remaining_edges:
        return []
    
    # Build adjacency for remaining edges
    rem_adj = defaultdict(set)
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    if start_node not in rem_adj:
        return []

    beam = [([start_node], 0)]
    
    for _ in range(max_depth):
        next_beam = []
        for path, score in beam:
            current = path[-1]
            visited = set(path)
            
            neighbors = [nb for nb in rem_adj[current] if nb not in visited]
            
            if not neighbors:
                next_beam.append((path, score))
                continue
            
            for nb in neighbors:
                new_path = path + [nb]
                new_score = len(new_path)
                next_beam.append((new_path, new_score))
                
        if not next_beam:
            break
            
        # Sort by score descending and keep top beam_width
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]
        
        # Check if any path can be extended
        can_extend = False
        for path, _ in beam:
            current = path[-1]
            visited = set(path)
            if any(nb not in visited for nb in rem_adj[current]):
                can_extend = True
                break
        if not can_extend:
            break
            
    if not beam:
        return []
        
    best_path = max(beam, key=lambda x: len(x[0]))[0]
    return best_path

def _solve_graph(n, p, seed, time_budget=30.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []
    
    odd_verts = _get_odd_vertices(edges)
    lower_bound = max(1, len(odd_verts) // 2)
    
    start_time = time.time()
    rng = random.Random(seed + 100)
    
    # Strategy 1: Greedy Random Walks (fast, often good)
    paths_greedy = _greedy_decomposition(n, edges, rng, time_budget / 2, start_time)
    count_greedy = len(paths_greedy)
    
    # Strategy 2: Beam Search (slower, might find longer paths)
    # Use remaining time
    remaining_time = time_budget - (time.time() - start_time)
    
    best_paths = paths_greedy
    best_count = count_greedy
    
    if remaining_time > 5.0:
        # Try beam search approach
        rem_edges = set((min(u, v), max(u, v)) for u, v in edges)
        paths_beam = []
        
        while rem_edges:
            if time.time() - start_time > time_budget - 5.0:
                break
                
            rem_deg = defaultdict(int)
            for u, v in rem_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
                
            current_odd = [v for v, d in rem_deg.items() if d % 2 != 0]
            
            if current_odd:
                # Pick random odd vertex
                start_node = rng.choice(current_odd)
            else:
                if rem_deg:
                    max_deg = max(rem_deg.values())
                    candidates = [v for v, d in rem_deg.items() if d == max_deg]
                    start_node = rng.choice(candidates)
                else:
                    break
                    
            path = _beam_search_longest_path(n, rem_edges, start_node, beam_width=20, max_depth=n)
            
            if not path or len(path) < 2:
                if rem_edges:
                    incident = [(u, v) for u, v in rem_edges if u == start_node or v == start_node]
                    if incident:
                        u, v = rng.choice(incident)
                        path = [u, v]
                    else:
                        u, v = rng.choice(list(rem_edges))
                        path = [u, v]
                else:
                    break
            
            paths_beam.append(path)
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = (min(u, v), max(u, v))
                rem_edges.discard(e)
                
        if len(paths_beam) < best_count:
            best_paths = paths_beam
            best_count = len(paths_beam)
            
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
