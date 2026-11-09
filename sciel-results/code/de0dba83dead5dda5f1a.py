import time
import random

def _can_add_edge(u, v, adj, n):
    """Check if adding edge (u, v) creates a cycle of length <= 6."""
    dist = [-1] * n
    dist[u] = 0
    queue = [u]
    for d in range(5):
        next_queue = []
        for curr in queue:
            for nei in adj[curr]:
                if dist[nei] == -1:
                    if nei == v:
                        return False
                    dist[nei] = d + 1
                    next_queue.append(nei)
        queue = next_queue
    return True

def _greedy_construct(n, seed, time_limit=1):
    random.seed(seed)
    adj = [set() for _ in range(n)]
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    start_time = time.time()
    edge_count = 0
    
    for u, v in all_edges:
        if time.time() - start_time > time_limit:
            break
        if _can_add_edge(u, v, adj, n):
            adj[u].add(v)
            adj[v].add(u)
            edge_count += 1
            
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges, edge_count

def _local_search(edges, n, time_limit=30):
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
        
    best_edges = list(edges)
    best_count = len(edges)
    start_time = time.time()
    
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try to add a random non-edge
        for _ in range(50):
            u, v = random.choice(all_possible)
            if (u, v) in edge_set:
                continue
            if _can_add_edge(u, v, adj, n):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                improved = True
                break
                
        if improved:
            best_count = len(edge_set)
            best_edges = [[u, v] for u, v in edge_set]
            continue
            
        # Try to swap
        if not edge_set:
            break
        rem_u, rem_v = random.choice(list(edge_set))
        adj[rem_u].discard(rem_v)
        adj[rem_v].discard(rem_u)
        edge_set.discard((rem_u, rem_v))
        
        added = False
        for _ in range(100):
            u, v = random.choice(all_possible)
            if (u, v) in edge_set:
                continue
            if _can_add_edge(u, v, adj, n):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                added = True
                break
                
        if not added:
            adj[rem_u].add(rem_v)
            adj[rem_v].add(rem_u)
            edge_set.add((rem_u, rem_v))
            
    return best_edges, best_count

def construct():
    n = 40
    deadline = 140
    start_time = time.time()
    
    best_edges = []
    best_count = 0
    good_starts = []
    
    # Many greedy starts
    seeds = list(range(100))
    random.shuffle(seeds)
    for seed in seeds:
        if time.time() - start_time > deadline - 40:
            break
        edges, count = _greedy_construct(n, seed, time_limit=1)
        if count > best_count:
            best_count = count
            best_edges = edges
        if count >= 100:
            good_starts.append((count, edges))
            
    # Local search on the best few
    good_starts.sort(reverse=True)
    for count, edges in good_starts[:5]:
        if time.time() - start_time > deadline - 5:
            break
        remaining = min(20, deadline - 5 - (time.time() - start_time))
        if remaining <= 0:
            break
        improved_edges, improved_count = _local_search(edges, n, time_limit=remaining)
        if improved_count > best_count:
            best_count = improved_count
            best_edges = improved_edges
            
    return best_edges
