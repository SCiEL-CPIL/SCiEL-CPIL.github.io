import random
import time
import itertools

def _check_valid(edges, n):
    """Check if graph satisfies the constraint. Return True if valid."""
    adj = [[False]*n for _ in range(n)]
    deg = [0]*n
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
    
    # Check every 3-edge path s-u-v-t with s!=t
    for u in range(n):
        for v in range(n):
            if u == v or not adj[u][v]:
                continue
            for s in range(n):
                if s == u or s == v or not adj[s][u]:
                    continue
                for t in range(n):
                    if t == u or t == v or t == s or not adj[t][v]:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _violation_count(edges, n):
    """Count number of violating 3-edge paths."""
    adj = [[False]*n for _ in range(n)]
    deg = [0]*n
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
    
    count = 0
    for u in range(n):
        for v in range(n):
            if u == v or not adj[u][v]:
                continue
            for s in range(n):
                if s == u or s == v or not adj[s][u]:
                    continue
                for t in range(n):
                    if t == u or t == v or t == s or not adj[t][v]:
                        continue
                    if deg[s] == deg[t]:
                        count += 1
    return count

def _greedy_build(n, rng):
    """Build a graph greedily by adding edges in random order if valid."""
    all_edges = list(itertools.combinations(range(n), 2))
    rng.shuffle(all_edges)
    edges = []
    edge_set = set()
    for u, v in all_edges:
        new_edges = edges + [[u, v]]
        if _check_valid(new_edges, n):
            edges.append([u, v])
            edge_set.add((min(u,v), max(u,v)))
    return edges

def _local_improve(edges, n, rng, max_iters=5000):
    """Try to add more edges via local search."""
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    all_edges = list(itertools.combinations(range(n), 2))
    best_edges = list(edges)
    best_count = len(edges)
    
    for it in range(max_iters):
        # Try adding a missing edge
        missing = [(u,v) for u,v in all_edges if (min(u,v),max(u,v)) not in edge_set]
        if not missing:
            break
        rng.shuffle(missing)
        improved = False
        for u, v in missing:
            new_edges = best_edges + [[u, v]]
            if _check_valid(new_edges, n):
                best_edges = new_edges
                best_count += 1
                edge_set.add((min(u,v), max(u,v)))
                improved = True
                break
        if not improved:
            # Try removing one edge and adding two others? Complex. Just stop.
            break
    
    return best_edges

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_edges = []
    best_count = 0
    
    # First, try greedy builds with different seeds
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        edges = _greedy_build(n, rng)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
    
    # Local improvement on best
    if time.time() < deadline - 10:
        rng = random.Random(42)
        improved = _local_improve(best_edges, n, rng)
        if len(improved) > best_count:
            best_count = len(improved)
            best_edges = improved
    
    # More restarts with random initial edges
    for seed in range(20, 100):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        # Start with random subset of edges
        all_edges = list(itertools.combinations(range(n), 2))
        rng.shuffle(all_edges)
        # Try random density
        k = rng.randint(10, 30)
        initial = [[u, v] for u, v in all_edges[:k]]
        if _check_valid(initial, n):
            # Greedy add from here
            edge_set = set((min(u,v), max(u,v)) for u, v in initial)
            edges = list(initial)
            remaining = [(u,v) for u,v in all_edges if (min(u,v),max(u,v)) not in edge_set]
            rng.shuffle(remaining)
            for u, v in remaining:
                if time.time() > deadline - 5:
                    break
                new_edges = edges + [[u, v]]
                if _check_valid(new_edges, n):
                    edges.append([u, v])
                    edge_set.add((min(u,v), max(u,v)))
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
    
    # Final local improvement
    if time.time() < deadline - 5:
        rng = random.Random(123)
        improved = _local_improve(best_edges, n, rng)
        if len(improved) > best_count:
            best_count = len(improved)
            best_edges = improved
    
    # Ensure proper types
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    return result
