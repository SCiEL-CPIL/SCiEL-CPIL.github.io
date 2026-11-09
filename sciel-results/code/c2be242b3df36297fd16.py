"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _has_c6(adj, n):
    """Check if graph with adjacency sets has a 6-cycle."""
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == w:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == w or z == x:
                                continue
                            if u in adj[z]:
                                return True
    return False

def _greedy_build(n, seed=None, time_limit=100):
    """Build a C6-free graph greedily."""
    if seed is not None:
        random.seed(seed)
    
    adj = [set() for _ in range(n)]
    edge_count = 0
    start_time = time.time()
    
    # Try all edges in random order
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > time_limit:
            break
        # Tentatively add edge
        adj[u].add(v)
        adj[v].add(u)
        # Check for C6
        if _has_c6(adj, n):
            adj[u].discard(v)
            adj[v].discard(u)
        else:
            edge_count += 1
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges, edge_count

def _local_search(edges, n, time_limit=30):
    """Try to improve the graph by edge swaps."""
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    best_edges = list(edges)
    best_count = len(edges)
    start_time = time.time()
    
    # Try removing an edge and adding a different one
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    current_edges = list(edge_set)
    
    while time.time() - start_time < time_limit:
        # Randomly pick an existing edge to remove
        if not current_edges:
            break
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        
        # Remove it
        adj[u].discard(v)
        adj[v].discard(u)
        edge_set.discard((u, v))
        current_edges.pop(idx)
        
        # Try adding a random non-edge
        added = False
        for _ in range(50):
            a, b = random.choice(all_possible)
            if (a, b) in edge_set or a == b:
                continue
            adj[a].add(b)
            adj[b].add(a)
            if not _has_c6(adj, n):
                edge_set.add((a, b))
                current_edges.append((a, b))
                added = True
                break
            else:
                adj[a].discard(b)
                adj[b].discard(a)
        
        if added:
            new_count = len(edge_set)
            if new_count > best_count:
                best_count = new_count
                best_edges = [[u, v] for u, v in edge_set]
        else:
            # Put the removed edge back
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            current_edges.append((u, v))
    
    return best_edges, best_count

def construct():
    n = 40
    deadline = 110  # safety margin before 120s limit
    
    # Start with the seed: 8 disjoint K5s (100 edges)
    seed_edges = []
    for k in range(8):
        seed_edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    
    best_edges = seed_edges
    best_count = len(seed_edges)
    
    # Try greedy builds with different seeds
    start_time = time.time()
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718]:
        if time.time() - start_time > deadline - 30:
            break
        edges, count = _greedy_build(n, seed=seed, time_limit=min(10, deadline - 30 - (time.time() - start_time)))
        if count > best_count:
            best_count = count
            best_edges = edges
    
    # Try local search from best found
    if time.time() - start_time < deadline - 10:
        remaining = deadline - 10 - (time.time() - start_time)
        improved_edges, improved_count = _local_search(best_edges, n, time_limit=min(remaining, 30))
        if improved_count > best_count:
            best_count = improved_count
            best_edges = improved_edges
    
    return best_edges
# EVOLVE-BLOCK-END
