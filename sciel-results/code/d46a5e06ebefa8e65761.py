import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    seed_result = _andrasfai_27()
    best_n = seed_result['n']
    best_graph = seed_result
    
    # Try local search on the seed to see if we can grow it
    result = _local_search(seed_result, deadline)
    if result['n'] > best_n:
        best_n = result['n']
        best_graph = result
    
    return best_graph

def _andrasfai_27():
    """Andrasfai graph A_9: 27 vertices, triangle-free, alpha=9."""
    n = 27
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _local_search(graph, deadline):
    """Try to grow the graph by adding vertices one at a time."""
    n = graph['n']
    adj = [set() for _ in range(n)]
    for u, v in graph['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def check_valid(adj, n):
        # Check triangle-free
        for i in range(n):
            neighbors = list(adj[i])
            for a_idx in range(len(neighbors)):
                for b_idx in range(a_idx + 1, len(neighbors)):
                    if neighbors[b_idx] in adj[neighbors[a_idx]]:
                        return False
        # Check no independent set of size 10
        if _has_independent_set(adj, n, 10, deadline):
            return False
        return True
    
    def _has_independent_set(adj, n, k, deadline):
        """Check if there's an independent set of size k using BFS."""
        if k <= 0:
            return True
        # Simple greedy + backtracking for small k
        # Use a recursive approach with pruning
        def _rec(candidates, remaining):
            if remaining == 0:
                return True
            if len(candidates) < remaining:
                return False
            if time.time() > deadline:
                return False
            # Greedy lower bound: if we can't pick enough, fail
            for v in candidates:
                new_cands = [u for u in candidates if u != v and v not in adj[u]]
                if _rec(new_cands, remaining - 1):
                    return True
            return False
        return _rec(list(range(n)), k)
    
    current_n = n
    current_adj = [set(s) for s in adj]
    current_edges = graph['edges'][:]
    
    # Try adding vertices
    while time.time() < deadline:
        new_v = current_n
        # Try all possible neighborhoods for the new vertex
        best_success = False
        for mask in range(1 << min(current_n, 20)):  # Limit search space
            if time.time() > deadline:
                break
            # This is too slow for large n; use a smarter approach
            break
        
        # Smarter approach: try random neighborhoods
        for _ in range(100):
            if time.time() > deadline:
                break
            # Random subset of neighbors
            subset = random.sample(range(current_n), random.randint(0, current_n))
            # Check if adding vertex with these neighbors is valid
            if not _try_add_vertex(current_adj, current_n, subset, deadline):
                continue
            # Check triangle-free: no two neighbors of new_v can be adjacent
            triangle_ok = True
            for a in subset:
                for b in subset:
                    if a != b and b in current_adj[a]:
                        triangle_ok = False
                        break
                if not triangle_ok:
                    break
            if not triangle_ok:
                continue
            # Check independent set condition
            new_adj = [set(s) for s in current_adj]
            new_adj.append(set(subset))
            for u in subset:
                new_adj[u].add(current_n)
            if _has_independent_set(new_adj, current_n + 1, 10, deadline):
                continue
            # Success!
            current_adj = new_adj
            current_n += 1
            # Rebuild edges
            edges = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    if j in current_adj[i]:
                        edges.append([i, j])
            current_edges = edges
            best_success = True
            break
        if not best_success:
            break
    
    return {'n': current_n, 'edges': current_edges}

def _try_add_vertex(adj, n, neighbors, deadline):
    """Check if adding a vertex with given neighbors preserves triangle-free property."""
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if neighbors[j] in adj[neighbors[i]]:
                return False
    return True
# EVOLVE-BLOCK-END
