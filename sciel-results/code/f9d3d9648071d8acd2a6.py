import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, try to quickly find a valid n=20 graph with local search
    for n in [20, 19, 18]:
        if time.time() > deadline - 30:
            break
        result = _local_search(n, start_time, deadline - 30)
        if result is not None:
            return result
    
    # Fallback: return n=5 (trivially valid)
    return {'n': 5, 'edges': []}

def _local_search(n, start_time, deadline):
    """Local search with simulated annealing for finding valid graph."""
    best_fitness = float('inf')
    best_graph = None
    
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Initialize random graph
        adj = _random_graph(n)
        fitness = _fitness(adj, n)
        
        if fitness == 0:
            return _to_result(adj, n)
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_graph = adj[:]
        
        # Simulated annealing
        current_temp = 100.0
        cooling = 0.995
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge
            _flip_edge(adj, u, v)
            new_fitness = _fitness(adj, n)
            
            if new_fitness == 0:
                return _to_result(adj, n)
            
            delta = new_fitness - fitness
            if delta < 0 or random.random() < np.exp(-delta / current_temp):
                fitness = new_fitness
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_graph = adj[:]
            else:
                _flip_edge(adj, u, v)  # revert
            
            current_temp *= cooling
    
    if best_graph is not None and _fitness(best_graph, n) == 0:
        return _to_result(best_graph, n)
    return None

def _random_graph(n):
    """Create a random graph with n vertices."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _flip_edge(adj, u, v):
    """Toggle the edge between u and v."""
    if (adj[u] >> v) & 1:
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    else:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

def _fitness(adj, n):
    """Count number of 6-vertex subsets that induce a regular graph."""
    count = 0
    for S in combinations(range(n), 6):
        if _is_regular(adj, S):
            count += 1
    return count

def _is_regular(adj, S):
    """Check if subset S induces a regular graph."""
    mask = 0
    for v in S:
        mask |= (1 << v)
    
    first_deg = None
    for v in S:
        deg = (adj[v] & mask).bit_count()
        if first_deg is None:
            first_deg = deg
        elif deg != first_deg:
            return False
    return True

def _to_result(adj, n):
    """Convert adjacency list to result format."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
