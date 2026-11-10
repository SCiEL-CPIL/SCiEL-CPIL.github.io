import time
import random
from itertools import combinations
from math import comb

START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # safety margin before 150s kill

def is_regular_subgraph(subset, adj_matrix, n):
    """Check if the induced subgraph on `subset` is regular.
    A regular subgraph means every vertex in the subset has the same degree
    within the subset (counting only edges to other vertices in the subset).
    """
    k = len(subset)
    if k < 2:
        # Single vertex: degree 0, trivially regular
        return True
    
    # Compute degrees of each vertex in the subset
    degrees = []
    for u in subset:
        deg = 0
        for v in subset:
            if u != v and adj_matrix[u][v]:
                deg += 1
        degrees.append(deg)
    
    # Check if all degrees are equal
    first_deg = degrees[0]
    for d in degrees[1:]:
        if d != first_deg:
            return False
    return True

def has_regular_subgraph_of_size_at_least_6(n, adj_matrix):
    """Check if there exists any subset of size >= 6 that induces a regular subgraph."""
    # We only need to check subsets of exactly size 6 (or more).
    # But checking all subsets of size >= 6 is expensive.
    # Key insight: if a subset of size m >= 6 induces a regular subgraph,
    # we can check subsets of size 6 first. But actually, a regular subgraph
    # of size m > 6 might exist without any size-6 regular subgraph within it.
    # However, the problem says "No set of 6 or more vertices may induce a regular subgraph."
    # So we need to check ALL subsets of size >= 6.
    # For efficiency, we'll check sizes 6, 7, ..., n.
    
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if is_regular_subgraph(subset, adj_matrix, n):
                return True
    return False

def validate_graph(n, edges):
    """Validate that the graph is simple and has no regular induced subgraph on >= 6 vertices."""
    # Build adjacency matrix
    adj_matrix = [[0] * n for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False  # no self-loops
        if not (0 <= u < n and 0 <= v < n):
            return False
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1
    
    # Check for duplicate edges
    edge_set = set()
    for u, v in edges:
        pair = (min(u, v), max(u, v))
        if pair in edge_set:
            return False  # duplicate edge
        edge_set.add(pair)
    
    # Check for regular subgraph of size >= 6
    if has_regular_subgraph_of_size_at_least_6(n, adj_matrix):
        return False
    
    return True

def construct():
    # We want to maximize n such that there exists a graph on n vertices
    # with no regular induced subgraph on 6 or more vertices.
    
    # Known: graphs with 20 vertices exist. Let's try to find one.
    # Strategy: Start with n=20 and try random graphs, or use a constructive approach.
    
    # A simple approach: try random graphs for n=20, 21, etc.
    # But validation is expensive for large n. Let's be smart.
    
    # For n=20, the number of subsets of size 6 is comb(20,6) = 38760, which is manageable.
    # For n=21, comb(21,6) = 54264, still manageable.
    # For n=22, comb(22,6) = 74613, still OK.
    
    # Let's try n=22 first, then 21, then 20.
    
    for n in [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > DEADLINE - 10:
            break
        
        # Try random graphs
        max_attempts = 50 if n <= 20 else 20
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 5:
                break
            
            # Generate a random graph
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        edges.append([u, v])
            
            # Validate
            if validate_graph(n, edges):
                return {'n': n, 'edges': edges}
        
        # If random didn't work, try a structured approach for smaller n
        # For n <= 15, we can do a more exhaustive search if needed
        if n <= 15:
            # Try complement of random graph
            for attempt in range(20):
                if time.time() > DEADLINE - 5:
                    break
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            edges.append([u, v])
                if validate_graph(n, edges):
                    return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
