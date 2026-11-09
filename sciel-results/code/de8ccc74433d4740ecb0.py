"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _build_adj_and_deg(n, edges):
    """Build adjacency matrix and degree list."""
    adj = [[0] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    deg = [sum(adj[i]) for i in range(n)]
    return adj, deg

def _find_triangles(n, adj):
    """Find all triangles in the graph."""
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                continue
            for w in range(v + 1, n):
                if adj[u][w] and adj[v][w]:
                    triangles.append((u, v, w))
    return triangles

def _score_graph(n, edges):
    """Compute the max degree sum over all triangles."""
    adj, deg = _build_adj_and_deg(n, edges)
    triangles = _find_triangles(n, adj)
    if not triangles:
        return 0
    best = 0
    for u, v, w in triangles:
        s = deg[u] + deg[v] + deg[w]
        if s > best:
            best = s
    return best

def _qubo_objective(n, edges):
    """
    Compute a QUBO-like objective: sum of squared triangle degree sums.
    This encourages uniform low-degree distribution across triangles.
    Also add a penalty for high max degree to keep degrees balanced.
    """
    adj, deg = _build_adj_and_deg(n, edges)
    triangles = _find_triangles(n, adj)
    if not triangles:
        return 0.0
    
    total_sq = 0.0
    max_sum = 0
    for u, v, w in triangles:
        s = deg[u] + deg[v] + deg[w]
        total_sq += s * s
        if s > max_sum:
            max_sum = s
    
    # Primary: minimize sum of squares (spreads degree sums evenly)
    # Secondary: small penalty on max to avoid extreme cases
    obj = total_sq / len(triangles) + 0.1 * max_sum
    return obj

def construct():
    n = 16
    deadline = time.time() + 115  # Safety margin
    
    # Start with K_{8,8} + one intra-part edge
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((min(u, v), max(u, v)))
    edges.add((0, 1))
    
    best_edges = set(edges)
    best_score = _score_graph(n, list(best_edges))
    best_qubo = _qubo_objective(n, list(best_edges))
    
    # All possible pairs
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    random.seed(42)
    
    # Simulated annealing with QUBO objective
    T0 = 5.0
    T_end = 0.01
    cooling_rate = 0.9999
    
    T = T0
    current_edges = set(best_edges)
    current_qubo = best_qubo
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5:
        # Generate a random swap
        removed = random.choice(tuple(current_edges))
        non_edges = all_pairs - current_edges
        if not non_edges:
            continue
        added = random.choice(tuple(non_edges))
        
        new_edges = current_edges - {removed} | {added}
        new_qubo = _qubo_objective(n, list(new_edges))
        
        delta_qubo = new_qubo - current_qubo
        
        # Metropolis acceptance criterion
        if delta_qubo < 0 or (T > 0 and random.random() < math.exp(-delta_qubo / T)):
            current_edges = new_edges
            current_qubo = new_qubo
            
            # Track best by actual score
            new_score = _score_graph(n, list(new_edges))
            if new_score < best_score:
                best_score = new_score
                best_edges = set(new_edges)
                best_qubo = new_qubo
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        # If stuck for too long, restart temperature
        if no_improve_count > max_no_improve:
            T = T0
            no_improve_count = 0
    
    # Final local search pass on best_edges to polish
    for it in range(50000):
        if time.time() > deadline - 2:
            break
        
        removed = random.choice(tuple(best_edges))
        non_edges = all_pairs - best_edges
        if not non_edges:
            continue
        added = random.choice(tuple(non_edges))
        
        new_edges = best_edges - {removed} | {added}
        new_score = _score_graph(n, list(new_edges))
        
        if new_score < best_score:
            best_score = new_score
            best_edges = new_edges
    
    # Convert to list format
    edge_list = [[u, v] for u, v in sorted(best_edges)]
    return {'n': n, 'edges': edge_list}
# EVOLVE-BLOCK-END
