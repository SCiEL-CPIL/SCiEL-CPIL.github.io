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

def _soft_max_objective(n, edges, beta=1.0):
    """
    Smooth surrogate for max triangle degree sum using log-sum-exp.
    As beta -> infinity, this approaches the true max.
    For finite beta, it's differentiable and guides the search.
    """
    adj, deg = _build_adj_and_deg(n, edges)
    triangles = _find_triangles(n, adj)
    if not triangles:
        return 0.0
    
    # Collect all triangle degree sums
    sums = []
    for u, v, w in triangles:
        s = deg[u] + deg[v] + deg[w]
        sums.append(s)
    
    if not sums:
        return 0.0
    
    # Log-sum-exp for numerical stability
    max_s = max(sums)
    if max_s == 0:
        return 0.0
    
    # soft_max = (1/beta) * log(sum(exp(beta * s_i)))
    # = max_s + (1/beta) * log(sum(exp(beta * (s_i - max_s))))
    log_terms = [math.exp(beta * (s - max_s)) for s in sums]
    log_sum = sum(log_terms)
    if log_sum <= 0:
        return max_s
    soft_max = max_s + (1.0 / beta) * math.log(log_sum)
    
    # Also add a small penalty for variance to encourage uniformity
    mean_s = sum(sums) / len(sums)
    variance = sum((s - mean_s) ** 2 for s in sums) / len(sums)
    
    return soft_max + 0.05 * variance

def _local_search_polish(n, edges, deadline, max_iter=100000):
    """High-precision local search to polish the best solution."""
    best_edges = set(edges)
    best_score = _score_graph(n, list(best_edges))
    
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    non_edges = all_pairs - best_edges
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try a random swap
        if not best_edges or not non_edges:
            break
        
        removed = random.choice(tuple(best_edges))
        added = random.choice(tuple(non_edges))
        
        new_edges = best_edges - {removed} | {added}
        new_score = _score_graph(n, list(new_edges))
        
        if new_score < best_score:
            best_score = new_score
            best_edges = new_edges
            non_edges = all_pairs - best_edges
    
    return best_edges, best_score

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    random.seed(42)
    
    # All possible pairs
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    # Start with K_{8,8} + one intra-part edge
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((min(u, v), max(u, v)))
    edges.add((0, 1))
    
    best_edges = set(edges)
    best_score = _score_graph(n, list(best_edges))
    best_obj = _soft_max_objective(n, list(best_edges), beta=2.0)
    
    # Simulated annealing with soft-max objective
    T0 = 10.0
    T_end = 0.01
    cooling_rate = 0.99995
    
    T = T0
    current_edges = set(best_edges)
    current_obj = best_obj
    
    no_improve_count = 0
    max_no_improve = 3000
    
    # Track beta for annealing: start with smaller beta (smoother), increase over time
    beta = 1.0
    beta_max = 5.0
    
    while time.time() < deadline - 10:
        # Generate a random swap
        removed = random.choice(tuple(current_edges))
        non_edges = all_pairs - current_edges
        if not non_edges:
            continue
        added = random.choice(tuple(non_edges))
        
        new_edges = current_edges - {removed} | {added}
        new_obj = _soft_max_objective(n, list(new_edges), beta=beta)
        
        delta_obj = new_obj - current_obj
        
        # Metropolis acceptance criterion
        if delta_obj < 0 or (T > 0 and random.random() < math.exp(-delta_obj / T)):
            current_edges = new_edges
            current_obj = new_obj
            
            # Track best by actual score
            new_score = _score_graph(n, list(new_edges))
            if new_score < best_score:
                best_score = new_score
                best_edges = set(new_edges)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        # Gradually increase beta to make objective sharper
        if beta < beta_max:
            beta = min(beta_max, beta * 1.0001)
        
        # If stuck for too long, restart temperature and reset beta
        if no_improve_count > max_no_improve:
            T = T0
            beta = 1.0
            no_improve_count = 0
    
    # Final high-precision polish pass
    polish_deadline = deadline - 2
    best_edges, best_score = _local_search_polish(n, best_edges, polish_deadline, max_iter=200000)
    
    # Convert to list format
    edge_list = [[u, v] for u, v in sorted(best_edges)]
    return {'n': n, 'edges': edge_list}
# EVOLVE-BLOCK-END
