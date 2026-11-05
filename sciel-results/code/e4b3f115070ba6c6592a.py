"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _score(graph_edges, n=16):
    """Compute the largest triangle degree sum."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in graph_edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    max_sum = 0
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u] & adj[v]:
                if w > v:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _simulated_annealing(n=16, target_edges=65, max_iters=50000, time_limit=100, seed=None):
    """
    Simulated annealing to minimize max triangle degree sum.
    Uses edge swap moves (remove one edge, add another).
    """
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    
    # Initialize with a near-Turán structure: K_{8,8} + one internal edge
    # But randomize which internal edge and which partition
    part_a = list(range(n // 2))
    part_b = list(range(n // 2, n))
    
    # Randomize the partition
    all_vertices = list(range(n))
    random.shuffle(all_vertices)
    part_a = all_vertices[:n // 2]
    part_b = all_vertices[n // 2:]
    
    # Build K_{8,8} edges
    edges = []
    for u in part_a:
        for v in part_b:
            edges.append([min(u, v), max(u, v)])
    
    # Add one internal edge within part_a
    random.shuffle(part_a)
    internal_edge = [part_a[0], part_a[1]]
    internal_edge = [min(internal_edge[0], internal_edge[1]), max(internal_edge[0], internal_edge[1])]
    edges.append(internal_edge)
    
    edge_set = set(map(tuple, edges))
    current_score = _score(edges, n)
    best_edges = [e[:] for e in edges]
    best_score = current_score
    
    # Temperature schedule
    T0 = 10.0
    T_final = 0.1
    cooling_rate = (T_final / T0) ** (1.0 / max(1, max_iters))
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        T = T0 * (cooling_rate ** it)
        
        # Pick a random edge to remove
        if not edges:
            break
        idx = random.randint(0, len(edges) - 1)
        u, v = edges[idx]
        removed = (u, v)
        
        # Find a non-edge to add
        # Try a few random non-edges
        found_swap = False
        for _ in range(30):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b:
                continue
            a, b = min(a, b), max(a, b)
            if (a, b) in edge_set:
                continue
            if (a, b) == removed:
                continue
            
            # Try the swap
            new_edge_set = set(edge_set)
            new_edge_set.remove(removed)
            new_edge_set.add((a, b))
            new_edges = [list(e) for e in new_edge_set]
            
            new_score = _score(new_edges, n)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                # Accept
                edge_set = new_edge_set
                edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [e[:] for e in edges]
                found_swap = True
                break
        
        if not found_swap:
            # If no swap was accepted, try a different strategy: 
            # remove a random edge and add a random non-edge without scoring first
            # This helps escape local minima
            pass
    
    return best_edges, best_score

def _construct_near_turan_with_optimization(n=16, target_edges=65):
    """
    Start from multiple near-Turán configurations and optimize each.
    """
    best_edges = None
    best_score = float('inf')
    
    # Try different random seeds for the partition and internal edge
    for seed in range(20):
        try:
            edges_opt, score_opt = _simulated_annealing(n, target_edges, max_iters=30000, time_limit=4, seed=seed)
            if score_opt < best_score:
                best_score = score_opt
                best_edges = [e[:] for e in edges_opt]
        except Exception:
            continue
    
    return best_edges, best_score

def construct():
    """
    Construct a graph on 16 vertices with 65 edges minimizing the max triangle degree sum.
    """
    n = 16
    target_edges = 65
    
    best_score = float('inf')
    best_edges = None
    
    # First try the baseline: K_{8,8} + one internal edge
    edges_seed = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    score_seed = _score(edges_seed, n)
    if score_seed < best_score:
        best_score = score_seed
        best_edges = [e[:] for e in edges_seed]
    
    # Now use simulated annealing from multiple starts
    edges_opt, score_opt = _construct_near_turan_with_optimization(n, target_edges)
    if edges_opt is not None and score_opt < best_score:
        best_score = score_opt
        best_edges = [e[:] for e in edges_opt]
    
    # Final fallback
    if best_edges is None:
        best_edges = edges_seed
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
