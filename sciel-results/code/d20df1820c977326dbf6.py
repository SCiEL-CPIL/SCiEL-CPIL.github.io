"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _score_graph(n, edges):
    """Compute max triangle degree sum for a given edge list."""
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    deg = adj.sum(axis=1)
    # Find all triangles
    max_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                # Common neighbors of i and j
                common = np.where(adj[i, :] & adj[j, :])[0]
                for k in common:
                    if k > j:  # avoid duplicates
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
    return max_sum

def _max_eigenvalue(adj):
    """Compute largest eigenvalue via power iteration."""
    n = adj.shape[0]
    v = np.ones(n) / np.sqrt(n)
    for _ in range(50):
        v_new = adj @ v
        norm = np.linalg.norm(v_new)
        if norm == 0:
            return 0
        v_new /= norm
        if np.linalg.norm(v_new - v) < 1e-10:
            break
        v = v_new
    return float(v @ (adj @ v))

def _spectral_coordinate_descent(n, edges, deadline, max_iter=5000):
    """
    Use spectral proxy + coordinate descent to minimize max triangle degree sum.
    Start from a good initial graph, then try edge swaps that reduce the spectral proxy.
    """
    start_time = time.time()
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    def get_edge_set():
        return {(min(u,v), max(u,v)) for u, v in edges}
    
    current_edges = list(edges)
    current_score = _score_graph(n, current_edges)
    best_edges = list(current_edges)
    best_score = current_score
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set((min(u,v), max(u,v)) for u, v in current_edges)
    
    for it in range(max_iter):
        if time.time() - start_time > deadline - 5:
            break
        
        # Try random edge swap: remove one edge, add another
        if len(current_edges) < 2:
            break
        
        # Pick a random edge to remove
        idx = np.random.randint(len(current_edges))
        u1, v1 = current_edges[idx]
        e1 = (min(u1, v1), max(u1, v1))
        
        # Pick a random edge to add (not already present)
        attempts = 0
        while attempts < 100:
            u2, v2 = all_edges[np.random.randint(len(all_edges))]
            e2 = (min(u2, v2), max(u2, v2))
            if e2 not in edge_set and e2 != e1:
                break
            attempts += 1
        else:
            continue
        
        # Apply swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        edge_set.remove(e1)
        edge_set.add(e2)
        current_edges[idx] = [u2, v2]
        
        # Compute new score
        new_score = _score_graph(n, current_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = list(current_edges)
        elif new_score > current_score:
            # Revert
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            edge_set.remove(e2)
            edge_set.add(e1)
            current_edges[idx] = [u1, v1]
        else:
            # Keep if same or better
            pass
    
    return best_edges, best_score

def _polish_local_search(n, edges, deadline):
    """High-precision local polish: try all single edge swaps that improve score."""
    start_time = time.time()
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    current_edges = list(edges)
    current_score = _score_graph(n, current_edges)
    best_edges = list(current_edges)
    best_score = current_score
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set((min(u,v), max(u,v)) for u, v in current_edges)
    
    improved = True
    while improved and time.time() - start_time < deadline - 3:
        improved = False
        for i in range(len(current_edges)):
            if time.time() - start_time > deadline - 3:
                break
            u1, v1 = current_edges[i]
            e1 = (min(u1, v1), max(u1, v1))
            for u2, v2 in all_edges:
                e2 = (min(u2, v2), max(u2, v2))
                if e2 in edge_set or e2 == e1:
                    continue
                
                # Try swap
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                edge_set.remove(e1)
                edge_set.add(e2)
                current_edges[i] = [u2, v2]
                
                new_score = _score_graph(n, current_edges)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = list(current_edges)
                    improved = True
                else:
                    # Revert
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    edge_set.remove(e2)
                    edge_set.add(e1)
                    current_edges[i] = [u1, v1]
    
    return best_edges, best_score

def construct():
    n = 16
    deadline = time.time() + 140  # 140s budget, 10s safety margin
    
    # Initial graph: K_{6,10} plus 5 disjoint edges in size-10 part
    edges = [[u, v] for u in range(6) for v in range(6, 16)]
    edges += [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    
    # Phase 1: Spectral-guided coordinate descent
    edges, score = _spectral_coordinate_descent(n, edges, deadline, max_iter=3000)
    
    # Phase 2: Local polish with exhaustive single-swap search
    edges, score = _polish_local_search(n, edges, deadline)
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
