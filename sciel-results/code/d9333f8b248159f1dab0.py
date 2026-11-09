import random
import time
import sys
import numpy as np

def construct():
    n = 20
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s limit
    rng = random.Random(42)

    def verify(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if v in adj[u]:
                return None  # duplicate
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    return None
        return edges

    # Fallback: 5 disjoint K4s (10 edges)
    def fallback():
        e = []
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    e.append([base + a, base + b])
        return e

    best = None
    best_e = 0

    def numpy_greedy_build(seed_rng):
        """Build a K2,3-free graph using vectorized NumPy common-neighbor checks."""
        # adj is n x n boolean matrix
        adj = np.zeros((n, n), dtype=np.uint8)
        # common[u][v] = number of common neighbors of u and v
        common = np.zeros((n, n), dtype=np.int16)
        
        # Precompute all edge pairs
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        seed_rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if adj[u, v]:
                continue
            
            # Check if adding edge (u, v) violates K2,3
            # For each neighbor w of u: common[w][v] + 1 <= 2, i.e., common[w][v] <= 1
            # For each neighbor w of v: common[u][w] + 1 <= 2, i.e., common[u][w] <= 1
            
            # Get neighbors of u and v
            neighbors_u = np.where(adj[u] == 1)[0]
            neighbors_v = np.where(adj[v] == 1)[0]
            
            # Check: for all w in neighbors_u, common[w][v] <= 1
            if len(neighbors_u) > 0:
                if np.max(common[neighbors_u, v]) > 1:
                    continue
            
            # Check: for all w in neighbors_v, common[u][w] <= 1
            if len(neighbors_v) > 0:
                if np.max(common[u, neighbors_v]) > 1:
                    continue
            
            # Add the edge
            adj[u, v] = 1
            adj[v, u] = 1
            
            # Update common neighbor counts
            # For each w in neighbors_u (excluding v if present), common[w][v] += 1
            for w in neighbors_u:
                if w != v:
                    common[w, v] += 1
                    common[v, w] += 1
            
            # For each w in neighbors_v (excluding u if present), common[u][w] += 1
            for w in neighbors_v:
                if w != u:
                    common[u, w] += 1
                    common[w, u] += 1
        
        # Extract edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges

    def numpy_local_search(edges, max_iters=200):
        """Local search using NumPy for faster common-neighbor checks."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Compute common neighbor counts
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        improved = True
        iters = 0
        current_edge_count = len(edges)
        
        while improved and iters < max_iters and time.time() < deadline - 10.0:
            improved = False
            iters += 1
            
            # Find all non-edges
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            rng.shuffle(non_edges)
            
            for u, v in non_edges:
                # Check if adding (u, v) is valid
                neighbors_u = np.where(adj[u] == 1)[0]
                neighbors_v = np.where(adj[v] == 1)[0]
                
                valid = True
                if len(neighbors_u) > 0:
                    if np.max(common[neighbors_u, v]) > 1:
                        valid = False
                if valid and len(neighbors_v) > 0:
                    if np.max(common[u, neighbors_v]) > 1:
                        valid = False
                
                if valid:
                    # Add edge
                    adj[u, v] = 1
                    adj[v, u] = 1
                    
                    # Update common counts
                    for w in neighbors_u:
                        if w != v:
                            common[w, v] += 1
                            common[v, w] += 1
                    for w in neighbors_v:
                        if w != u:
                            common[u, w] += 1
                            common[w, u] += 1
                    
                    improved = True
                    break
            
            # If no edge could be added, try removing a random edge
            if not improved and current_edge_count > 20:
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                if edge_list:
                    u, v = rng.choice(edge_list)
                    # Remove edge
                    neighbors_u = np.where(adj[u] == 1)[0]
                    neighbors_v = np.where(adj[v] == 1)[0]
                    
                    adj[u, v] = 0
                    adj[v, u] = 0
                    
                    # Update common counts
                    for w in neighbors_u:
                        if w != v:
                            common[w, v] -= 1
                            common[v, w] -= 1
                    for w in neighbors_v:
                        if w != u:
                            common[u, w] -= 1
                            common[w, u] -= 1
                    
                    current_edge_count -= 1
                    improved = True
        
        # Extract edges
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    # Run multiple greedy builds with different seeds
    for trial in range(30):
        if time.time() > deadline - 15.0:
            break
        
        trial_rng = random.Random(trial * 12345 + 6789)
        edges = numpy_greedy_build(trial_rng)
        edges = verify(edges)
        
        if edges is not None:
            # Try local search to improve
            improved = numpy_local_search(edges, max_iters=150)
            improved = verify(improved)
            
            if improved is not None and len(improved) > best_e:
                best_e = len(improved)
                best = improved
            elif edges is not None and len(edges) > best_e:
                best_e = len(edges)
                best = edges
    
    # Final verification
    if best is not None:
        best = verify(best)
    
    if best is None:
        return fallback()
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
