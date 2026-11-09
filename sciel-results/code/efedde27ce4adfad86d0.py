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

    # ---- Vectorized NumPy approach ----
    def compute_common(adj_mat):
        """Compute common neighbor counts from adjacency matrix."""
        common = np.dot(adj_mat.astype(np.int32), adj_mat.astype(np.int32))
        np.fill_diagonal(common, 0)
        return common

    def greedy_build_np(seed_rng, order='random'):
        """Build K2,3-free graph using vectorized operations."""
        adj_mat = np.zeros((n, n), dtype=np.int8)
        common = np.zeros((n, n), dtype=np.int32)
        
        edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
        
        if order == 'random':
            seed_rng.shuffle(edges_to_try)
        elif order == 'degree':
            # Sort by sum of vertex indices for diversity
            seed_rng.shuffle(edges_to_try)
        elif order == 'reverse':
            edges_to_try.reverse()
            seed_rng.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if adj_mat[u, v] == 1:
                continue
            
            adj_u = np.where(adj_mat[u] == 1)[0]
            adj_v = np.where(adj_mat[v] == 1)[0]
            
            valid = True
            if len(adj_u) > 0:
                if np.any(common[adj_u, v] > 1):
                    valid = False
            if valid and len(adj_v) > 0:
                if np.any(common[u, adj_v] > 1):
                    valid = False
            
            if valid:
                adj_mat[u, v] = 1
                adj_mat[v, u] = 1
                
                for w in adj_u:
                    common[w, v] += 1
                    common[v, w] += 1
                for w in adj_v:
                    common[u, w] += 1
                    common[w, u] += 1
        
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj_mat[u, v] == 1:
                    edges.append([u, v])
        return edges

    def local_search_np(edges, max_iters=500, use_swaps=True):
        """Enhanced local search: add all valid edges, then try swaps."""
        adj_mat = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj_mat[u, v] = 1
            adj_mat[v, u] = 1
        
        common = compute_common(adj_mat)
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 10.0:
            improved = False
            iters += 1
            
            # Phase 1: Find ALL valid edge additions and add them all
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj_mat[u, v] == 0:
                        adj_u = np.where(adj_mat[u] == 1)[0]
                        adj_v = np.where(adj_mat[v] == 1)[0]
                        
                        valid = True
                        if len(adj_u) > 0:
                            if np.any(common[adj_u, v] > 1):
                                valid = False
                        if valid and len(adj_v) > 0:
                            if np.any(common[u, adj_v] > 1):
                                valid = False
                        
                        if valid:
                            candidates.append((u, v))
            
            if candidates:
                # Add ALL valid candidates (they're all independently valid)
                for u, v in candidates:
                    adj_mat[u, v] = 1
                    adj_mat[v, u] = 1
                    adj_u = np.where(adj_mat[u] == 1)[0]
                    adj_v = np.where(adj_mat[v] == 1)[0]
                    for w in adj_u:
                        if w != v:
                            common[w, v] += 1
                            common[v, w] += 1
                    for w in adj_v:
                        if w != u:
                            common[u, w] += 1
                            common[w, u] += 1
                improved = True
            elif use_swaps:
                # Phase 2: No additions possible, try edge swaps
                edge_list = [(u, v) for u in range(n) for v in range(u+1, n) if adj_mat[u, v] == 1]
                if not edge_list:
                    break
                
                # Try swapping: remove one edge, see if we can add two new ones
                swapped = False
                for rem_u, rem_v in edge_list:
                    if time.time() > deadline - 12.0:
                        break
                    
                    # Temporarily remove edge
                    adj_mat[rem_u, rem_v] = 0
                    adj_mat[rem_v, rem_u] = 0
                    adj_u = np.where(adj_mat[rem_u] == 1)[0]
                    adj_v = np.where(adj_mat[rem_v] == 1)[0]
                    for w in adj_u:
                        common[w, rem_v] -= 1
                        common[rem_v, w] -= 1
                    for w in adj_v:
                        common[rem_u, w] -= 1
                        common[w, rem_u] -= 1
                    
                    # Find all addable edges
                    new_candidates = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if adj_mat[u, v] == 0:
                                a_u = np.where(adj_mat[u] == 1)[0]
                                a_v = np.where(adj_mat[v] == 1)[0]
                                valid = True
                                if len(a_u) > 0 and np.any(common[a_u, v] > 1):
                                    valid = False
                                if valid and len(a_v) > 0 and np.any(common[u, a_v] > 1):
                                    valid = False
                                if valid:
                                    new_candidates.append((u, v))
                    
                    # If we can add at least 2 edges, do the swap
                    if len(new_candidates) >= 2:
                        # Add up to 2 new edges
                        for u, v in new_candidates[:2]:
                            adj_mat[u, v] = 1
                            adj_mat[v, u] = 1
                            a_u = np.where(adj_mat[u] == 1)[0]
                            a_v = np.where(adj_mat[v] == 1)[0]
                            for w in a_u:
                                if w != v:
                                    common[w, v] += 1
                                    common[v, w] += 1
                            for w in a_v:
                                if w != u:
                                    common[u, w] += 1
                                    common[w, u] += 1
                        improved = True
                        swapped = True
                    else:
                        # Revert the removal
                        adj_mat[rem_u, rem_v] = 1
                        adj_mat[rem_v, rem_u] = 1
                        adj_u = np.where(adj_mat[rem_u] == 1)[0]
                        adj_v = np.where(adj_mat[rem_v] == 1)[0]
                        for w in adj_u:
                            if w != rem_v:
                                common[w, rem_v] += 1
                                common[rem_v, w] += 1
                        for w in adj_v:
                            if w != rem_u:
                                common[rem_u, w] += 1
                                common[w, rem_u] += 1
                    
                    if swapped:
                        break
                
                if not swapped and not new_candidates:
                    # Try removing a single random edge to escape
                    if len(edge_list) > 15:
                        rem_u, rem_v = rng.choice(edge_list)
                        adj_mat[rem_u, rem_v] = 0
                        adj_mat[rem_v, rem_u] = 0
                        adj_u = np.where(adj_mat[rem_u] == 1)[0]
                        adj_v = np.where(adj_mat[rem_v] == 1)[0]
                        for w in adj_u:
                            common[w, rem_v] -= 1
                            common[rem_v, w] -= 1
                        for w in adj_v:
                            common[rem_u, w] -= 1
                            common[w, rem_u] -= 1
                        improved = True
        
        result = []
        for u in range(n):
            for v in range(u+1, n):
                if adj_mat[u, v] == 1:
                    result.append([u, v])
        return result

    # Run multiple greedy builds with different seeds and strategies
    orders = ['random', 'random', 'reverse', 'random']
    for trial in range(40):
        if time.time() > deadline - 15.0:
            break
        
        trial_rng = random.Random(trial * 12345 + 6789)
        order = orders[trial % len(orders)]
        edges = greedy_build_np(trial_rng, order=order)
        edges = verify(edges)
        
        if edges is not None:
            improved = local_search_np(edges, max_iters=300, use_swaps=True)
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
