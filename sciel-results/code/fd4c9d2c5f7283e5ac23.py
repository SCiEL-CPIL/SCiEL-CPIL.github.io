import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    n = 31
    target_edges = 91
    
    # PG(2,5) polarity: 31 vertices, 90 edges, C4-free
    # Generate all 31 points of PG(2,5)
    def gf5_mul(a, b):
        return (a * b) % 5
    
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)
                        vec = [gf5_mul(v, inv) for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: x·y = x0*y0 + x1*y1 + x2*y2 = 0 (mod 5)
    def is_polar(u, v):
        s = 0
        for i in range(3):
            s = (s + points[u][i] * points[v][i]) % 5
        return s == 0
    
    # Build the polarity graph (C4-free, 90 edges)
    polarity_edges = []
    for i in range(31):
        for j in range(i + 1, 31):
            if is_polar(i, j):
                polarity_edges.append((i, j))
    
    assert len(polarity_edges) == 90, f"Expected 90 polarity edges, got {len(polarity_edges)}"
    
    # Fast C4 count using vectorized numpy
    def count_c4_fast(adj_matrix):
        """Count 4-cycles: (1/2) * sum_{u<v} C(common(u,v), 2)"""
        n = adj_matrix.shape[0]
        c4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = int(np.bitwise_and(adj_matrix[u], adj_matrix[v]).sum())
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2
    
    # Delta C4 computation: compute change in C4 count when adding/removing edges
    def delta_c4_add_edge(adj, u, v):
        """Compute increase in C4 count when adding edge (u,v)"""
        common = np.where(adj[u] & adj[v])[0]
        count = 0
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                if adj[common[i], common[j]]:
                    count += 1
        return count
    
    def delta_c4_remove_edge(adj, u, v):
        """Compute decrease in C4 count when removing edge (u,v)"""
        # C4s containing edge (u,v): for each common neighbor w of u and v,
        # and each neighbor x of w that is also adjacent to both u and v... 
        # Actually: a C4 containing edge (u,v) has form u-w-x-v-u where w~x
        # So: for each w in N(u)∩N(v), for each x in N(w)∩N(u)∩N(v), x≠w
        # But we need to be careful: each C4 is counted twice (once per diagonal)
        # Let me think: C4 u-w-x-v-u. The edge (u,v) is in this C4.
        # Common neighbors of u and v: w and x. And w~x must hold.
        # So: number of C4s containing (u,v) = number of edges in the subgraph induced by N(u)∩N(v)
        common = np.where(adj[u] & adj[v])[0]
        count = 0
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                if adj[common[i], common[j]]:
                    count += 1
        return count
    
    # Build adjacency matrix for polarity graph
    adj = np.zeros((31, 31), dtype=np.uint8)
    for u, v in polarity_edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Find the best 91st edge to add (minimizes new C4s)
    best_edge = None
    best_c4s = float('inf')
    
    existing_edges = set()
    for u, v in polarity_edges:
        existing_edges.add((min(u,v), max(u,v)))
    
    for u in range(31):
        for v in range(u + 1, 31):
            if (u, v) in existing_edges:
                continue
            c4s = delta_c4_add_edge(adj, u, v)
            if c4s < best_c4s:
                best_c4s = c4s
                best_edge = (u, v)
    
    # Add the best edge
    base_edges = list(polarity_edges)
    if best_edge:
        base_edges.append(best_edge)
    
    def get_edge_set(edges):
        return set((min(u,v), max(u,v)) for u, v in edges)
    
    def build_adj(edges):
        a = np.zeros((31, 31), dtype=np.uint8)
        for u, v in edges:
            a[u, v] = 1
            a[v, u] = 1
        return a
    
    current_edges = base_edges[:]
    current_edge_set = get_edge_set(current_edges)
    current_adj = build_adj(current_edges)
    current_c4 = count_c4_fast(current_adj)
    
    best_edges = current_edges[:]
    best_c4 = current_c4
    
    # Local search with delta-based evaluation
    random.seed(42)
    
    def try_swaps(edges, edge_set, adj, c4_count, time_limit):
        """Try random edge swaps to reduce C4 count using delta computation"""
        best_local_edges = edges[:]
        best_local_set = edge_set.copy()
        best_local_adj = adj.copy()
        best_local_c4 = c4_count
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            # Pick a random edge to remove
            edge_idx = random.randint(0, len(best_local_edges) - 1)
            remove_u, remove_v = best_local_edges[edge_idx]
            
            # Try several random non-edges to add
            best_add = None
            best_delta = float('inf')
            
            for _ in range(20):
                add_u = random.randint(0, 30)
                add_v = random.randint(0, 30)
                if add_u == add_v:
                    continue
                key = (min(add_u, add_v), max(add_u, add_v))
                if key in best_local_set:
                    continue
                
                # Compute delta: remove (remove_u, remove_v), add (add_u, add_v)
                # Delta = -C4s_containing(remove) + C4s_created_by(add)
                # But these interact, so we need to be more careful
                
                # Simulate: compute C4s in modified graph
                new_adj = best_local_adj.copy()
                new_adj[remove_u, remove_v] = 0
                new_adj[remove_v, remove_u] = 0
                new_adj[add_u, add_v] = 1
                new_adj[add_v, add_u] = 1
                
                new_c4 = count_c4_fast(new_adj)
                delta = new_c4 - best_local_c4
                
                if delta < best_delta:
                    best_delta = delta
                    best_add = (add_u, add_v, key)
            
            if best_add is None:
                no_improve_count += 1
                continue
            
            add_u, add_v, key = best_add
            
            if best_delta < 0:
                # Accept the swap
                best_local_edges[edge_idx] = (add_u, add_v)
                best_local_set.discard((min(remove_u, remove_v), max(remove_u, remove_v)))
                best_local_set.add(key)
                best_local_adj[remove_u, remove_v] = 0
                best_local_adj[remove_v, remove_u] = 0
                best_local_adj[add_u, add_v] = 1
                best_local_adj[add_v, add_u] = 1
                best_local_c4 += best_delta
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return best_local_edges, best_local_set, best_local_adj, best_local_c4
    
    # Run local search with time budget
    if time.time() < deadline - 10:
        current_edges, current_edge_set, current_adj, current_c4 = try_swaps(
            current_edges, current_edge_set, current_adj, current_c4, deadline - 10
        )
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_edges = current_edges[:]
    
    # Try multiple restarts with different random seeds if time allows
    if time.time() < deadline - 15:
        for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 10:
                break
            random.seed(seed)
            restart_edges = list(polarity_edges) + [best_edge]
            restart_set = get_edge_set(restart_edges)
            restart_adj = build_adj(restart_edges)
            restart_c4 = count_c4_fast(restart_adj)
            
            restart_edges, restart_set, restart_adj, restart_c4 = try_swaps(
                restart_edges, restart_set, restart_adj, restart_c4, deadline - 10
            )
            
            if restart_c4 < best_c4:
                best_c4 = restart_c4
                best_edges = restart_edges[:]
    
    # Final verification
    final_edge_set = get_edge_set(best_edges)
    assert len(final_edge_set) == 91, f"Expected 91 edges, got {len(final_edge_set)}"
    
    result = [[u, v] for u, v in sorted(final_edge_set)]
    
    assert len(result) == 91
    assert all(0 <= u < 31 and 0 <= v < 31 for u, v in result)
    assert all(u < v for u, v in result)
    
    return result
