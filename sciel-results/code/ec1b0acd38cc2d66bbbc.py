"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    n = 31
    target_edges = 91

    def build_polarity_graph():
        """Construct a polarity graph on PG(2,5) - C4-free with 90 edges."""
        q = 5
        points = []
        seen = set()
        for x0 in range(q):
            for x1 in range(q):
                for x2 in range(q):
                    if x0 == 0 and x1 == 0 and x2 == 0:
                        continue
                    if x0 != 0:
                        inv = pow(x0, q-2, q)
                        canon = (1, x1 * inv % q, x2 * inv % q)
                    elif x1 != 0:
                        inv = pow(x1, q-2, q)
                        canon = (0, 1, x2 * inv % q)
                    else:
                        canon = (0, 0, 1)
                    if canon not in seen:
                        seen.add(canon)
                        points.append(canon)
        
        assert len(points) == 31, f"Expected 31 points, got {len(points)}"
        
        def are_conjugate(p, r):
            return (p[0]*r[2] + p[1]*r[1] + p[2]*r[0]) % q == 0
        
        adj = [set() for _ in range(31)]
        edges = []
        for i in range(31):
            for j in range(i+1, 31):
                if are_conjugate(points[i], points[j]):
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        
        return edges, adj

    def count_c4(adj_sets):
        """Count 4-cycles: for each pair (u,v), C(|N(u)∩N(v)|, 2) / 2."""
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = len(adj_sets[u] & adj_sets[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    def c4_delta_add(adj_sets, a, b):
        """C4s created by adding edge (a,b)."""
        common = len(adj_sets[a] & adj_sets[b])
        return common * (common - 1) // 2

    def c4_delta_remove(adj_sets, a, b):
        """C4s destroyed by removing edge (a,b)."""
        common = len(adj_sets[a] & adj_sets[b])
        return common * (common - 1) // 2

    # Step 1: Build the polarity graph (C4-free, 90 edges)
    core_edges, core_adj = build_polarity_graph()
    
    if len(core_edges) != 90:
        core_edges = []
        core_adj = [set() for _ in range(n)]
        all_pairs = list(range(n*(n-1)//2))
        random.shuffle(all_pairs)
        for idx in all_pairs:
            if len(core_edges) >= 90:
                break
            u = idx // (n - 1)
            v = (idx % (n - 1)) + u + 1
            if u >= v:
                continue
            common = core_adj[u] & core_adj[v]
            if len(common) < 2:
                core_edges.append([u, v])
                core_adj[u].add(v)
                core_adj[v].add(u)
    
    # Step 2: Add the 91st edge to minimize total C4
    edge_set = set()
    for e in core_edges:
        edge_set.add(tuple(sorted(e)))
    
    best_add_c4 = float('inf')
    best_add_edge = None
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                common = core_adj[i] & core_adj[j]
                new_c4s = len(common) * (len(common) - 1) // 2
                if new_c4s < best_add_c4:
                    best_add_c4 = new_c4s
                    best_add_edge = (i, j)
    
    if best_add_edge is not None:
        core_edges.append(list(best_add_edge))
        core_adj[best_add_edge[0]].add(best_add_edge[1])
        core_adj[best_add_edge[1]].add(best_add_edge[0])
        edge_set.add(best_add_edge)
    
    current_c4 = count_c4(core_adj)
    
    # Step 3: Efficient local search with incremental C4 updates
    if time.time() < deadline - 5:
        adj = [set() for _ in range(n)]
        for e in core_edges:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
        
        edge_set = set(tuple(sorted(e)) for e in core_edges)
        
        # Build adjacency matrix for spectral analysis
        A = np.zeros((n, n), dtype=float)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    A[u, v] = 1
                    A[v, u] = 1
        
        def spectral_proxy(A_mat):
            """
            Use trace(A^4) as spectral proxy.
            trace(A^4) = 2*sum_v deg(v)^2 + 2*sum_{(u,v) in E} 2*|N(u)∩N(v)| + 8*C4
            So C4 = (trace(A^4) - 2*sum deg^2 - 4*sum_edges 2*common) / 8
            But simpler: just minimize trace(A^4) as a smooth surrogate.
            """
            A2 = A_mat @ A_mat
            trace_A4 = np.trace(A2 @ A2)
            return float(trace_A4)
        
        current_spec = spectral_proxy(A)
        
        # Precompute common neighbor counts for all pairs
        def compute_common_matrix(adj_sets):
            common = np.zeros((n, n), dtype=np.int32)
            for u in range(n):
                for v in adj_sets[u]:
                    common[u, v] = len(adj_sets[u] & adj_sets[v])
                    common[v, u] = common[u, v]
            return common
        
        common_mat = compute_common_matrix(adj)
        
        # Local search: try swaps (remove one edge, add one non-edge)
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and time.time() < deadline - 5 and iteration < max_iterations:
            improved = False
            iteration += 1
            
            edge_list = list(edge_set)
            random.shuffle(edge_list)
            
            for (u, v) in edge_list:
                if time.time() > deadline - 5:
                    break
                
                # Delta from removing (u,v)
                common_uv = common_mat[u, v]
                delta_remove = -common_uv * (common_uv - 1) // 2
                
                # Temporarily remove edge (u,v)
                adj[u].discard(v)
                adj[v].discard(u)
                edge_set.discard((u, v))
                A[u, v] = 0
                A[v, u] = 0
                common_mat[u, v] = 0
                common_mat[v, u] = 0
                
                # Update common_mat for affected vertices
                for x in adj[u]:
                    common_mat[u, x] = len(adj[u] & adj[x])
                    common_mat[x, u] = common_mat[u, x]
                for x in adj[v]:
                    common_mat[v, x] = len(adj[v] & adj[x])
                    common_mat[x, v] = common_mat[v, x]
                
                best_delta = 0
                best_new_edge = None
                best_spec = current_spec
                
                # Try adding each non-edge
                for a in range(n):
                    for b in range(a+1, n):
                        if (a, b) == (u, v):
                            continue
                        if (a, b) in edge_set:
                            continue
                        
                        common_ab = common_mat[a, b]
                        delta_add = common_ab * (common_ab - 1) // 2
                        total_delta = delta_remove + delta_add
                        
                        if total_delta < best_delta:
                            # Compute spectral proxy for this candidate
                            A[a, b] = 1
                            A[b, a] = 1
                            spec_candidate = spectral_proxy(A)
                            A[a, b] = 0
                            A[b, a] = 0
                            
                            if spec_candidate < best_spec or (spec_candidate == best_spec and total_delta < best_delta):
                                best_delta = total_delta
                                best_new_edge = (a, b)
                                best_spec = spec_candidate
                
                # Restore (u,v)
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                A[u, v] = 1
                A[v, u] = 1
                common_mat[u, v] = len(adj[u] & adj[v])
                common_mat[v, u] = common_mat[u, v]
                
                for x in adj[u]:
                    common_mat[u, x] = len(adj[u] & adj[x])
                    common_mat[x, u] = common_mat[u, x]
                for x in adj[v]:
                    common_mat[v, x] = len(adj[v] & adj[x])
                    common_mat[x, v] = common_mat[v, x]
                
                if best_new_edge is not None and best_delta < 0:
                    # Perform the swap
                    a, b = best_new_edge
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edge_set.discard((u, v))
                    adj[a].add(b)
                    adj[b].add(a)
                    edge_set.add((a, b))
                    
                    A[u, v] = 0
                    A[v, u] = 0
                    A[a, b] = 1
                    A[b, a] = 1
                    
                    # Update common_mat
                    common_mat[u, v] = 0
                    common_mat[v, u] = 0
                    common_mat[a, b] = len(adj[a] & adj[b])
                    common_mat[b, a] = common_mat[a, b]
                    
                    for x in adj[u]:
                        common_mat[u, x] = len(adj[u] & adj[x])
                        common_mat[x, u] = common_mat[u, x]
                    for x in adj[v]:
                        common_mat[v, x] = len(adj[v] & adj[x])
                        common_mat[x, v] = common_mat[v, x]
                    for x in adj[a]:
                        common_mat[a, x] = len(adj[a] & adj[x])
                        common_mat[x, a] = common_mat[a, x]
                    for x in adj[b]:
                        common_mat[b, x] = len(adj[b] & adj[x])
                        common_mat[x, b] = common_mat[b, x]
                    
                    current_c4 += best_delta
                    current_spec = best_spec
                    improved = True
                    break
        
        # Final verification
        current_c4 = count_c4(adj)
        core_edges = [list(sorted(e)) for e in edge_set]
    
    # Ensure exactly 91 distinct edges
    edge_set = set()
    unique_edges = []
    for e in core_edges:
        key = tuple(sorted(e))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(list(e))
    
    if len(unique_edges) > target_edges:
        unique_edges = unique_edges[:target_edges]
    
    if len(unique_edges) < target_edges:
        adj_temp = [set() for _ in range(n)]
        for e in unique_edges:
            adj_temp[e[0]].add(e[1])
            adj_temp[e[1]].add(e[0])
        for i in range(n):
            for j in range(i+1, n):
                if len(unique_edges) >= target_edges:
                    break
                if (i, j) not in edge_set:
                    unique_edges.append([i, j])
                    edge_set.add((i, j))
            if len(unique_edges) >= target_edges:
                break
    
    return unique_edges[:target_edges]
# EVOLVE-BLOCK-END
