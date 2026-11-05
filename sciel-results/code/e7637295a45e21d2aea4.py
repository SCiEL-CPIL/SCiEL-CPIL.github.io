import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 147.0  # safety margin before 150s kill

    n = 31
    m_target = 91
    avg_degree = 2 * m_target / n  # ~5.87

    # Spectral relaxation initialization:
    # Start with a regular graph of degree floor(avg_degree), then add remaining edges
    # to minimize C4s. Use a structured approach based on spectral properties.
    
    # First, create a base graph using a near-regular construction
    # We'll use a greedy approach to build a graph that minimizes C4s
    # Start with a random regular graph and optimize
    
    np.random.seed(42)
    
    # Create adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    edges = []
    edge_set = set()
    
    # Step 1: Build a base graph with target degree using greedy C4-minimization
    # For each vertex, connect to vertices that minimize added C4s
    degrees = np.zeros(n, dtype=np.int32)
    target_degrees = np.full(n, int(avg_degree), dtype=np.int32)
    # Distribute extra edges
    extra = m_target * 2 - n * int(avg_degree)
    for i in range(extra):
        target_degrees[i] += 1
    
    # Greedy edge addition minimizing C4s
    for u in range(n):
        for v in range(u+1, n):
            if degrees[u] >= target_degrees[u] or degrees[v] >= target_degrees[v]:
                continue
            # Count C4s that would be created by adding (u,v)
            common = np.sum((A[u] == 1) & (A[v] == 1))
            # Only add if it helps reach target
            if common == 0 or (degrees[u] < target_degrees[u] and degrees[v] < target_degrees[v]):
                if degrees[u] < target_degrees[u] and degrees[v] < target_degrees[v]:
                    A[u, v] = A[v, u] = 1
                    edges.append((u, v))
                    edge_set.add((u, v))
                    degrees[u] += 1
                    degrees[v] += 1
    
    # If we don't have enough edges, add more
    while len(edges) < m_target:
        best_u, best_v = -1, -1
        best_common = float('inf')
        for u in range(n):
            if degrees[u] >= target_degrees[u]:
                continue
            for v in range(u+1, n):
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if degrees[v] >= target_degrees[v]:
                    continue
                common = np.sum((A[u] == 1) & (A[v] == 1))
                if common < best_common:
                    best_common = common
                    best_u, best_v = u, v
        if best_u == -1:
            # Relax degree constraints
            best_u, best_v = -1, -1
            best_common = float('inf')
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set or (v, u) in edge_set:
                        continue
                    common = np.sum((A[u] == 1) & (A[v] == 1))
                    if common < best_common:
                        best_common = common
                        best_u, best_v = u, v
        if best_u == -1:
            break
        A[best_u, best_v] = A[best_v, best_u] = 1
        edges.append((best_u, best_v))
        edge_set.add((best_u, best_v))
        degrees[best_u] += 1
        degrees[best_v] += 1
    
    # If we have too many edges, remove some
    while len(edges) > m_target:
        # Remove edge that contributes most to C4s
        best_edge = None
        best_c4 = -1
        for (u, v) in edges:
            common = np.sum((A[u] == 1) & (A[v] == 1))
            if common > best_c4:
                best_c4 = common
                best_edge = (u, v)
        if best_edge is None:
            break
        u, v = best_edge
        A[u, v] = A[v, u] = 0
        edges.remove((u, v))
        edge_set.discard((u, v))
        degrees[u] -= 1
        degrees[v] -= 1
    
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)

    def count_c4(u, v):
        """Count 4-cycles containing edge (u,v)."""
        nu = adj[u]
        nv = adj[v]
        c = 0
        for a in nu:
            if a == v:
                continue
            for b in nv:
                if b == u or b == a:
                    continue
                if b in adj[a]:
                    c += 1
        return c
    
    def count_c4_fast(u, v):
        """Fast count using numpy."""
        common = np.sum((A[u] == 1) & (A[v] == 1))
        # Each common neighbor pair forms a C4 through (u,v)
        # But we need to count actual 4-cycles, not just common neighbors
        # For edge (u,v), C4s through it = sum over common neighbors a,b of [a-b edge exists]
        # This is (A^2)[u,v] - degree contribution... actually:
        # Number of 4-cycles containing edge (u,v) = 1/2 * sum_{a in N(u)∩N(v)} (deg(a) - 2)
        # Wait, that's not quite right either. Let me think...
        # A 4-cycle u-a-v-b-u requires a,b common neighbors of u,v with edge a-b
        # So count = number of edges in the subgraph induced by N(u)∩N(v)
        common_neighbors = [x for x in range(n) if A[u,x]==1 and A[v,x]==1]
        count = 0
        for i in range(len(common_neighbors)):
            for j in range(i+1, len(common_neighbors)):
                if A[common_neighbors[i], common_neighbors[j]] == 1:
                    count += 1
        return count

    def remove_edge(u, v):
        A[u, v] = A[v, u] = 0
        adj[u].discard(v)
        adj[v].discard(u)

    def add_edge(u, v):
        A[u, v] = A[v, u] = 1
        adj[u].add(v)
        adj[v].add(u)

    def compute_total_c4():
        """Exact count of 4-cycles using matrix formula."""
        # Total C4s = (trace(A^4) - 2*m - 4*sum_v C(deg_v,2)) / 8
        # Actually: trace(A^4) = 8*C4_count + 4*sum_v C(deg_v,2) + 2*m
        # So C4_count = (trace(A^4) - 4*sum_v C(deg_v,2) - 2*m) / 8
        A2 = A @ A
        trace_A4 = np.trace(A2 @ A2)
        deg_sum_sq = np.sum(np.diag(A2) * (np.diag(A2) - 1)) // 2  # sum C(deg_v, 2)
        c4_count = (trace_A4 - 4 * deg_sum_sq - 2 * m_target) // 8
        return int(c4_count)

    # ---- Verify we have exactly 91 edges ----
    assert len(edges) == m_target, f"Expected {m_target} edges, got {len(edges)}"

    # ---- Fast local search with incremental C4 tracking ----
    # Precompute all non-edges
    all_nonedges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]

    # Compute initial total C4
    total_c4 = compute_total_c4()
    best_edges = list(edges)
    best_total = total_c4
    
    # Debug: print initial state
    print(f"Initial: {len(edges)} edges, {total_c4} C4s, avg_deg={np.mean([len(adj[i]) for i in range(n)])}")

    def delta_swap(ru, rv, au, av):
        """
        Compute delta in total C4 count for removing (ru,rv) and adding (au,av).
        Uses matrix-based counting for speed.
        """
        # C4s destroyed: those containing (ru,rv) in current graph
        destroyed = count_c4(ru, rv)
        
        # Temporarily remove (ru,rv)
        A[ru, rv] = A[rv, ru] = 0
        adj[ru].discard(rv)
        adj[rv].discard(ru)
        
        # C4s created: those containing (au,av) in the modified graph
        created = count_c4(au, av)
        
        # Restore
        A[ru, rv] = A[rv, ru] = 1
        adj[ru].add(rv)
        adj[rv].add(ru)
        
        return created - destroyed

    # Local search: steepest descent on edge swaps
    no_improve_count = 0
    max_no_improve = 100

    for it in range(50000):
        if time.time() > deadline - 5.0:
            break
        
        # Try to find an improving swap
        improved = False
        
        # Score all removal candidates (C4s through that edge)
        rem_scores = [(count_c4(e[0], e[1]), e) for e in edges]
        rem_scores.sort(key=lambda x: x[0])
        
        # Score all addition candidates
        add_scores = []
        for (u, v) in all_nonedges:
            add_scores.append((count_c4(u, v), (u, v)))
        add_scores.sort(key=lambda x: x[0])
        
        # Try combinations of top removals and additions
        found = False
        for rc, re in rem_scores[:20]:
            if found:
                break
            ru, rv = re
            for ac, ae in add_scores[:20]:
                au, av = ae
                d = delta_swap(ru, rv, au, av)
                if d < 0:
                    # Commit the swap
                    remove_edge(ru, rv)
                    edge_set.discard(tuple(sorted((ru, rv))))
                    add_edge(au, av)
                    edge_set.add(tuple(sorted((au, av))))
                    edges.remove(re)
                    edges.append((au, av))
                    # Update nonedges list
                    all_nonedges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
                    total_c4 += d
                    found = True
                    improved = True
                    break
        
        if improved:
            no_improve_count = 0
            if total_c4 < best_total:
                best_total = total_c4
                best_edges = list(edges)
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Perturbation: make 5 random swaps to escape local minimum
                for _ in range(5):
                    re = edges[np.random.randint(len(edges))]
                    nonedge_list = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
                    if not nonedge_list:
                        break
                    ae = nonedge_list[np.random.randint(len(nonedge_list))]
                    ru, rv = re
                    au, av = ae
                    d = delta_swap(ru, rv, au, av)
                    remove_edge(ru, rv)
                    edge_set.discard(tuple(sorted((ru, rv))))
                    add_edge(au, av)
                    edge_set.add(tuple(sorted((au, av))))
                    edges.remove(re)
                    edges.append((au, av))
                    all_nonedges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
                    total_c4 += d
                no_improve_count = 0

    # Final polish: try all pairs of edges for best swap
    if time.time() < deadline - 5.0:
        for _ in range(5):  # Multiple polish passes
            improved_polish = True
            while improved_polish and time.time() < deadline - 3.0:
                improved_polish = False
                rem_scores = [(count_c4(e[0], e[1]), e) for e in edges]
                rem_scores.sort(key=lambda x: x[0])
                add_scores = []
                for (u, v) in all_nonedges:
                    add_scores.append((count_c4(u, v), (u, v)))
                add_scores.sort(key=lambda x: x[0])
                
                for rc, re in rem_scores[:30]:
                    ru, rv = re
                    for ac, ae in add_scores[:30]:
                        au, av = ae
                        d = delta_swap(ru, rv, au, av)
                        if d < 0:
                            remove_edge(ru, rv)
                            edge_set.discard(tuple(sorted((ru, rv))))
                            add_edge(au, av)
                            edge_set.add(tuple(sorted((au, av))))
                            edges.remove(re)
                            edges.append((au, av))
                            all_nonedges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
                            total_c4 += d
                            improved_polish = True
                            break
                    if improved_polish:
                        break
                
                if total_c4 < best_total:
                    best_total = total_c4
                    best_edges = list(edges)

    # Verify final count
    final_total = compute_total_c4()
    if final_total < best_total:
        best_total = final_total
        best_edges = list(edges)

    # Final debug
    print(f"Final: {len(best_edges)} edges, {best_total} C4s")

    return [[e[0], e[1]] for e in best_edges]
