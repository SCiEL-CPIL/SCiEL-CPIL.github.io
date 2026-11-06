"""Structural: C4-free base via exhaustive 3-edge removal + optimal add + fast delta local search."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np
    from itertools import combinations

    start = time.time()
    DEADLINE = 145.0
    n = 31
    target_edges = 91

    # ---- Build polarity graph of PG(2,5) ----
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))

    poly_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                poly_edges.add((i, j))

    poly_edge_list = list(poly_edges)
    m = len(poly_edge_list)  # 93

    # ---- Fast C4 counting using adjacency matrix ----
    def build_adj_matrix(edge_set):
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edge_set:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def count_c4_matrix(A):
        # C4 count = 0.5 * sum over all pairs (i,j) of C(common_neighbors(i,j), 2)
        # Using matrix: (A^2)[i,j] = common neighbors of i and j
        A2 = A @ A
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = A2[i, j]
                total += c * (c - 1) // 2
        return total

    # ---- Step 1: Exhaustive search for C4-free 90-edge subgraph ----
    # Try all C(93,3) = 130,816 triples of edges to remove
    # Use fast matrix-based C4 counting
    
    base_A = build_adj_matrix(poly_edges)
    base_c4 = count_c4_matrix(base_A)
    
    best_90_edge_set = None
    best_90_c4 = float('inf')
    
    # Precompute: for each edge, which C4s does it participate in?
    # Faster: just enumerate all triples and count C4s
    
    # To speed up, use the fact that removing an edge only affects C4s containing it
    # But for correctness, let's do direct counting with early termination
    
    poly_edge_list_sorted = sorted(poly_edge_list)
    num_triples = 0
    found_c4_free = False
    
    # Try all triples, but use a smarter approach:
    # First, find edges that are in the most C4s
    # Then try triples involving those edges
    
    # Compute C4 count per edge in the full graph
    A2_full = base_A @ base_A
    
    def c4s_containing_edge(u, v, A):
        """Count C4s that use edge (u,v)."""
        # A C4 using (u,v) has form u-v-x-y-u where x,y are common neighbors
        # and x-y is an edge
        cn_u = set(np.where(A[u] > 0)[0])
        cn_v = set(np.where(A[v] > 0)[0])
        common = cn_u & cn_v
        common.discard(u)
        common.discard(v)
        common_list = list(common)
        cnt = 0
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                if A[common_list[i], common_list[j]] > 0:
                    cnt += 1
        return cnt

    edge_c4_counts = {}
    for e in poly_edge_list_sorted:
        edge_c4_counts[e] = c4s_containing_edge(e[0], e[1], base_A)
    
    # Sort edges by C4 count (descending)
    edges_by_c4 = sorted(poly_edge_list_sorted, key=lambda e: -edge_c4_counts[e])
    
    # Try all C(93,3) triples, prioritizing high-C4 edges first
    # But that's still 130K. Let's do it with fast counting.
    
    # Alternative: use the fact that we need to remove 3 edges to hit all 45 C4s.
    # Each C4 has 4 edges. We need a hitting set of size 3.
    # This is a small set cover problem.
    
    # First, enumerate all C4s in the polarity graph
    all_c4s = []
    for i in range(n):
        for j in range(i + 1, n):
            if base_A[i, j] == 0:
                continue
            cn_i = set(np.where(base_A[i] > 0)[0])
            cn_j = set(np.where(base_A[j] > 0)[0])
            common = cn_i & cn_j
            common.discard(i)
            common.discard(j)
            common_list = sorted(list(common))
            for k in range(len(common_list)):
                for l in range(k + 1, len(common_list)):
                    a, b = common_list[k], common_list[l]
                    if base_A[a, b] > 0:
                        # C4: i-j-b-a-i
                        c4 = frozenset([(i,j), (j,b), (b,a), (a,i)])
                        # Normalize edge representation
                        edges_in_c4 = frozenset((min(x,y), max(x,y)) for x,y in [(i,j), (j,b), (b,a), (a,i)])
                        all_c4s.append(edges_in_c4)
    
    # Deduplicate C4s
    unique_c4s = list(set(all_c4s))
    num_c4s = len(unique_c4s)
    
    # Find minimum hitting set of size 3
    # Try all triples from the top edges
    best_triple = None
    best_triple_c4 = float('inf')
    
    # Only try triples from top 40 edges by C4 count
    top_edges = edges_by_c4[:40]
    
    for triple in combinations(top_edges, 3):
        triple_set = set(triple)
        # Check if this triple hits all C4s
        hits_all = True
        for c4 in unique_c4s:
            if not (triple_set & c4):
                hits_all = False
                break
        if hits_all:
            # Found a C4-free 90-edge subgraph!
            current_edge_set = set(poly_edges) - triple_set
            A_test = build_adj_matrix(current_edge_set)
            c4_count = count_c4_matrix(A_test)
            if c4_count < best_triple_c4:
                best_triple_c4 = c4_count
                best_triple = triple
            if c4_count == 0:
                best_triple = triple
                best_triple_c4 = 0
                found_c4_free = True
                break
    
    if best_triple is None:
        # Fallback: greedy removal
        current_edge_set = set(poly_edges)
        for _ in range(3):
            best_e = None
            best_cnt = -1
            for e in current_edge_set:
                cnt = c4s_containing_edge(e[0], e[1], build_adj_matrix(current_edge_set))
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_e = e
            if best_e:
                current_edge_set.discard(best_e)
        best_triple = tuple(set(poly_edges) - current_edge_set)
    
    # Build 90-edge C4-free (or near-free) graph
    current_edge_set = set(poly_edges) - set(best_triple)
    current_A = build_adj_matrix(current_edge_set)
    current_c4 = count_c4_matrix(current_A)
    
    # ---- Step 2: Add 1 edge to get 91 edges, minimizing new C4s ----
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
    non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
    
    # For each non-edge, compute C4s it would create
    def c4s_created_by_adding(u, v, A):
        """Count C4s that would be created by adding edge (u,v)."""
        cn_u = set(np.where(A[u] > 0)[0])
        cn_v = set(np.where(A[v] > 0)[0])
        common = cn_u & cn_v
        common_list = list(common)
        cnt = 0
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                if A[common_list[i], common_list[j]] > 0:
                    cnt += 1
        return cnt
    
    best_add_edge = None
    best_add_c4 = float('inf')
    
    for ne in non_edges:
        u, v = ne
        cnt = c4s_created_by_adding(u, v, current_A)
        if cnt < best_add_c4:
            best_add_c4 = cnt
            best_add_edge = ne
    
    if best_add_edge is not None:
        u, v = best_add_edge
        current_edge_set.add((u, v))
        current_A = build_adj_matrix(current_edge_set)
        current_c4 = count_c4_matrix(current_A)
    
    # ---- Step 3: Fast local search using delta C4 computation ----
    # Precompute common neighbor counts for fast delta evaluation
    
    def compute_common_neighbors(A):
        """Return list of sets of common neighbors for each pair."""
        cn = [set() for _ in range(n*n)]
        for i in range(n):
            for j in range(i+1, n):
                cn[i*n+j] = set(np.where((A[i] > 0) & (A[j] > 0))[0])
                cn[i*n+j].discard(i)
                cn[i*n+j].discard(j)
        return cn
    
    # Use a simpler but fast delta approach
    def delta_c4_remove_edge(u, v, A):
        """Estimate change in C4 count when removing edge (u,v)."""
        # C4s destroyed = C4s containing edge (u,v)
        cn_u = set(np.where(A[u] > 0)[0])
        cn_v = set(np.where(A[v] > 0)[0])
        common = cn_u & cn_v
        common.discard(u)
        common.discard(v)
        common_list = list(common)
        cnt = 0
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                if A[common_list[i], common_list[j]] > 0:
                    cnt += 1
        return cnt  # Number of C4s destroyed (negative delta)
    
    def delta_c4_add_edge(u, v, A):
        """Estimate change in C4 count when adding edge (u,v)."""
        return c4s_created_by_adding(u, v, A)  # Number of C4s created (positive delta)
    
    # Local search: try 1-edge swaps using delta approximation
    if time.time() - start < DEADLINE - 10:
        improved = True
        max_rounds = 50
        for rnd in range(max_rounds):
            if not improved or time.time() - start > DEADLINE - 5:
                break
            improved = False
            best_swap = None
            best_delta = 0  # We want negative delta (reduction)
            
            current_edge_list = list(current_edge_set)
            current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
            non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
            
            rng = random.Random(42 + rnd)
            max_evals = 5000
            evals = 0
            
            # Try all edges to remove
            for e1 in current_edge_list:
                if evals >= max_evals or time.time() - start > DEADLINE - 5:
                    break
                u1, v1 = e1
                if u1 > v1:
                    u1, v1 = v1, u1
                delta_remove = -delta_c4_remove_edge(u1, v1, current_A)
                
                # Try random non-edges to add
                num_adds = min(50, len(non_edges))
                add_indices = rng.sample(range(len(non_edges)), num_adds)
                
                for idx in add_indices:
                    if evals >= max_evals or time.time() - start > DEADLINE - 5:
                        break
                    e2 = non_edges[idx]
                    u2, v2 = e2
                    if (u1, v1) == (u2, v2):
                        continue
                    
                    delta_add = delta_c4_add_edge(u2, v2, current_A)
                    total_delta = delta_remove + delta_add
                    evals += 1
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_swap = ((u1, v1), (u2, v2))
            
            if best_swap is not None and best_delta < 0:
                e1, e2 = best_swap
                current_edge_set.discard(e1)
                current_edge_set.add(e2)
                current_A = build_adj_matrix(current_edge_set)
                current_c4 = count_c4_matrix(current_A)
                improved = True
    
    # ---- Step 4: Simulated annealing style refinement ----
    if time.time() - start < DEADLINE - 10:
        rng_sa = random.Random(99)
        current_temp = 2.0
        cooling = 0.95
        
        for _ in range(200):
            if time.time() - start > DEADLINE - 5:
                break
            if current_temp < 0.1:
                break
            
            current_edge_list = list(current_edge_set)
            current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
            non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
            
            if not current_edge_list or not non_edges:
                break
            
            e1 = current_edge_list[rng_sa.randint(0, len(current_edge_list) - 1)]
            e2 = non_edges[rng_sa.randint(0, len(non_edges) - 1)]
            
            u1, v1 = e1
            if u1 > v1:
                u1, v1 = v1, u1
            u2, v2 = e2
            
            if (u1, v1) == (u2, v2):
                continue
            
            # Perform swap
            current_edge_set.discard((u1, v1))
            current_edge_set.add((u2, v2))
            new_A = build_adj_matrix(current_edge_set)
            new_c4 = count_c4_matrix(new_A)
            
            delta = new_c4 - current_c4
            
            if delta < 0 or rng_sa.random() < np.exp(-delta / current_temp):
                # Accept
                current_c4 = new_c4
                current_A = new_A
            else:
                # Revert
                current_edge_set.discard((u2, v2))
                current_edge_set.add((u1, v1))
            
            current_temp *= cooling
    
    # ---- Final verification and return ----
    final_edges = list(current_edge_set)
    
    # Ensure exactly 91 edges
    if len(final_edges) != target_edges:
        if len(final_edges) > target_edges:
            # Remove excess edges
            current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in final_edges)
            final_edges = final_edges[:target_edges]
        elif len(final_edges) < target_edges:
            current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in final_edges)
            for ne in all_pairs:
                if len(final_edges) >= target_edges:
                    break
                if ne not in current_edge_tuple_set:
                    final_edges.append(ne)
    
    # Verify distinctness and validity
    seen = set()
    valid_edges = []
    for e in final_edges:
        u, v = e
        key = (min(u,v), max(u,v))
        if 0 <= u < n and 0 <= v < n and u != v and key not in seen:
            seen.add(key)
            valid_edges.append(key)
    
    # Pad if needed
    if len(valid_edges) < target_edges:
        for ne in all_pairs:
            if len(valid_edges) >= target_edges:
                break
            if ne not in seen:
                valid_edges.append(ne)
                seen.add(ne)
    
    return [[u, v] for u, v in valid_edges[:target_edges]]
# EVOLVE-BLOCK-END
