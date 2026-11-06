import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    n = 40

    def time_ok():
        return time.time() < deadline

    # Bitset-based adjacency
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def would_create_c6_fast(adj, u, v):
        """Fast bitset check: does adding edge (u,v) create a C6?
        A C6 using edge (u,v): u-a-b-c-d-v, all 6 distinct.
        Equivalent: there exist a in N(u)\\{v}, d in N(v)\\{u}, a!=d,
        and a path a-b-c-d of length 3 in G\\{u,v} with b,c not in {u,v,a,d}.
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        # Iterate over a in N(u)\{v}
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            # B_base = N(a) excluding u, v
            Na_base = adj[a] & ~(1 << u) & ~(1 << v)
            
            # Iterate over d in N(v)\{u, a}
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                # B = N(a) \ {u,v,d}
                B = Na_base & ~(1 << d)
                if B == 0:
                    continue
                
                # C = N(d) \ {u,v,a}
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0:
                    continue
                
                # For each b in B, check if N(b) & C & ~(1<<b) != 0
                mask_b = B
                found = False
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    # Check if any neighbor of b is in C and is not b itself
                    if adj[b] & C & ~(1 << b):
                        found = True
                        break
                
                if found:
                    return True
        
        return False

    def greedy_add(adj, edge_set, edges, candidate_edges):
        """Add all non-conflicting edges from candidate_edges."""
        added_any = True
        while added_any and time_ok():
            added_any = False
            remaining = []
            for item in candidate_edges:
                if len(item) == 2:
                    u, v = item
                else:
                    u, v = item[1], item[2]
                if (min(u,v), max(u,v)) in edge_set:
                    continue
                if would_create_c6_fast(adj, u, v):
                    remaining.append(item)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u,v), max(u,v)))
                    edges.append([u, v])
                    added_any = True
            candidate_edges = remaining
        return edges

    # Strategy 1: Spectral-guided initialization
    def spectral_init():
        """Create a random 4-regular graph, compute eigenvector, use it to guide edge selection."""
        rng = np.random.default_rng(42)
        
        # Create a random 4-regular graph on 40 vertices
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            candidates = list(range(n))
            candidates.remove(i)
            rng.shuffle(candidates)
            # Pick 4 neighbors (ensuring symmetry)
            for j in candidates[:4]:
                if adj_matrix[i][j] == 0:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
            # If degree < 4, fill in
            deg = adj_matrix[i].sum()
            if deg < 4:
                for j in range(n):
                    if i != j and adj_matrix[i][j] == 0:
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1
                        deg += 1
                        if deg >= 4:
                            break
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
        
        # Find eigenvalue closest to sqrt(5) ≈ 2.236
        target = np.sqrt(5)
        idx = np.argmin(np.abs(eigenvalues - target))
        spec_vec = eigenvectors[:, idx]
        
        # Use spectral components to guide edge inclusion
        weights = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                weights[i][j] = abs(spec_vec[i] * spec_vec[j])
        
        # Get all potential edges sorted by weight (descending)
        potential_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                potential_edges.append((weights[i][j], i, j))
        potential_edges.sort(reverse=True)
        
        return potential_edges, adj_matrix

    # Strategy 2: Standard K5-based initialization
    def k5_init():
        """Create 8 disjoint copies of K5."""
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges

    # Run multiple strategies and keep the best
    best_edges = []
    best_count = 0

    # Strategy 1: Spectral
    potential_edges, _ = spectral_init()
    edges1 = []
    adj1 = [0] * n
    edge_set1 = set()
    greedy_add(adj1, edge_set1, edges1, potential_edges)
    
    if len(edges1) > best_count:
        best_count = len(edges1)
        best_edges = edges1[:]

    # Strategy 2: K5-based
    edges2 = k5_init()
    adj2 = make_adj(edges2)
    edge_set2 = set((min(u,v), max(u,v)) for u, v in edges2)
    
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set2]
    random.seed(42)
    random.shuffle(all_possible)
    
    greedy_add(adj2, edge_set2, edges2, all_possible)
    
    if len(edges2) > best_count:
        best_count = len(edges2)
        best_edges = edges2[:]

    # Strategy 3: Multiple random orderings
    all_edges_list = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for trial in range(15):
        if not time_ok():
            break
        random.seed(1000 + trial)
        random.shuffle(all_edges_list)
        
        adj3 = [0] * n
        edges3 = []
        edge_set3 = set()
        greedy_add(adj3, edge_set3, edges3, all_edges_list)
        
        if len(edges3) > best_count:
            best_count = len(edges3)
            best_edges = edges3[:]

    # Local search on the best solution
    if time_ok():
        adj_best = make_adj(best_edges)
        edge_set_best = set((min(u,v), max(u,v)) for u, v in best_edges)
        
        improved = True
        while improved and time_ok():
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not time_ok():
                        break
                    if adj_best[u] & (1 << v):
                        continue
                    if not would_create_c6_fast(adj_best, u, v):
                        adj_best[u] |= (1 << v)
                        adj_best[v] |= (1 << u)
                        edge_set_best.add((u, v))
                        best_edges.append([u, v])
                        improved = True
                        best_count += 1

    return best_edges
