"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
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
        We check: for each a in N(u)\\{v}, d in N(v)\\{u}, a!=d:
          Let S_a = N(a) \\ {u,v,d}
          Let T_d = N(d) \\ {u,v,a}
          We need b in S_a, c in T_d, edge(b,c) exists, b!=c.
          Equivalent: exists b in S_a such that (N(b) & T_d & ~{a}) != 0
          where we also need c != b (guaranteed since c in T_d and b not in T_d if b not neighbor of d... 
          actually b could be neighbor of d. Let's be careful.)
          
        Better: For fixed a, d:
          We need a path a-b-c-d of length 3 in the graph with u,v removed.
          That means: exists b, c such that:
            b in N(a) \\ {u,v,d}
            c in N(d) \\ {u,v,a}  
            edge(b,c) exists
            b != c
            b != a (already: b in N(a) and simple graph means b != a)
            c != d (already: c in N(d) and simple graph)
            b != d (ensured by b not in {d})
            c != a (ensured by c not in {a})
          
          So: for b in N(a)\\{u,v,d}, check if N(b) & (N(d)\\{u,v,a}) & ~(1<<b) is non-empty.
          But we also need c != b, which is handled by ~(1<<b).
          And we need c != a, handled by excluding a from N(d).
          
        Even faster: for fixed a,d:
          Let B = N(a) & ~(1<<u) & ~(1<<v) & ~(1<<d)
          Let C = N(d) & ~(1<<u) & ~(1<<v) & ~(1<<a)
          We need: exists b in B such that N(b) & C has a bit set that is not b.
          i.e., OR over b in B of (N(b) & C & ~(1<<b)) != 0
          
        We can precompute for each vertex b, N(b). Then for each (a,d) pair,
        we iterate over b in B and check N(b) & C & ~(1<<b).
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        # Iterate over a in N(u)\{v}
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            # B = N(a) excluding u, v, and we'll exclude d per-d
            Na_base = adj[a] & ~(1 << u) & ~(1 << v)
            
            mask_d = Nv & ~(1 << a)  # d != a
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
    
    # Spectral-guided initialization
    def spectral_init():
        """Create a random 4-regular graph, compute eigenvector, use it to guide edge selection."""
        rng = np.random.default_rng(42)
        
        # Create a random 4-regular graph on 40 vertices
        # Use configuration model or simple approach
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
        # Edges between vertices with similar spectral components are more likely in C6-free graphs
        # Use product of absolute values as a weight
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
    
    # Greedy addition using fast C6 check
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
    
    # Strategy 1: Spectral-guided initialization
    potential_edges, _ = spectral_init()
    
    edges1 = []
    adj1 = [0] * n
    edge_set1 = set()
    greedy_add(adj1, edge_set1, edges1, potential_edges)
    
    best_edges = edges1[:]
    best_count = len(edges1)
    
    # Strategy 2: Standard K5-based initialization
    edges2 = []
    for k in range(8):
        edges2 += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    adj2 = make_adj(edges2)
    edge_set2 = set((min(u,v), max(u,v)) for u, v in edges2)
    
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set2]
    greedy_add(adj2, edge_set2, edges2, all_possible)
    
    if len(edges2) > best_count:
        best_count = len(edges2)
        best_edges = edges2[:]
    
    # Strategy 3: Randomized restarts from best
    for restart in range(200):
        if not time_ok():
            break
        
        if restart < 5:
            # Start from best
            current_edges = best_edges[:]
        else:
            # Start from a random subset of best
            current_edges = [e for e in best_edges if random.random() < 0.85]
        
        current_edge_set = set((min(u,v), max(u,v)) for u, v in current_edges)
        current_adj = make_adj(current_edges)
        
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in current_edge_set]
        random.shuffle(remaining)
        
        greedy_add(current_adj, current_edge_set, current_edges, remaining)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
    
    # Final polish: try to add any remaining edges
    final_edge_set = set((min(u,v), max(u,v)) for u, v in best_edges)
    final_adj = make_adj(best_edges)
    final_remaining = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in final_edge_set]
    greedy_add(final_adj, final_edge_set, best_edges, final_remaining)
    
    return best_edges
# EVOLVE-BLOCK-END
