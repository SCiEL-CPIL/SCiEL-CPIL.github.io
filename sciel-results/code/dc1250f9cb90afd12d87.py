import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy:
    1. Initialize with a deterministic algebraic graph (Paley-like on 19 vertices, padded to 21).
    2. Use a fast vectorized verification to find regular induced subgraphs of size >= 6.
    3. Perform local search: flip edges that are part of regular induced subgraphs to break them.
    4. Return the best valid graph found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def fast_verify(n, adj):
        """
        Checks if the graph defined by adj (n x n int array) has no 
        regular induced subgraph on 6 or more vertices.
        Returns True if valid, False otherwise.
        If False, returns the first regular induced subgraph found as a mask.
        """
        # We need to check all subsets S of size k >= 6.
        # A subset S is regular if all vertices in S have the same degree within S.
        # Let A be the adjacency matrix.
        # For a subset mask M (bitmask), the degree of vertex i in S is sum(A[i, j] for j in S).
        
        # Iterating all subsets of size >= 6 for n=21 is ~2M subsets.
        # Checking each by summing is slow in Python loops.
        # We can vectorize by checking subsets of a specific size k.
        
        # Optimization:
        # If a subset S of size k is regular, then for any i, j in S, 
        # deg_S(i) == deg_S(j).
        # deg_S(i) = popcount(adj_bits[i] & mask_S)
        
        # Let's use bitmasks for adjacency for speed in Python, but try to vectorize the check.
        # Actually, pure Python bitmask iteration is often faster than NumPy overhead for small n.
        # But we can precompute popcounts for all 2^21 masks? No, too big.
        
        # Alternative: Iterate combinations.
        # C(21, 6) = 54k, C(21, 7) = 116k, ..., C(21, 21) = 1. Total ~2M.
        # For each combo, compute degrees.
        
        adj_bits = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            adj_bits[i] = mask
        
        # Check sizes from 6 to n
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Calculate degrees
                d0 = (adj_bits[combo[0]] & mask).bit_count()
                is_reg = True
                for v in combo[1:]:
                    if (adj_bits[v] & mask).bit_count() != d0:
                        is_reg = False
                        break
                
                if is_reg:
                    return False, mask
        return True, 0

    def get_edges(adj):
        edges = []
        n = adj.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    def legendre_symbol(a, p):
        ls = pow(a, (p - 1) // 2, p)
        return -1 if ls == p - 1 else ls

    # Initialize with a Paley graph on 19 vertices (p=19 is prime, 19 % 4 == 3, so Paley is not defined directly for p=3 mod 4? 
    # Wait, Paley graph exists for q = p^k where q ≡ 1 mod 4.
    # 19 ≡ 3 mod 4. So Paley(19) is not a standard undirected graph? 
    # Actually, for p ≡ 3 mod 4, -1 is a non-residue, so the graph is directed? 
    # No, the Paley graph is defined for q ≡ 1 mod 4.
    # Let's use p=17 (17 % 4 = 1). Paley(17) is a strongly regular graph.
    # We need 21 vertices. We can take Paley(17) and add 4 isolated vertices? 
    # Or better, use a random initialization but seeded deterministically.
    
    # Let's try a different algebraic construction: 
    # Graph on Z_21. Edge (u, v) if (v-u) mod 21 is a quadratic residue mod 21?
    # 21 is not prime. Quadratic residues mod 21: 
    # Squares mod 21: 0, 1, 4, 9, 16, 25=4, 36=15, 49=7, 64=1, 81=18, 100=16, 121=16, 144=18, 169=1, 196=7, 225=15, 256=4, 289=4, 324=1, 361=16, 400=1, 441=0.
    # Residues: {0, 1, 4, 7, 9, 15, 16, 18}.
    # If we use a circulant graph, it has high symmetry, which might create regular subgraphs.
    
    # Let's stick to random initialization but with a specific seed, and use local search.
    # The key is the local search: if we find a regular subgraph, flip an edge in it.
    
    n_target = 21
    np.random.seed(42)
    
    # Initialize adj matrix
    adj = np.zeros((n_target, n_target), dtype=int)
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if np.random.rand() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1

    best_n = 5
    best_edges = []
    
    # Local search
    for iteration in range(1000):
        if time.time() > deadline - 10.0:
            break
            
        is_valid, bad_mask = fast_verify(n_target, adj)
        
        if is_valid:
            edges = get_edges(adj)
            return {'n': n_target, 'edges': edges}
        
        # If not valid, flip a random edge in the bad_mask to try to break the regularity.
        # The bad_mask is a bitmask of vertices in the regular subgraph.
        # We pick a random vertex in the subgraph and flip its connection to another vertex in the subgraph.
        vertices_in_sub = [i for i in range(n_target) if (bad_mask >> i) & 1]
        if len(vertices_in_sub) < 2:
            # Should not happen for k>=6
            break
            
        u = np.random.choice(vertices_in_sub)
        v = np.random.choice(vertices_in_sub)
        if u != v:
            # Flip edge (u, v)
            adj[u, v] = 1 - adj[u, v]
            adj[v, u] = adj[u, v]
            
    # If we didn't find a valid graph on 21, try 20?
    # The task says 20 is known. If 21 is too hard, we should return 20.
    # But we need to construct it.
    # Let's try the same process for n=20 if n=21 fails.
    
    n_target = 20
    adj = np.zeros((n_target, n_target), dtype=int)
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if np.random.rand() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
                
    for iteration in range(1000):
        if time.time() > deadline - 5.0:
            break
            
        is_valid, bad_mask = fast_verify(n_target, adj)
        
        if is_valid:
            edges = get_edges(adj)
            return {'n': n_target, 'edges': edges}
        
        vertices_in_sub = [i for i in range(n_target) if (bad_mask >> i) & 1]
        if len(vertices_in_sub) < 2:
            break
            
        u = np.random.choice(vertices_in_sub)
        v = np.random.choice(vertices_in_sub)
        if u != v:
            adj[u, v] = 1 - adj[u, v]
            adj[v, u] = adj[u, v]

    # Fallback to a known small valid graph if all else fails
    # Empty graph on 5 vertices is valid.
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
