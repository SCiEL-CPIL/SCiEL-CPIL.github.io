import time
import random
from itertools import combinations

def _is_k4_free_bitmask(adj):
    """Check if graph with adjacency list of bitmasks is K4-free."""
    n = 18
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] & (1 << b)):
                continue
            # Common neighbors of a and b
            common = adj[a] & adj[b]
            # Check if there is an edge within common
            # Iterate through bits in common
            c_mask = common
            while c_mask:
                lsb = c_mask & -c_mask
                c = lsb.bit_length() - 1
                c_mask ^= lsb
                # If c is connected to any other node in common, we have a K4
                # adj[c] & common will contain neighbors of c in common
                # We need to ensure it's not just c itself (which it isn't, no self loops)
                # But we need to check if there is ANOTHER node d in common such that (c,d) is edge.
                # So if (adj[c] & common) != 0, it means c is connected to some d in common.
                # Since d != c, this forms a triangle c-d-...?
                # Wait. The K4 is {a, b, c, d}.
                # Edges: (a,b), (a,c), (b,c), (a,d), (b,d), (c,d).
                # a,b are connected. c, d are common neighbors of a,b.
                # So (a,c), (b,c), (a,d), (b,d) are edges.
                # We need (c,d) to be an edge.
                # So if (adj[c] & common) != 0, it implies there is a d in common such that (c,d) is an edge.
                # Yes.
                if (adj[c] & common) != 0:
                    return False
    return True

def _max_triangle_free_size_bitmask(adj):
    """
    Exact branch and bound / exhaustive check for largest triangle-free induced subgraph.
    Given the small n=18, we can check all subsets of size k from high to low.
    """
    n = 18
    # Precompute subsets of size 12, 11, 10 as bitmasks
    # We check 12, then 11, then 10.
    
    def is_triangle_free(mask):
        """Check if subgraph induced by mask is triangle-free."""
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                # Common neighbors of u and v in mask
                if (adj[u] & adj[v] & mask) != 0:
                    return False
        return True

    # Check size 12
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if is_triangle_free(mask):
            return 12
            
    # Check size 11
    for combo in combinations(range(n), 11):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if is_triangle_free(mask):
            return 11
            
    # Check size 10
    for combo in combinations(range(n), 10):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if is_triangle_free(mask):
            return 10
            
    # If we get here, size is < 10
    # We can assume 9 or lower, but 10 is a very safe lower bound for random graphs
    # We return 9 to be conservative if we don't want to compute further, 
    # but let's just return 9 as "small".
    return 9

def construct():
    deadline = time.time() + 115  # Safety margin
    n = 18
    
    # Seed graph: K6,6,6
    # Parts: 0-5, 6-11, 12-17
    seed_adj = [0] * n
    for i in range(6):
        for j in range(6, 12):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
        for j in range(12, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
            
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    best_edges = adj_to_edges(seed_adj)
    best_score = 12 # K6,6,6 has score 12
    
    # Precompute all masks for sizes 12, 11, 10 to speed up evaluation
    # However, generating them on the fly in the check function above is fine 
    # because the check function is called infrequently relative to edge additions?
    # No, we call it for every candidate graph.
    # Let's precompute the masks once.
    
    masks_12 = []
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        masks_12.append(mask)
        
    masks_11 = []
    for combo in combinations(range(n), 11):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        masks_11.append(mask)
        
    masks_10 = []
    for combo in combinations(range(n), 10):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        masks_10.append(mask)

    def fast_is_triangle_free(mask, adj):
        """Check if subgraph induced by mask is triangle-free."""
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if (adj[u] & adj[v] & mask) != 0:
                    return False
        return True

    def evaluate_score(adj):
        # Check 12
        for mask in masks_12:
            if fast_is_triangle_free(mask, adj):
                return 12
        # Check 11
        for mask in masks_11:
            if fast_is_triangle_free(mask, adj):
                return 11
        # Check 10
        for mask in masks_10:
            if fast_is_triangle_free(mask, adj):
                return 10
        return 9

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        common = adj[u] & adj[v]
        c_mask = common
        while c_mask:
            lsb = c_mask & -c_mask
            w = lsb.bit_length() - 1
            c_mask ^= lsb
            if (adj[w] & common) != 0:
                return False
        return True

    def generate_random_max_k4_free(rng):
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    rng = random.Random(42)
    
    # Try random graphs
    trials = 0
    while time.time() < deadline:
        trials += 1
        adj = generate_random_max_k4_free(rng)
        
        score = evaluate_score(adj)
        
        if score < best_score:
            best_score = score
            best_edges = adj_to_edges(adj)
            if best_score <= 10:
                # Found a very good graph, return immediately
                return best_edges
                
    return best_edges

# EVOLVE-BLOCK-END
