import random
import time
import math

def construct():
    n = 20
    adj = [0] * n
    
    def popcount(x):
        # Python 3.10+ has int.bit_count(), but bin is portable
        # For 20-bit ints, bin is fast enough
        return bin(x).count('1')

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(u, v):
        # Check if adding edge (u, v) violates the constraint
        # Constraint: For all triples {a,b,c}, |N(a) & N(b) & N(c)| <= 2
        # Adding (u,v) updates N(u) and N(v).
        # Only triples containing u or v are affected.
        
        # Triples {u, w, z} where w, z != u, v
        # New common neighbors for {u, w, z} = (N(u) | {v}) & N(w) & N(z)
        # Increase by 1 if v in N(w) & N(z).
        # If v in N(w) & N(z), we need old count < 2.
        # Old count = popcount(N(u) & N(w) & N(z)).
        
        # Triples {v, w, z}
        # If u in N(w) & N(z), we need popcount(N(v) & N(w) & N(z)) < 2.
        
        for w in range(n):
            if w == u or w == v:
                continue
            mask_w = adj[w]
            # Check if w is connected to v? No, we check if v is in N(w).
            # Wait, the condition is: if v is a common neighbor of w and z.
            # i.e., v in N(w) AND v in N(z).
            
            # Optimization: Only check z if w is connected to v?
            # If w is NOT connected to v, then v is not in N(w), so the condition "v in N(w) & N(z)" is false.
            # So we can skip the inner loop for w if v is not in N(w).
            if not (mask_w >> v) & 1:
                continue
                
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                
                mask_z = adj[z]
                
                # Check triple {u, w, z}
                # Condition: v in N(w) (already checked) AND v in N(z)
                if (mask_z >> v) & 1:
                    common = adj[u] & mask_w & mask_z
                    if popcount(common) >= 2:
                        return False
                
                # Check triple {v, w, z}
                # Condition: u in N(w) AND u in N(z)
                if (mask_w >> u) & 1 and (mask_z >> u) & 1:
                    common = adj[v] & mask_w & mask_z
                    if popcount(common) >= 2:
                        return False
        
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def edge_count():
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2

    def greedy_construct():
        nonlocal adj
        adj = [0] * n
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        for u, v in edges_copy:
            if is_valid_addition(u, v):
                add_edge(u, v)
        
        # Polish phase: try to add any missing edge that is valid
        # We scan all edges, if missing and valid, add it.
        # Repeat until no changes.
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_addition(u, v):
                        add_edge(u, v)
                        improved = True
        
        return edge_count()

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj = [0] * n
    best_count = 0
    
    # Run multiple greedy restarts
    while time.time() < deadline:
        count = greedy_construct()
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
    # Reconstruct edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                best_edges.append([u, v])
                
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
