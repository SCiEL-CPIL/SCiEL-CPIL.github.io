import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    n = 20
    
    # 1. Algebraic Construction: Paley Graph on 23 vertices
    # Paley graph P_q for q = 4k+1 is a strongly regular graph.
    # For q=23, it is (23, 11, 5, 6).
    # It is known that Paley graphs with q = 4k+1 are K_{3,3}-free?
    # Let's verify: In P_23, any 3 vertices have at most 2 common neighbors?
    # Actually, P_23 is triangle-free? No, P_5 has triangles.
    # Wait, P_5 is C5, triangle free. P_13 is (13, 6, 2, 3).
    # P_23 is (23, 11, 5, 6). 
    # Does P_23 contain K_3,3?
    # A K_3,3 subgraph would require 3 vertices with 3 common neighbors.
    # In a strongly regular graph (v, k, lambda, mu):
    # If two vertices are adjacent, they have lambda common neighbors.
    # If two vertices are non-adjacent, they have mu common neighbors.
    # For 3 vertices u, v, w:
    # Case 1: u, v, w form a triangle.
    #   u and v are adjacent. Common neighbors of u, v include w (since triangle).
    #   Total common neighbors of u, v is lambda=5.
    #   w is one of them. The other 4 are common neighbors of u and v.
    #   We need to count common neighbors of u, v, w.
    #   Let N(u) be neighbors of u.
    #   Common neighbors of u, v, w = N(u) ∩ N(v) ∩ N(w).
    #   Since w is adjacent to u and v, w is in N(u) and N(v).
    #   This analysis is tricky. Let's just compute it for the 20-vertex subset.
    
    # However, a safer algebraic construction for K_3,3-free is often the 
    # "Hadamard graph" or specific conference graphs.
    # But let's stick to the Paley(23) idea first, as it's dense (11/23 ~ 0.48 density).
    # If P_23 is K_3,3-free, the induced subgraph on 20 vertices is also K_3,3-free.
    # If P_23 is NOT K_3,3-free, we might need to repair.
    
    # Let's check P_23 properties.
    # Quadratic residues mod 23:
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=2, 6^2=13, 7^2=3, 8^2=18, 9^2=12, 10^2=8, 11^2=6
    # QR = {1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18}
    
    q = 23
    QR = set()
    for x in range(1, q):
        QR.add((x * x) % q)
    
    # Build adjacency for P_23
    adj_23 = [0] * q
    for u in range(q):
        for v in range(u + 1, q):
            diff = (v - u) % q
            if diff in QR:
                adj_23[u] |= (1 << v)
                adj_23[v] |= (1 << u)
    
    # Check if P_23 is K_3,3 free
    def is_k33_free_mask(adj_mask, n_vertices):
        vertices = list(range(n_vertices))
        for u, v, w in combinations(vertices, 3):
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if common.bit_count() > 2:
                return False
        return True

    # P_23 has 23 vertices. We need 20.
    # Let's check if the full P_23 is K_3,3 free.
    # If not, we will use a repair strategy.
    # Actually, it is a known result that Paley graphs P_q are K_{3,3}-free for small q?
    # For q=5 (C5), yes.
    # For q=13, yes.
    # For q=17? (17, 8, 3, 4).
    # For q=29?
    # Let's just test P_23.
    
    # We will construct a 20-vertex graph by taking the first 20 vertices of P_23.
    # This is an induced subgraph.
    
    # Initial candidate
    init_adj = [0] * n
    for i in range(n):
        init_adj[i] = adj_23[i] & ((1 << n) - 1) # Mask to 20 bits
        # Ensure symmetry within the 20 vertices
        # adj_23 is symmetric. Masking preserves symmetry for pairs within 0..19.
    
    # Verify initial candidate
    if not is_k33_free_mask(init_adj, n):
        # If P_23 induced subgraph is not K_3,3 free, we need a fallback or repair.
        # Let's try a different algebraic construction or just use a dense random start with repair.
        # But let's assume the algebraic structure is good enough to start.
        # If it fails, the repair step below will handle it.
        pass

    # 2. Local Search: Combinatorial 2-Opt / Edge Swaps
    # State: adj_mask (list of int bitmasks)
    # Move: Remove edge (u,v), add edge (x,y) where (u,v) != (x,y) and (x,y) is not an edge.
    # Condition: New graph is K_3,3 free.
    # Objective: Maximize edges.
    # Since we start with a valid graph, we only accept moves that keep it valid.
    # But we want to ADD edges. We can't just swap if we want to increase density.
    # Actually, if we are at a local maximum for "add only", we are stuck.
    # To increase density, we might need to remove an edge to allow adding two others?
    # Or just try adding edges first.
    
    # Phase 1: Greedy Add
    def can_add_edge(u, v, adj):
        # Check if adding (u,v) creates a K_3,3
        # Only need to check triples involving u or v
        bit_u = 1 << u
        bit_v = 1 << v
        
        new_u = adj[u] | bit_v
        new_v = adj[v] | bit_u
        
        # Iterate over all pairs (a,b) and check if they have 3 common neighbors with u or v?
        # No, K_3,3 is defined by 3 vertices having 3 common neighbors.
        # The new edge (u,v) affects the common neighbor counts for triples containing u or v.
        # Specifically, for any pair (a,b), the number of common neighbors might change?
        # No, the common neighbors of (a,b) are N(a) ∩ N(b).
        # Adding (u,v) changes N(u) and N(v).
        # It does NOT change N(a) or N(b) for a,b not u,v.
        # So, the only triples whose common neighbor count can change are those containing u or v.
        
        # Triples containing u: (u, a, b) for a,b in 0..n-1, a!=u, b!=u, a<b
        # Common neighbors of (u, a, b) = N(u) ∩ N(a) ∩ N(b).
        # With new edge, N(u) becomes N(u) ∪ {v}.
        # So new common = (N(u) ∩ N(a) ∩ N(b)) ∪ ({v} ∩ N(a) ∩ N(b)).
        # We need popcount(new common) <= 2.
        # If popcount(old common) >= 2, we must ensure v is NOT in N(a) ∩ N(b).
        # If popcount(old common) == 2, and v is in N(a) ∩ N(b), then new count is 3 -> Violation.
        # If popcount(old common) < 2, adding v might make it 3 only if old was 2 and v is common?
        # Wait.
        # old_common = N(u) ∩ N(a) ∩ N(b).
        # new_common = old_common ∪ ({v} ∩ N(a) ∩ N(b)).
        # Note that v is not in N(u) initially (since we are adding it).
        # So v is not in old_common.
        # So new_common size = |old_common| + (1 if v in N(a) ∩ N(b) else 0).
        # We need |new_common| <= 2.
        # So:
        # 1. If |old_common| > 2: Impossible (graph was valid, so this doesn't happen).
        # 2. If |old_common| == 2: We need v NOT in N(a) ∩ N(b).
        # 3. If |old_common| == 1: We need v NOT in N(a) ∩ N(b) if we want to stay <=2?
        #    If v in N(a) ∩ N(b), new size = 2. This is allowed.
        # 4. If |old_common| == 0: New size 0 or 1. Allowed.
        
        # So the constraint is: For all pairs (a,b) such that (u,a,b) is a triple,
        # if |N(u) ∩ N(a) ∩ N(b)| == 2, then v must NOT be a common neighbor of a and b.
        # i.e., v cannot be in N(a) ∩ N(b).
        
        # Similarly for triples containing v: (v, a, b).
        # Constraint: If |N(v) ∩ N(a) ∩ N(b)| == 2, then u must NOT be in N(a) ∩ N(b).
        
        # Let's implement this check efficiently.
        
        # Precompute common neighbors for all pairs? Too expensive for dynamic updates.
        # But n=20 is small. 20*19/2 = 190 pairs.
        # We can just loop.
        
        # Optimization: 
        # For each a in 0..n-1 (a != u, a != v):
        #   For each b in a+1..n-1 (b != u, b != v):
        #      Check triple (u, a, b)
        #      common_ub = adj[u] & adj[a] & adj[b]
        #      if common_ub.bit_count() == 2:
        #          if (adj[a] & bit_v) and (adj[b] & bit_v):
        #              return False
        #      Check triple (v, a, b)
        #      common_vb = adj[v] & adj[a] & adj[b]
        #      if common_vb.bit_count() == 2:
        #          if (adj[a] & bit_u) and (adj[b] & bit_u):
        #              return False
              
        # Wait, is that sufficient?
        # What if |old_common| > 2? The graph was valid, so this is 0 or 1 or 2.
        # If it was 2, adding v makes it 3 ONLY IF v is common to a and b.
        # If it was 1, adding v makes it 2 (if v common) or 1. Both <= 2.
        # If it was 0, adding v makes it 1 (if v common) or 0.
        
        # So yes, the check is:
        # For all a,b not u,v:
        #   if (adj[u] & adj[a] & adj[b]).bit_count() == 2 and (adj[a] & bit_v) and (adj[b] & bit_v): return False
        #   if (adj[v] & adj[a] & adj[b]).bit_count() == 2 and (adj[a] & bit_u) and (adj[b] & bit_u): return False
        
        # This is O(n^2) per edge addition check.
        # n=20, so 400 iterations. Very fast.
        
        bit_v = 1 << v
        bit_u = 1 << u
        
        for a in range(n):
            if a == u or a == v: continue
            for b in range(a + 1, n):
                if b == u or b == v: continue
                
                # Check triple (u, a, b)
                common_uab = adj[u] & adj[a] & adj[b]
                if common_uab.bit_count() == 2:
                    # If v is common neighbor of a and b, adding (u,v) makes 3 common neighbors for (u,a,b)
                    if (adj[a] & bit_v) and (adj[b] & bit_v):
                        return False
                
                # Check triple (v, a, b)
                common_vab = adj[v] & adj[a] & adj[b]
                if common_vab.bit_count() == 2:
                    # If u is common neighbor of a and b, adding (u,v) makes 3 common neighbors for (v,a,b)
                    if (adj[a] & bit_u) and (adj[b] & bit_u):
                        return False
                        
        return True

    def greedy_add(adj):
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if can_add_edge(i, j, adj):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            changed = True
        return adj

    # Apply Greedy Add to initial algebraic graph
    current_adj = list(init_adj)
    current_adj = greedy_add(current_adj)
    
    best_adj = list(current_adj)
    best_edges = sum(m.bit_count() for m in best_adj) // 2
    
    # Phase 2: Local Search with Edge Swaps (2-Opt)
    # Try to remove an edge and add two others? Or just swap?
    # Since we want to maximize edges, we are likely at a maximal K_3,3-free graph.
    # To improve, we need to find a set of edges to remove and a set of edges to add 
    # such that |add| > |remove| and the result is valid.
    # Simplest: Remove 1 edge, Add 2 edges.
    # Or: Remove 2 edges, Add 3 edges.
    
    # Let's try a simpler "Kick" strategy:
    # 1. Remove k random edges (k=1 or 2)
    # 2. Run Greedy Add
    # 3. If edge count increases, accept.
    
    def local_search_kick(adj, k, iterations, rng):
        best = list(adj)
        best_cnt = sum(m.bit_count() for m in best) // 2
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            # Select k random edges to remove
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best[i] & (1 << j):
                        edges.append((i, j))
                        
            if len(edges) < k:
                break
                
            to_remove = rng.sample(edges, k)
            
            new_adj = list(best)
            for u, v in to_remove:
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
            # Greedy Add
            new_adj = greedy_add(new_adj)
            
            new_cnt = sum(m.bit_count() for m in new_adj) // 2
            
            if new_cnt > best_cnt:
                best_cnt = new_cnt
                best = new_adj
                if best_cnt > 190: # High density
                    break
        
        return best, best_cnt

    # Run multiple kicks
    rng = random.Random(42)
    for k in [1, 2, 3]:
        if time.time() > deadline - 10:
            break
        # Do several kicks of size k
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            current_adj, current_edges = local_search_kick(best_adj, k, 10, rng)
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = current_adj

    # Final verification
    if not is_k33_free_mask(best_adj, n):
        # Fallback to a known valid graph if something went wrong
        # Cycle C20
        best_adj = [0] * n
        for i in range(n):
            best_adj[i] |= (1 << ((i + 1) % n))
            best_adj[i] |= (1 << ((i - 1) % n))
        best_edges = n

    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
