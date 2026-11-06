"""Construct a C5-free graph with no independent 9-set, maximizing n."""

import time
import random
import itertools

def _has_c5(n, adj):
    """Check if graph has a C5 (5-cycle as subgraph) using bitmasks."""
    # Convert adj to bitmask for fast intersection
    mask = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        mask[i] = m
    
    # A C5 exists iff there are 5 distinct vertices v0..v4 with edges
    # v0-v1, v1-v2, v2-v3, v3-v4, v4-v0.
    # We can fix v0 < v1 to avoid some symmetries, but for n<=35, 
    # checking all 5-tuples with distinct vertices is C(35,5)*12 ~ 3.9M ops, which is fast.
    # However, a smarter check:
    # For each edge (u, v), look for a path of length 3 from u to v that doesn't use u or v internally?
    # No, that finds C4 or C5.
    # Let's just do the 5-subset check but optimized.
    
    # Actually, the most efficient exact check for n<=35 is to iterate over all 5-subsets
    # and check if the induced subgraph contains a C5.
    # There are C(35,5) = 324,632 subsets.
    # For each subset, check if it has a Hamiltonian cycle.
    # 5 vertices, 12 possible cycles.
    # 324632 * 12 * 5 edge checks = ~20M operations. This might be too slow if done repeatedly in a search loop.
    
    # Alternative: incremental check.
    # But for the final verification, this is fine.
    # For the search, we need a faster "does adding edge (u,v) create a C5?" check.
    
    # For now, let's implement the full check efficiently.
    for a in range(n):
        for b in range(a + 1, n):
            if not (mask[a] & (1 << b)):
                continue
            for c in range(b + 1, n):
                if not (mask[b] & (1 << c)):
                    continue
                # We need d, e such that c-d, d-e, e-a are edges, and all distinct.
                # d must be neighbor of c, e must be neighbor of d and a.
                # d cannot be a, b, c.
                # e cannot be a, b, c, d.
                # Also e must be neighbor of a.
                
                # Iterate d in N(c) \ {a, b, c}
                d_candidates = [d for d in range(n) if d != a and d != b and d != c and (mask[c] & (1 << d))]
                for d in d_candidates:
                    # e must be in N(d) ∩ N(a) \ {a, b, c, d}
                    # And e must be distinct from a,b,c,d.
                    # Also, we need to ensure the set {a,b,c,d,e} is exactly the cycle.
                    # The cycle is a-b-c-d-e-a.
                    # Edges: (a,b), (b,c), (c,d), (d,e), (e,a).
                    # We already have (a,b), (b,c), (c,d).
                    # Need (d,e) and (e,a).
                    # e must be neighbor of d and a.
                    e_candidates = [e for e in range(n) if e != a and e != b and e != c and e != d and (mask[d] & (1 << e)) and (mask[a] & (1 << e))]
                    for e in e_candidates:
                        # Check if this forms a simple cycle.
                        # Vertices a,b,c,d,e are distinct.
                        # Edges are present.
                        # Is it a C5 subgraph? Yes.
                        # We don't need to check if there are *extra* edges, just that a C5 exists.
                        return True
    return False

def _max_independent_set_size(n, adj):
    """Find size of maximum independent set. For n<=35, use Bron-Kerbosch on complement."""
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    best = [0]
    
    def bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) > best[0]:
                best[0] = len(R)
            return
        if P:
            pivot = max(P | X, key=lambda u: len(comp_adj[u] & P))
            candidates = P - comp_adj[pivot]
        else:
            candidates = P
        
        for v in candidates:
            R.add(v)
            bron_kerbosch(R, P & comp_adj[v], X & comp_adj[v])
            R.discard(v)
            P.discard(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    return best[0]

def construct():
    import time
    start = time.time()
    deadline = start + 140  # Safety margin
    
    # Start with 8 disjoint K4s: n=32, alpha=8, C5-free
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    best_n = n
    best_edges = list(edges)
    
    # Helper to build adjacency from edges
    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # Try to extend to n=33, 34, 35
    # Strategy: Add a vertex v.
    # To keep alpha <= 8, v must be connected to all vertices in at least one K4 block.
    # To avoid C5, v must not create a 5-cycle.
    
    # Let's try adding vertices one by one.
    current_n = 32
    current_edges = list(edges)
    current_adj = build_adj(current_n, current_edges)
    
    # We will try to add vertices 32, 33, 34 (to reach n=35)
    # For each new vertex, we try different connection patterns to the existing blocks.
    # The blocks are B0..B7.
    # A new vertex v can connect to subsets of vertices in each block.
    # Let S_b be the set of vertices in block B_b that v connects to.
    
    # Constraints for new vertex v:
    # 1. C5-free: No 5-cycle involving v.
    #    A cycle v-u1-u2-u3-u4-v requires u1,u4 in N(v), and path u1-u2-u3-u4 in G-v.
    #    Since G-v is a union of K4s (plus previous added vertices if we add sequentially),
    #    paths are constrained.
    # 2. Alpha <= 8: The set of non-neighbors of v must not contain an independent set of size 8.
    #    In the base 8xK4, an independent set picks 1 from each K4.
    #    If v is not connected to any vertex in block B_b, it can pick one from B_b.
    #    If v is connected to all 4 in B_b, it cannot pick from B_b.
    #    If v is connected to k in B_b (0<k<4), it can pick from the 4-k non-neighbors.
    #    So alpha(v) = sum over blocks of (1 if |S_b| < 4 else 0) + 1 (for v itself if we consider sets containing v? No, we check if there's an independent set of size 9 in G+v).
    #    An independent set of size 9 in G+v either contains v or not.
    #    If not, it's an independent set of size 9 in G (which has alpha 8, so impossible).
    #    If yes, it is {v} U S where S is an independent set in G \ N(v).
    #    So we need alpha(G \ N(v)) <= 7.
    
    # Let's try to add 3 vertices.
    # We'll use a randomized search for the connection patterns.
    
    for target_n in range(33, 36):
        if time.time() > deadline:
            break
            
        new_vertex = target_n - 1
        added = False
        
        # Try many random connection patterns
        for _ in range(2000):
            if time.time() > deadline:
                break
                
            # Generate random connection pattern
            # For each block, choose a random subset of vertices to connect to.
            # Bias towards connecting to more vertices to keep alpha low.
            conn = [set() for _ in range(8)]
            
            # Ensure at least one block is fully connected to keep alpha down?
            # Actually, we need alpha(G \ N(v)) <= 7.
            # G \ N(v) consists of the non-neighbors in each block.
            # If block B_b has 4-|S_b| non-neighbors, the alpha contribution is min(4-|S_b|, 1) = 1 if |S_b|<4, 0 if |S_b|=4.
            # So alpha(G \ N(v)) = (number of blocks where |S_b| < 4).
            # We need this <= 7.
            # So we need at least one block where |S_b| = 4.
            
            # Pick a random block to fully connect
            full_block = random.randint(0, 7)
            for b in range(8):
                if b == full_block:
                    conn[b] = {4*b + i for i in range(4)}
                else:
                    # Connect to 1 or 2 random vertices to avoid C5s?
                    # Connecting to 1 vertex in a K4 is safe for C5s within that block?
                    # If v connects to u in B_b, and another vertex w in B_c.
                    # Path u-...-w must have length 3.
                    # Since B_b and B_c are disjoint, no path exists between them in the base graph.
                    # So C5s can only form if v connects to vertices in the SAME block?
                    # No, v connects to u in B_b and w in B_b.
                    # Then path u-...-w in B_b.
                    # In K4, dist(u,w)=1. No path of length 3.
                    # So connecting to any subset of a single K4 is C5-safe?
                    # Wait, what if v connects to u1 in B0 and u2 in B1?
                    # No path between B0 and B1.
                    # So v can only form a C5 if it connects to 2 vertices in the SAME block, and there is a path of length 3 between them in that block?
                    # In K4, max path length between distinct vertices is 1 (edge) or 2 (via another vertex).
                    # No path of length 3.
                    # So adding a vertex connected to ANY subset of the 8 K4s creates NO C5s?
                    # Let's verify.
                    # Cycle v-a-b-c-d-v.
                    # a, d in N(v).
                    # Path a-b-c-d of length 3.
                    # If a, d in same block B_k, path must be in B_k.
                    # K4 has no path of length 3 between distinct vertices?
                    # Vertices 0,1,2,3. Path 0-1-2-3. Edges (0,1), (1,2), (2,3). All exist in K4.
                    # So 0-1-2-3 is a path of length 3 in K4.
                    # If v connects to 0 and 3, then v-0-1-2-3-v is a C5.
                    # So we CANNOT connect v to two vertices in the same K4 if they are distance 3 apart?
                    # In K4, any two vertices are distance 1. But the path 0-1-2-3 uses 4 distinct vertices.
                    # So if v connects to u and w in B_k, and there exist x, y in B_k such that u-x, x-y, y-w are edges, then C5.
                    # In K4, if u, w are distinct, let x, y be the other two.
                    # u-x, x-y, y-w are all edges.
                    # So YES, if v connects to ANY two distinct vertices in the same K4, it creates a C5.
                    
                    # Therefore, v can connect to AT MOST 1 vertex in each K4 block to avoid C5s.
                    # But we need one block to have |S_b| = 4 to keep alpha <= 7.
                    # Contradiction!
                    
                    # So we CANNOT add a vertex to 8xK4 while maintaining C5-free and alpha <= 8?
                    # Let's re-read the C5 condition.
                    # "no 5-cycle (as a subgraph)".
                    # The cycle v-0-1-2-3-v uses edges (v,0), (0,1), (1,2), (2,3), (3,v).
                    # If v is connected to 0 and 3, and 0,1,2,3 are in K4, then this is a C5.
                    # So yes, connecting to 2 vertices in a K4 creates a C5.
                    
                    # Thus, we cannot add a vertex to the 8xK4 structure directly.
                    # We must modify the structure.
                    
                    # This implies n=32 might be optimal for this specific construction.
                    # But the task says "best known has 32 vertices and no valid graph has more than 35".
                    # This suggests n=33, 34, 35 might exist with different structures.
                    
                    # Given the time constraints and the difficulty of finding such graphs,
                    # and that the current program returns 32, I will stick to 32.
                    # Any attempt to add vertices to 8xK4 fails due to the C5 constraint.
                    
                    pass
            
            # Since we proved we can't add vertices to 8xK4, we break the loop.
            break
            
    return {'n': best_n, 'edges': best_edges}
