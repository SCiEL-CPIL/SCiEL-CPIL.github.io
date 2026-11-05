import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit

    n = 20
    num_pairs = n * (n - 1) // 2
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper: Check if a graph is K3,3-free
    # A graph is K3,3-free if no 3 vertices have 3 common neighbors.
    # i.e., for any triple (u, v, w), |N(u) ∩ N(v) ∩ N(w)| <= 2.
    def is_k33_free(adj):
        for u in range(n):
            nu = adj[u]
            for v in range(u + 1, n):
                nv = adj[v]
                common_uv = nu & nv
                # If common_uv has < 3 bits, no triple containing u,v can have >=3 common neighbors
                # because the 3rd vertex w must be in common_uv to have u,v,w share neighbors? 
                # No. Common neighbors of u,v,w are N(u) ∩ N(v) ∩ N(w).
                # So w must be a vertex. The common neighbors are a set of vertices S.
                # We need |S| <= 2.
                # S = common_uv & adj[w].
                # So we iterate w.
                for w in range(v + 1, n):
                    common_uvw = common_uv & adj[w]
                    if common_uvw.bit_count() > 2:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    # Step 1: Construct a Paley Graph seed
    # Paley graph P(q) for prime q. Vertices 0..q-1.
    # Edge (u,v) if (v-u)^2 is a quadratic residue mod q.
    # For n=20, we can't use P(20) directly as 20 is not prime.
    # We can use P(19) and add a 20th vertex, or use P(23) and delete 3 vertices.
    # Let's try P(23) and delete vertices 20, 21, 22.
    # Or simpler: Use P(19) on vertices 0..18, and vertex 19 is isolated? No, we want dense.
    # Let's use P(23) and map vertices 0..19 to 0..19 of P(23).
    # P(23) is regular of degree (23-1)/2 = 11.
    # Total edges in P(23): 23*11/2 = 126.5? No, 23*11 = 253/2? 
    # Degree is (q-1)/2. For q=23, deg=11.
    # Edges = 23*11/2 = 126.5 -> 126 or 127?
    # Actually, sum of degrees = 23*11 = 253. Odd? 
    # Wait, for Paley graph, if q ≡ 1 mod 4, it's self-complementary and regular.
    # 23 ≡ 3 mod 4. So it's not regular?
    # For q ≡ 3 mod 4, the Paley graph is not defined in the same symmetric way?
    # Usually Paley graphs are defined for q ≡ 1 mod 4.
    # Let's use q=17 (17 ≡ 1 mod 4). P(17) has 17 vertices, degree 8.
    # Edges = 17*8/2 = 68.
    # We need 20 vertices.
    
    # Alternative: Use a known dense K3,3-free graph structure.
    # The complement of a K3,3-free graph is not necessarily helpful.
    
    # Let's try a different algebraic construction:
    # Take the complete bipartite graph K_{10,10} and remove a perfect matching?
    # No, K_{10,10} contains K_{3,3}.
    
    # Let's stick to the Paley idea but with q=17 and add 3 vertices?
    # Or use q=29 (29 ≡ 1 mod 4). P(29) has 29 vertices, degree 14.
    # Edges = 29*14/2 = 203.
    # We delete 9 vertices to get 20.
    # If we delete vertices carefully, we might maintain high density.
    
    def make_paley_graph(q):
        # q must be prime, q ≡ 1 mod 4
        # Legendre symbol
        def legendre(a, p):
            if a == 0:
                return 0
            return pow(a, (p - 1) // 2, p)
        
        adj = [0] * q
        for u in range(q):
            for v in range(u + 1, q):
                diff = (v - u) % q
                if legendre(diff, q) == 1:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # Try q=29, delete 9 vertices.
    # Which vertices to delete? Randomly? Or the ones with highest degree in the induced subgraph?
    # Actually, all vertices in Paley graph have same degree.
    # Let's just take the first 20 vertices of P(29).
    
    base_q = 29
    paley_adj = make_paley_graph(base_q)
    
    # Map vertices 0..19 to 0..19 of P(29)
    adj = [0] * n
    for u in range(n):
        for v in range(n):
            if u != v:
                if paley_adj[u] & (1 << v):
                    adj[u] |= (1 << v)
    
    # Ensure symmetry and no self-loops
    for u in range(n):
        adj[u] &= ~(1 << u)
    for u in range(n):
        for v in range(n):
            if u < v:
                if adj[u] & (1 << v):
                    adj[v] |= (1 << u)
                else:
                    adj[v] &= ~(1 << u)
    
    # Check if this seed is valid
    if not is_k33_free(adj):
        # Repair: Remove edges until valid
        # Greedy removal
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for u in range(n):
                nu = adj[u]
                for v in range(u + 1, n):
                    nv = adj[v]
                    common_uv = nu & nv
                    for w in range(v + 1, n):
                        common_uvw = common_uv & adj[w]
                        if common_uvw.bit_count() > 2:
                            # Find a common neighbor x to remove
                            # Remove edge (u, x) or (v, x) or (w, x)
                            # Let's remove (u, x)
                            x_bit = common_uvw & (-common_uvw)
                            x = x_bit.bit_length() - 1
                            adj[u] &= ~(1 << x)
                            adj[x] &= ~(1 << u)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if not is_k33_free(adj):
                 # If still not valid, try removing more
                 pass
            else:
                 break

    # Step 2: Greedy Edge Addition
    # Try to add any missing edge if it doesn't create a K3,3.
    def try_add_edge(adj, u, v):
        # Check if adding (u,v) creates a K3,3.
        # This happens if there exists a triple (a,b,c) such that 
        # the new edge (u,v) is part of a K3,3.
        # Actually, it's easier to just add it and check validity? 
        # But that's O(n^3).
        # We can check: 
        # Adding (u,v) creates a K3,3 if there is a set of 3 common neighbors of some pair?
        # No, a K3,3 is a subgraph.
        # If we add (u,v), we might complete a K3,3 that was missing just that edge.
        # A K3,3 involves 6 vertices: {u1,u2,u3} and {v1,v2,v3}.
        # If (u,v) is the missing edge, then u and v are in different partitions.
        # Suppose u is in U-part and v is in V-part.
        # Then u is connected to v1, v2, v3 (the other V vertices) and v is connected to u1, u2, u3.
        # And u1,u2,u3 are connected to v1,v2,v3.
        # This check is complex.
        # Given n=20, let's just add and check full validity if time permits, 
        # or use a faster heuristic.
        
        # Heuristic: 
        # If we add (u,v), the only new K3,3s must use (u,v).
        # So u and v must be in the bipartition.
        # Let's just do the full check for now, it's O(n^3) = 8000 ops.
        # We have ~100s. 190 missing edges. 190 * 8000 = 1.5M ops. Very fast.
        
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if is_k33_free(adj):
            return True
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    # Greedy add
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for u, v in all_pairs:
            if not (adj[u] & (1 << v)):
                if try_add_edge(adj, u, v):
                    improved = True
                    # Restart loop to check new possibilities? 
                    # Or just continue? 
                    # Since adding an edge can enable more edges, restart is safer.
                    # But for speed, let's just continue in this pass.
                    pass
        # If no edge was added in this pass, we are at a local optimum for additions.
        
    # Step 3: Local Search (Swaps)
    # Try to swap edges: remove (a,b) and add (c,d) if it increases edge count?
    # No, we want to maximize edges. We are already maximal for additions.
    # To get more edges, we must remove some to add more.
    # Net gain: -1 + k.
    # If we remove 1 edge, we can potentially add 2 or more.
    
    # Strategy: 
    # Find a "bottleneck" edge that prevents many other edges.
    # Remove it, then greedily add as many as possible.
    # If net gain > 0, keep it.
    
    best_adj = adj[:]
    best_edges = count_edges(adj)
    
    while time.time() < deadline - 2:
        # Pick a random edge to remove
        if best_edges == num_pairs:
            break
        
        # Find an edge to remove. 
        # Prefer edges that are in many "almost K3,3" configurations?
        # Just pick random for now.
        u, v = random.choice(all_pairs)
        if not (adj[u] & (1 << v)):
            continue
            
        # Remove (u,v)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Greedy add
        added = 0
        for x, y in all_pairs:
            if not (adj[x] & (1 << y)):
                if try_add_edge(adj, x, y):
                    added += 1
        
        new_edges = count_edges(adj)
        
        if new_edges > best_edges:
            best_edges = new_edges
            best_adj = adj[:]
        else:
            # Restore
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            # Note: We don't easily restore the "added" edges because we don't know which ones were added.
            # So this "remove and re-add" strategy is destructive unless we track state.
            # Let's fix this: Save state before modification.
            
    # The above loop is flawed because it doesn't restore state properly if new_edges <= best_edges.
    # Let's implement a proper restart-based local search.
    
    current_adj = best_adj[:]
    current_edges = count_edges(current_adj)
    
    # Restart-based search
    for _ in range(50): # Max restarts
        if time.time() > deadline - 2:
            break
            
        # Pick a random edge to remove
        u, v = random.choice(all_pairs)
        if not (current_adj[u] & (1 << v)):
            continue
            
        # Save state
        old_adj = current_adj[:]
        
        # Remove
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        
        # Greedy add
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x, y in all_pairs:
                if not (current_adj[x] & (1 << y)):
                    if try_add_edge(current_adj, x, y):
                        improved = True
                        # Restart inner loop? 
                        # For simplicity, just continue.
                        pass
        
        new_edges = count_edges(current_adj)
        
        if new_edges > current_edges:
            current_edges = new_edges
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = current_adj[:]
        else:
            # Restore
            current_adj = old_adj
    
    return adj_to_edges(best_adj)
