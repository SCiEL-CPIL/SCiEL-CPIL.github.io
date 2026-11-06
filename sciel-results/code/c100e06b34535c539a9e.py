import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Helper to calculate cost (K4 in G + K6 in G_complement)
    def calculate_cost(adj, n):
        # adj is a list of integers (bitmasks)
        # Count K4 in G
        k4_count = 0
        # Count K6 in G_complement (Independent sets of size 6 in G)
        # We can compute this on the complement
        # Comp adjacency: comp_adj[i] = ((1 << n) - 1) ^ (1 << i) ^ adj[i]
        
        # To save time, we can compute K4 in G and K6 in G' separately
        # Or just K4 in G and K6 in G'
        
        # K4 in G:
        # Sum over edges (u,v) of edges in G[common_neighbors]
        # common = adj[u] & adj[v]
        # We need count of edges in subgraph induced by 'common'
        
        # K6 in G':
        # Sum over edges (u,v) in G' of edges in G'[common_neighbors_in_G']
        
        # This is expensive if done naively.
        # For n<=40, we can precompute neighbor lists?
        # Let's just implement the bitmask count.
        
        total_k4 = 0
        total_k6 = 0
        
        # Precompute bit counts?
        # For each pair u < v:
        for u in range(n):
            for v in range(u + 1, n):
                # Check if edge (u,v) in G
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Count edges in G induced by 'common'
                    # For each w in common, count edges from w to common
                    # Sum of degrees in induced subgraph / 2
                    # Or simply: for each w in common, count bits in (common & adj[w])
                    # This is O(n) per edge. Total O(E * n) ~ 40^3 = 64000 ops. Very fast.
                    e_count = 0
                    # Iterate through bits in common
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        # Count neighbors of w in common
                        e_count += (common & adj[w]).bit_count()
                        c ^= lsb
                    total_k4 += e_count // 2 # Each edge counted twice? 
                    # Wait. 
                    # If we sum over w in common the number of neighbors in common, 
                    # each edge (x,y) in common is counted once for x and once for y.
                    # So sum is 2 * |E(G[common])|.
                    # So |E(G[common])| = sum / 2.
                    # And K4 count contribution is |E(G[common])|.
                    # So total_k4 += (e_count // 2)
                    
                # Check if edge (u,v) in G' (i.e., NOT in G)
                else:
                    # u,v not connected in G.
                    # In G', u and v are connected.
                    # Common neighbors in G' are vertices w such that w is NOT connected to u in G AND w is NOT connected to v in G.
                    # w != u, w != v.
                    # mask_all = (1 << n) - 1
                    # not_u = mask_all ^ (1 << u) ^ adj[u]  # Non-neighbors of u (excluding u)
                    # not_v = mask_all ^ (1 << v) ^ adj[v]
                    # common_comp = not_u & not_v
                    # We need to count edges in G' induced by common_comp.
                    # Edge (x,y) in G' means x,y NOT connected in G.
                    # So we need to count pairs (x,y) in common_comp such that NOT (adj[x] & (1<<y)).
                    
                    mask_all = (1 << n) - 1
                    # Exclude u and v from consideration? 
                    # common_comp should not include u or v because u is neighbor of u in G' (self-loop? no, simple graph).
                    # In G', u is not connected to u.
                    # not_u excludes u.
                    # So common_comp is correct.
                    
                    not_u = mask_all ^ (1 << u) ^ adj[u]
                    not_v = mask_all ^ (1 << v) ^ adj[v]
                    common_comp = not_u & not_v
                    
                    # Count edges in G' induced by common_comp
                    # An edge (x,y) in G' exists if (adj[x] & (1<<y)) == 0
                    # Iterate x in common_comp
                    e_count_comp = 0
                    c = common_comp
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        # Neighbors of x in common_comp in G'
                        # These are y in common_comp such that NOT (adj[x] & (1<<y))
                        # Let's find non-neighbors of x in common_comp
                        # non_neigh_x = common_comp & (mask_all ^ (1 << x) ^ adj[x])
                        # But we only care about those in common_comp.
                        # So count = common_comp & (mask_all ^ (1 << x) ^ adj[x])
                        # Note: adj[x] does not contain x.
                        non_neigh_x_in_common = common_comp & (mask_all ^ (1 << x) ^ adj[x])
                        e_count_comp += non_neigh_x_in_common.bit_count()
                        c ^= lsb
                    # Each edge (x,y) in G' is counted twice (once for x, once for y)
                    total_k6 += e_count_comp // 2
                    
        # K4 in G = total_k4 / 6 ?
        # Let's re-verify the factor.
        # Sum over edges (u,v) in G of |E(G[N(u) cap N(v)])|.
        # Each K4 {a,b,c,d} contributes:
        # Edge (a,b): common {c,d}, edge (c,d) exists. Count 1.
        # Edge (a,c): common {b,d}, edge (b,d) exists. Count 1.
        # ... 6 edges total.
        # So Sum = 6 * N(K4).
        # So N(K4) = total_k4 / 6.
        
        # K6 in G' = total_k6 / 15 ?
        # Sum over edges (u,v) in G' of |E(G'[N'(u) cap N'(v)])|.
        # Each K6 in G' has 15 edges.
        # Each edge contributes 1 to the sum (the other 4 vertices form a K4 in G'? No.)
        # Wait.
        # For K_k, the formula is Sum over edges (u,v) of |E(G[N(u) cap N(v)])| / (k-1 choose 2)?
        # Let's check K4. k=4. (k-1 choose 2) = 3 choose 2 = 3.
        # If N(K4) = Sum / 3, then Sum = 3 * N(K4).
        # But we found Sum = 6 * N(K4).
        # Why?
        # In the sum, for edge (u,v), we count edges in the common neighborhood.
        # A K4 has 4 vertices. Pick an edge (u,v). The other 2 vertices x,y are in common neighborhood.
        # Do x,y form an edge? Yes, because it's a K4.
        # So for each of the 6 edges in K4, we count 1.
        # So Sum = 6 * N(K4).
        # So N(K4) = Sum / 6.
        
        # For K6 in G'.
        # Sum over edges (u,v) in G' of |E(G'[N'(u) cap N'(v)])|.
        # Pick a K6 in G'. Vertices v1..v6.
        # Pick an edge (u,v) from this K6. Say (v1, v2).
        # Common neighbors in G' are v3, v4, v5, v6.
        # We count edges in G' induced by {v3, v4, v5, v6}.
        # Since it's a K6, all edges exist.
        # Number of edges in K4 is 6.
        # So for edge (v1, v2), we count 6.
        # There are 15 edges in K6.
        # Total Sum = 15 * 6 = 90 * N(K6).
        # So N(K6) = total_k6 / 90.
        
        k4_count = total_k4 // 6
        k6_count = total_k6 // 90
        
        return k4_count + k6_count

    # SA Parameters
    def run_sa(adj, n, time_limit):
        current_cost = calculate_cost(adj, n)
        best_adj = adj[:]
        best_cost = current_cost
        
        T = 1.0
        cooling = 0.995
        
        t_start = time.time()
        while time.time() - t_start < time_limit:
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Check current state
            if adj[u] & (1 << v):
                # Remove edge
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                new_cost = calculate_cost(adj, n)
                # If we removed, we are in "remove" mode
                # To revert, we add it back
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_adj = adj[:]
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            else:
                # Add edge
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                new_cost = calculate_cost(adj, n)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_adj = adj[:]
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            
            T *= cooling
            if T < 0.01:
                break
        
        return best_adj, best_cost

    # Initialize with Paley(17)
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Verify initial cost
    cost = calculate_cost(adj, n)
    # Paley(17) should have 0 K4 and 0 alpha 6?
    # Paley(17) is (17, 8, 4, 3, 3)? No, parameters are (v, k, lambda, mu).
    # v=17, k=8.
    # It is known to be K4-free? 
    # Actually, Paley(17) has cliques of size 3? Or 4?
    # If cost > 0, we run SA to fix it first.
    if cost > 0:
        adj, cost = run_sa(adj, n, 2.0)
    
    best_valid_n = n
    best_valid_adj = adj[:]
    
    # Try to extend
    for target_n in range(n + 1, 36):
        if time.time() > deadline - 10:
            break
        
        # Extend graph
        new_adj = adj[:] + [0]
        new_n = target_n
        
        # Randomize connections of new vertex
        for i in range(new_n - 1):
            if random.random() < 0.5:
                new_adj[i] |= (1 << (new_n - 1))
                new_adj[new_n - 1] |= (1 << i)
        
        # Run SA to minimize cost
        # Give more time for larger n?
        time_slice = min(5.0, (deadline - time.time()) / 2)
        new_adj, new_cost = run_sa(new_adj, new_n, time_slice)
        
        if new_cost == 0:
            # Success
            adj = new_adj
            n = new_n
            best_valid_n = n
            best_valid_adj = adj[:]
        else:
            # Failed to find valid extension
            # Try a few more times with different random seeds?
            # Or just break?
            # Let's try 3 more times with shorter time
            success = False
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                # Re-randomize new vertex
                for i in range(new_n - 1):
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << (new_n - 1))
                        new_adj[new_n - 1] |= (1 << i)
                    else:
                        new_adj[i] &= ~(1 << (new_n - 1))
                        new_adj[new_n - 1] &= ~(1 << i)
                
                time_slice = min(2.0, (deadline - time.time()) / 3)
                new_adj, new_cost = run_sa(new_adj, new_n, time_slice)
                if new_cost == 0:
                    adj = new_adj
                    n = new_n
                    best_valid_n = n
                    best_valid_adj = adj[:]
                    success = True
                    break
            
            if not success:
                break

    # Return best valid graph
    edges = []
    for i in range(best_valid_n):
        for j in range(i + 1, best_valid_n):
            if best_valid_adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': best_valid_n, 'edges': edges}
