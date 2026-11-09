import time
import random
import sys

def construct():
    """
    Attempts to find a graph with n vertices (n > 37) that is (5,5)-free.
    Uses local search to minimize the number of K5s and independent 5-sets.
    Falls back to the Paley graph on 37 vertices if no better graph is found.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 20s margin
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
    
    # Fallback: Paley graph on 37 vertices
    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges
    
    best_n = 37
    best_edges = get_paley_37()[1]
    
    # Try to find graphs with n = 45, 44, 43, 42
    # We try larger n first to maximize score, but 42 is known to exist.
    # If we find 45, great. If not, 44, etc.
    target_ns = [45, 44, 43, 42]
    
    for n in target_ns:
        if time_left() < 30:
            break
            
        found = False
        # Multiple restarts for each n
        max_restarts = 5
        for r in range(max_restarts):
            if time_left() < 10:
                break
                
            # Initialize random graph
            # adj[i] is a bitmask of neighbors of i
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Precompute bit masks for vertices
            masks = [1 << i for i in range(n)]
            
            # Function to count triangles in a subgraph defined by bitmask
            def count_triangles(bits):
                cnt = 0
                b = bits
                while b:
                    lsb = b & -b
                    i = lsb.bit_length() - 1
                    b ^= lsb
                    neigh = adj[i] & b
                    nb = neigh
                    while nb:
                        lsb2 = nb & -nb
                        j = lsb2.bit_length() - 1
                        nb ^= lsb2
                        cnt += (adj[j] & neigh).bit_count()
                return cnt // 2
            
            # Function to count independent triangles (triangles in complement)
            def count_ind_triangles(bits):
                # Independent triangle in G is a triangle in G_bar
                # G_bar edges are where adj has 0
                # We can simulate by using complement masks
                # comp_adj[i] = full_mask ^ adj[i] ^ (1<<i)
                # But we don't have full_mask handy in inner loop?
                # full_mask = (1<<n)-1
                # Let's just count triangles in the complement subgraph.
                cnt = 0
                b = bits
                while b:
                    lsb = b & -b
                    i = lsb.bit_length() - 1
                    b ^= lsb
                    # Neighbors of i in the complement within 'bits'
                    # i is connected to j in G_bar if NOT (i~j in G)
                    # adj[i] has 1s for neighbors in G.
                    # So non-neighbors in bits are bits & ~(adj[i] | masks[i])
                    non_neigh = bits & ~(adj[i] | masks[i])
                    nb = non_neigh
                    while nb:
                        lsb2 = nb & -nb
                        j = lsb2.bit_length() - 1
                        nb ^= lsb2
                        # Check if j and k are connected in G_bar?
                        # We need to count triangles in G_bar induced by bits.
                        # This logic is getting complex. 
                        # Alternative: Just count K5s and IndK5s directly for initial eval?
                        # No, we need incremental.
                        # Let's stick to the previous logic: 
                        # IndK5 count is K5 count in complement.
                        # So we need count_triangles in the complement graph.
                        # We can pass a flag or use a separate adj list for complement?
                        # For simplicity and speed, let's just use the same function 
                        # but with complement adj.
                        # We'll precompute comp_adj.
                        pass
                return 0 # Placeholder, will fix below
            
            # Let's refactor to support both G and G_bar efficiently.
            # We'll maintain adj and comp_adj.
            comp_adj = [0] * n
            full_mask = (1 << n) - 1
            for i in range(n):
                comp_adj[i] = full_mask ^ adj[i] ^ masks[i]
            
            def count_triangles_in(bits, adj_list):
                cnt = 0
                b = bits
                while b:
                    lsb = b & -b
                    i = lsb.bit_length() - 1
                    b ^= lsb
                    neigh = adj_list[i] & b
                    nb = neigh
                    while nb:
                        lsb2 = nb & -nb
                        j = lsb2.bit_length() - 1
                        nb ^= lsb2
                        cnt += (adj_list[j] & neigh).bit_count()
                return cnt // 2

            def total_energy():
                # Count K5s: sum over all 4-cliques? No, sum over all pairs (u,v) of triangles in N(u)v?
                # K5s = sum_{u<v} count_triangles(N(u) & N(v))
                # This is O(n^2 * T) where T is triangle count cost.
                # n=45, n^2=2000. T~100. 200k ops. 
                # This is fast enough for initial eval and occasional checks.
                e = 0
                for u in range(n):
                    for v in range(u + 1, n):
                        common = adj[u] & adj[v]
                        e += count_triangles_in(common, adj)
                        # For independent sets, use comp_adj
                        common_comp = comp_adj[u] & comp_adj[v]
                        e += count_triangles_in(common_comp, comp_adj)
                return e

            # Initial energy
            current_e = total_energy()
            if current_e == 0:
                found = True
                break
            
            # Local Search
            # We flip edges to reduce energy.
            # Delta E for flipping (u,v):
            # If 1->0: 
            #   Removes K5s containing (u,v): count_triangles(N(u)&N(v))
            #   Adds IndK5s containing (u,v): count_triangles(IndN(u)&IndN(v))
            #   Delta = -tri_G + tri_Comp
            # If 0->1:
            #   Adds K5s: tri_G
            #   Removes IndK5s: tri_Comp
            #   Delta = tri_G - tri_Comp
            
            steps = 5000
            for _ in range(steps):
                if time_left() < 5:
                    break
                
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                is_edge = (adj[u] & masks[v]) != 0
                
                # Calculate delta
                # Common neighbors in G
                common_g = adj[u] & adj[v]
                tri_g = count_triangles_in(common_g, adj)
                
                # Common non-neighbors in G (neighbors in G_bar)
                # Note: u and v are not in their own adj lists.
                # In comp_adj, u and v are connected if they were not in G.
                # common_comp = comp_adj[u] & comp_adj[v]
                common_c = comp_adj[u] & comp_adj[v]
                tri_c = count_triangles_in(common_c, comp_adj)
                
                if is_edge:
                    # 1 -> 0
                    delta = -tri_g + tri_c
                else:
                    # 0 -> 1
                    delta = tri_g - tri_c
                
                if delta < 0:
                    # Accept
                    current_e += delta
                    # Flip edge
                    adj[u] ^= masks[v]
                    adj[v] ^= masks[u]
                    comp_adj[u] ^= masks[v]
                    comp_adj[v] ^= masks[u]
                    
                    if current_e == 0:
                        found = True
                        break
                # Optional: Metropolis acceptance for delta > 0 to escape local minima
                # if random.random() < math.exp(-delta / temp): ...
                # But for now, simple hill climbing.
            
            if found:
                break
        
        if found:
            # Extract edges
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & masks[j]:
                        edges.append((i, j))
            best_n = n
            best_edges = edges
            # We found a valid graph for this n. 
            # Since we iterate n in descending order, this is the best so far.
            # We can break if we found 45, but let's check if we can do better? 
            # No, 45 is the max we try. 
            # If we found 42, we might want to try 43? 
            # The loop structure tries 45, then 44... 
            # If we found 45, we break the outer loop? 
            # Yes, because 45 is the best possible in our list.
            if n == 45:
                break
            # If we found 42, we should stop trying 41 etc, but we don't try 41.
            # We should continue to see if we can find 43? 
            # The loop is `for n in [45, 44, 43, 42]`.
            # If we found 42, the loop ends.
            # If we found 43, the loop continues to 42? No, 43 > 42.
            # We want the MAX n.
            # So if we find 43, we should stop? 
            # No, we should try 45 first. If 45 fails, try 44.
            # So the order [45, 44, 43, 42] is correct.
            # If we find 45, we break.
            # If we find 44, we break? No, we found 44. Is there a 45? 
            # We already tried 45 and failed (or didn't find it).
            # So we should accept the first (highest) n we find.
            break
        
    return best_n, best_edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Found graph with {n} vertices")
