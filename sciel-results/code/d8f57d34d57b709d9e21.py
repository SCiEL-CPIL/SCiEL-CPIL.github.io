import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 150.0  # Safety margin for 150s limit
    
    # Fallback to the known valid seed solution
    def seed_solution():
        return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
    
    best_graph = seed_solution()
    
    def has_c5(n, adj_mask):
        """
        Check if the graph has a C5 subgraph.
        A C5 is a cycle of length 5. 
        We check for a path u-x-y-v where u and v are not adjacent, 
        and x and y are not adjacent. Then u-x-y-v-u forms a C5.
        """
        # Precompute non-adjacent pairs? No, just iterate.
        # For efficiency, we can iterate u < v, and check if u-v is NOT an edge.
        # Then check if there is a path of length 3 between u and v.
        
        # Optimization: Iterate over all pairs (u, v) with u < v.
        # If u and v are NOT connected, check if dist(u, v) == 3.
        # Actually, checking dist==3 is equivalent to:
        # Exists x in N(u) and y in N(v) such that x-y is an edge? No, that's length 3 path u-x-y-v.
        # Wait, the cycle is u-x-y-v-u. So u-v must be an edge? No, the cycle is u-x-y-v-w-u.
        # Let's stick to the standard check: 
        # A C5 exists if there is a path of length 4 between two adjacent vertices?
        # Or simply: for every edge (u, v), is there a path of length 4 between u and v?
        
        # Let's use the property: C5-free implies for any edge (u,v), N(u) \ {v} and N(v) \ {u} are disjoint? No.
        
        # Robust method:
        # For each pair of non-adjacent vertices (u, v), check if there is a path of length 3.
        # If yes, u-x-y-v-u is a C5.
        
        for u in range(n):
            for v in range(u + 1, n):
                # If u and v are adjacent, they cannot form the "missing" edge of the C5 in this specific check logic
                # But a C5 can have any edge. 
                # Let's check: Is there a path of length 3 between u and v?
                # If u and v are NOT adjacent, and there is a path u-x-y-v, then u-x-y-v-u is a C5.
                if not (adj_mask[u] & (1 << v)):
                    # Check for path of length 3: u - x - y - v
                    # x is in N(u), y is in N(v), and x-y is an edge.
                    # Also x != v, y != u (implied by u-v non-edge and simple graph usually, but x can be v? No, x in N(u), if x=v then u-v edge. So x!=v. Similarly y!=u.)
                    # x and y must be distinct? If x=y, then u-x-v is a path of length 2. 
                    # If u-x-v is a path, and u-v is not an edge, that's a triangle? No, u-x-v is just a path.
                    # If x=y, we have u-x-v. This is length 2.
                    # The cycle would be u-x-v-x-u? No, that's not a simple cycle.
                    # So we need x != y.
                    
                    # We need to find x in N(u) and y in N(v) such that x-y is an edge and x != y.
                    # Actually, if x=y, then u-x-v is a path of length 2.
                    # Does u-x-v-x-u form a C5? No.
                    # So we strictly need x != y.
                    
                    nu = adj_mask[u]
                    nv = adj_mask[v]
                    
                    # Iterate x in N(u)
                    x = nu
                    while x:
                        xb = x & -x
                        xi = xb.bit_length() - 1
                        x ^= xb
                        
                        # y must be in N(v) and N(x)
                        # y is in N(v) AND N(x)
                        # Also y != u (since x in N(u), if y=u then x-u edge, so x in N(u) is consistent, but y=u means v-u path? No, y is neighbor of v. If y=u, then v-u edge, which we already checked is false. So y!=u automatically.)
                        # Also y != x? If y=x, then x is in N(v) and N(x) (self loop? No). 
                        # If y=x, then x is in N(v). So x is neighbor of u and v.
                        # Then u-x-v is a path of length 2.
                        # We need a path of length 3: u-x-y-v.
                        # If y=x, it's u-x-v. Length 2.
                        # So we need y != x.
                        
                        common = nv & adj_mask[xi]
                        # Remove x from common? 
                        # adj_mask[xi] does not contain xi (no self loops).
                        # So common does not contain x.
                        # So any y in common is distinct from x.
                        
                        if common:
                            return True
                            
        return False

    def has_independent_set_9(n, adj_mask):
        """
        Check if there is an independent set of size 9.
        """
        # Backtracking with bitmasks
        # candidates: set of vertices that can be added to the current independent set
        # current_size: number of vertices in the current independent set
        
        # We want to find if there is a set of 9 vertices with no edges between them.
        # This is equivalent to finding a clique of size 9 in the complement graph.
        
        # Let's use the standard Bron-Kerbosch style or simple DFS on the complement.
        # Complement graph adjacency: c_adj[u] = ~(adj_mask[u] | (1<<u)) & ((1<<n)-1)
        
        all_mask = (1 << n) - 1
        c_adj = [0] * n
        for u in range(n):
            c_adj[u] = all_mask ^ adj_mask[u] ^ (1 << u)
            
        def dfs(candidates, current_size):
            if current_size == 9:
                return True
            if not candidates:
                return False
            
            # Heuristic: pick a vertex with high degree in the complement? 
            # Or just pick the first one.
            # To be faster, we can bound: if current_size + popcount(candidates) < 9, return False.
            if current_size + bin(candidates).count('1') < 9:
                return False
            
            # Pick a vertex v from candidates
            # Choosing the vertex with the highest degree in the induced subgraph of candidates 
            # might be better, but for n=35, simple selection is often enough.
            # Let's pick the vertex with the highest index or lowest index.
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            # Option 1: Include v
            # New candidates: candidates & c_adj[v]
            if dfs(candidates & c_adj[v], current_size + 1):
                return True
                
            # Option 2: Exclude v
            candidates ^= v_bit
            if dfs(candidates, current_size):
                return True
                
            return False
            
        return dfs(all_mask, 0)

    def is_valid(n, edges):
        # Build adj masks
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        if has_c5(n, adj_mask):
            return False
        if has_independent_set_9(n, adj_mask):
            return False
        return True

    # Search strategy
    # Try n from 35 down to 29
    # For each n, try random edge counts
    # C5-free max edges for n=35 is approx 35 * (35/3)^(1/3) ? 
    # Kim's bound: ex(n, C5) <= c n^(4/3). For n=35, 35^(4/3) ~ 260.
    # So m should be around 150-250.
    
    rng = random.Random(42)
    
    for n in range(35, 28, -1):
        # If time is running out, break
        if time.time() > deadline - 10:
            break
            
        # Try several random graphs for this n
        # We want high density to minimize independence number, but low enough to avoid C5.
        # Let's try m from 200 down to 100? Or randomize.
        # Actually, to minimize alpha, we want dense. To avoid C5, we want sparse.
        # There is a sweet spot.
        
        attempts = 0
        while time.time() < deadline - 5 and attempts < 1000:
            attempts += 1
            
            # Random m
            # Let's bias towards higher m
            m = rng.randint(150, 250)
            
            # Generate random graph
            # Use G(n, p) or G(n, m). G(n, m) is easier to control.
            # To avoid duplicates, use a set of edges.
            # But for speed, we can just generate random pairs and check validity.
            # If m is too high, it's likely to have C5.
            
            edges = []
            edge_set = set()
            
            # Generate m random edges
            # For n=35, total possible edges = 35*34/2 = 595.
            # We need m edges.
            # Random sampling without replacement.
            # Since 595 is small, we can just shuffle all possible edges and take first m.
            # But doing this 1000 times might be slow if not optimized.
            # Let's just generate random pairs and add to set until size m.
            
            while len(edge_set) < m:
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) not in edge_set:
                    edge_set.add((u, v))
                    edges.append([u, v])
            
            # Check validity
            if is_valid(n, edges):
                best_graph = {'n': n, 'edges': edges}
                # Found a valid graph for this n. 
                # We can try to find a better one? No, we want max n.
                # Since we iterate n downwards, the first valid n is the best.
                # We can break the inner loop and return immediately?
                # Yes, because we are looking for the maximum n.
                return best_graph
                
    # If no better graph found, return the seed
    return best_graph
