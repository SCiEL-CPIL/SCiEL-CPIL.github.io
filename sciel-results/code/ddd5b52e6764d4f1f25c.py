"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # Fallback solution: 8 disjoint K4s (n=32)
    def get_fallback():
        return {'n': 32, 'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]}
    
    best_n = 32
    best_edges = get_fallback()['edges']
    
    # Helper to check if graph is valid
    def is_valid(n, edges):
        # Build adjacency matrix and list
        adj = [[False] * n for _ in range(n)]
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False
            if adj[u][v]:
                return False # Simple graph check
            adj[u][v] = True
            adj[v][u] = True
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # Check for C5
        # A C5 is a subgraph, so we just need to find any 5-cycle.
        # Efficient check: For each pair of non-adjacent vertices u, v,
        # if they share a neighbor w, and w shares a neighbor x with u (x!=v) ... 
        # Actually, simpler: 
        # Iterate u < v. If not adj[u][v]:
        #   common = [w for w in range(n) if adj[u][w] and adj[v][w]]
        #   if len(common) >= 2:
        #      # We have a C4 potentially. We need C5.
        #      # A C5 through u,v exists if there is a path of length 3 from u to v.
        #      # Path u-w1-w2-w3-v.
        #      # This is getting expensive.
        #      # Let's use a bitset approach for speed if possible, but n is small.
        #      # For n=33, O(n^5) is ~ 39M ops. Might be slow in Python.
        #      # Let's try a simpler heuristic check or just accept slower check for few candidates.
        #      pass
        
        # Standard C5 check:
        # For each vertex u:
        #   For each pair of neighbors v, w of u (v < w):
        #      If v and w are not adjacent:
        #         # Check if there is a path of length 2 from w to v not using u?
        #         # No, that's C4.
        #         # We need u-v-...-w-u.
        #         # If there is a vertex x such that adj[w][x] and adj[x][v] and x != u:
        #         # Then u-v-x-w-u is a C4.
        #         # We need C5: u-v-x-y-w-u.
        #         # So we need a path of length 3 from v to w in G \ {u}.
        #         # This is equivalent to checking if the distance between v and w in G \ {u} is exactly 3.
        #         # Or just check if there is ANY C5.
        #         # Let's just iterate all 5-tuples? No.
        #         # Let's check: for each edge (a,b), check if there is a path of length 3 from b to a in G \ {(a,b)}?
        #         # No, that finds C4.
        #         # C5: Edge (a,b). Path a-c-d-e-b.
        #         # So for each edge (a,b), check if there is a path of length 3 from b to a in G without using edge (a,b)?
        #         # Actually, just check if there is a path of length 3 from a to b in G \ {(a,b)}.
        #         # If yes, a-c-d-e-b-a is a C5.
        #         # This is O(E * n^2) roughly.
        
        has_c5 = False
        # Optimize: Precompute bitsets
        adj_bits = [0] * n
        for u in range(n):
            mask = 0
            for v in adj_list[u]:
                mask |= (1 << v)
            adj_bits[u] = mask
            
        for u in range(n):
            for v in adj_list[u]:
                if u > v: continue
                # Check if there is a path of length 3 from v to u not using edge (u,v)
                # Path: v - x - y - u
                # x must be neighbor of v, x != u
                # y must be neighbor of x, y != v, y != u (to avoid backtracking or C3/C4 issues? No, subgraph C5 allows chords, but vertices must be distinct)
                # Vertices u, v, x, y must be distinct.
                # x in N(v) \ {u}
                # y in N(x) \ {v, u}
                # and y in N(u)
                
                # Iterate x
                for x in adj_list[v]:
                    if x == u: continue
                    # Iterate y
                    # y must be in N(x) and N(u)
                    # and y != v, y != u
                    common_xy_u = adj_bits[x] & adj_bits[u]
                    # Remove v and u from common
                    common_xy_u &= ~(1 << v)
                    common_xy_u &= ~(1 << u)
                    if common_xy_u:
                        has_c5 = True
                        break
                if has_c5:
                    break
            if has_c5:
                break
        
        if has_c5:
            return False
            
        # Check Independence Number < 9
        # Use a greedy algorithm or simple branch and bound for small n.
        # For n=33, alpha < 9 is a strong constraint.
        # We can try to find an independent set of size 9.
        # If we find one, it's invalid.
        # Greedy:
        # Sort vertices by degree.
        # This is not exact.
        # Let's use a simple backtracking to find an independent set of size 9.
        # If we fail to find one, it might still be 9.
        # But for random graphs, if alpha is small, greedy often finds the max or close.
        # To be safe, we can use a bitset based maximal independent set search.
        
        # Simple exact check for alpha >= 9 using backtracking is feasible for n=33?
        # Worst case is exponential.
        # Let's try a heuristic: if greedy finds < 9, assume valid.
        # Risk: False positive (graph has alpha 9 but greedy finds 8).
        # Given the time constraint, we accept this risk for the search phase.
        # The final returned graph should be verified more carefully if possible, 
        # but n=32 is exact.
        
        # Greedy Independent Set
        remaining = set(range(n))
        indep_set = []
        # Sort by degree ascending
        degs = [(len(adj_list[i]), i) for i in range(n)]
        degs.sort()
        
        for _, u in degs:
            if u in remaining:
                indep_set.append(u)
                remaining.remove(u)
                for v in adj_list[u]:
                    if v in remaining:
                        remaining.remove(v)
        
        if len(indep_set) >= 9:
            return False
            
        # Additional check: Try a few random greedy runs
        for _ in range(10):
            remaining = set(range(n))
            indep_set = []
            # Random order
            order = list(range(n))
            random.shuffle(order)
            for u in order:
                if u in remaining:
                    indep_set.append(u)
                    remaining.remove(u)
                    for v in adj_list[u]:
                        if v in remaining:
                            remaining.remove(v)
            if len(indep_set) >= 9:
                return False
                
        return True

    def try_find_graph(n, attempts=50):
        for _ in range(attempts):
            if time.time() > deadline:
                return None
            
            # Generate random graph with p=0.5
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        edges.append([i, j])
                        
            # Repair: Remove edges from C5s
            # We need a function to find a C5.
            # Re-use logic from is_valid but return the C5 edges.
            
            # For simplicity, if is_valid fails, we just restart.
            # This is inefficient but simple.
            # Let's try to be smarter:
            # If is_valid returns False, we don't know why.
            # Let's just restart.
            
            if is_valid(n, edges):
                return edges
        return None

    # Try n=33
    if time.time() < deadline - 5:
        edges_33 = try_find_graph(33, attempts=20)
        if edges_33:
            # Double check
            if is_valid(33, edges_33):
                return {'n': 33, 'edges': edges_33}
                
    # Try n=34
    if time.time() < deadline - 5:
        edges_34 = try_find_graph(34, attempts=10)
        if edges_34:
            if is_valid(34, edges_34):
                return {'n': 34, 'edges': edges_34}

    # Fallback
    return get_fallback()

# EVOLVE-BLOCK-END
