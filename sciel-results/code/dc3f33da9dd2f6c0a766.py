import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s margin for safety/overhead

    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def find_k4(n, adj):
        """Find a K4 in the graph. Returns tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[b] >> a) & 1):
                    continue
                # Common neighbors of a and b, with index > b
                common = adj[a] & adj[b]
                # Only consider vertices > b to avoid duplicates
                common &= ~((1 << (b + 1)) - 1)
                
                # Iterate over pairs in common
                c_list = []
                c = common
                while c:
                    lsb = c & -c
                    idx = lsb.bit_length() - 1
                    c_list.append(idx)
                    c ^= lsb
                
                for i in range(len(c_list)):
                    c = c_list[i]
                    for j in range(i + 1, len(c_list)):
                        d = c_list[j]
                        if (adj[c] >> d) & 1:
                            return (a, b, c, d)
        return None

    def find_ind6(n, adj):
        """Find an independent set of size 6. Returns tuple of 6 vertices or None."""
        # Independent set in G is a Clique in complement G_bar.
        # We use a recursive backtracking search on G_bar.
        # To speed up, we order vertices by degree in G_bar (descending) 
        # which is (n-1-degree in G).
        
        # Precompute complement adjacency
        comp_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = all_mask & ~(adj[i] | (1 << i))
        
        # Sort vertices by degree in comp_adj (highest first) to prune early?
        # Actually, for finding a clique, high degree is good.
        # Let's just use the natural order but with a strong pruning:
        # If remaining candidates + current size < 6, return False.
        
        # To make it faster, we can use a greedy heuristic first to find a large independent set,
        # but we specifically need size 6.
        
        found = [None]
        
        def search(candidates, current):
            if found[0] is not None:
                return
            if len(current) == 6:
                found[0] = tuple(current)
                return
            
            # Pruning: if we can't reach 6, return
            if len(current) + len(candidates) < 6:
                return
            
            # Heuristic: pick the vertex in candidates with highest degree in the induced subgraph?
            # Or just iterate.
            # To speed up, we can pick a pivot.
            # If candidates is empty, return.
            
            # Try to pick a vertex v from candidates
            # We iterate through candidates
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                c ^= lsb
                
                # Add v to current
                current.append(v)
                
                # New candidates must be neighbors of v in comp_adj AND in candidates
                # AND must have index > v (if we enforce ordering) to avoid duplicates
                # But since we are removing from candidates, we just intersect.
                # To avoid finding the same set in different orders, we can enforce
                # that new candidates have index > v.
                
                # Let's enforce index > v to keep it simple and avoid permutations.
                new_cands = candidates & comp_adj[v] & ~((1 << (v + 1)) - 1)
                
                search(new_cands, current)
                
                current.pop()
                
                if found[0] is not None:
                    return

        # Initial candidates: all vertices
        search((1 << n) - 1, [])
        return found[0]

    def can_add_edge(n, adj, u, v):
        """Check if adding edge (u,v) creates a K4."""
        if (adj[u] >> v) & 1:
            return False # Already an edge
        
        # Adding (u,v) creates a K4 if there is an edge (x,y) in N(u) intersect N(v)
        common = adj[u] & adj[v]
        # Check if any two vertices in 'common' are connected
        c = common
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c ^= lsb
            # Check if x is connected to any other vertex in common
            if (adj[x] & c):
                return False
        return True

    def repair_graph(n, adj, time_limit):
        """Iteratively repair the graph to be K4-free and Ind6-free."""
        while time.time() < time_limit:
            # 1. Fix K4s
            k4 = find_k4(n, adj)
            if k4:
                a, b, c, d = k4
                # Remove one random edge from the K4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = random.choice(edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                continue

            # 2. Fix Ind6s
            ind6 = find_ind6(n, adj)
            if ind6:
                # We have an independent set of 6.
                # We need to add an edge within this set to break it.
                # Try all pairs in the set to find a safe one to add.
                verts = list(ind6)
                added = False
                # Shuffle to randomize choice
                random.shuffle(verts)
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        u, v = verts[i], verts[j]
                        if can_add_edge(n, adj, u, v):
                            # Add the edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added = True
                            break
                    if added:
                        break
                
                if not added:
                    # We couldn't add any edge without creating a K4.
                    # This means the graph is "stuck" in a local minimum where
                    # every potential edge in the Ind6 creates a K4.
                    # Strategy: Randomly remove an edge elsewhere to "loosen" constraints.
                    # Pick a random existing edge and remove it.
                    removed = False
                    for _ in range(10): # Try a few times
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u != v and (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            removed = True
                            break
                    if not removed:
                        # Graph is empty? Unlikely.
                        break
                continue

            # 3. If no K4 and no Ind6, we are valid!
            return adj

        return adj

    def is_valid(n, adj):
        if find_k4(n, adj) is not None:
            return False
        if find_ind6(n, adj) is not None:
            return False
        return True

    best_n = 17
    best_adj = None

    # Seed: Paley graph on 17 (Known K4-free, alpha=4)
    # This provides a valid baseline.
    q = 17
    squares = {x * x % q for x in range(1, q)}
    adj17 = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                adj17[i] |= (1 << j)
                adj17[j] |= (1 << i)
    best_n = q
    best_adj = adj17

    # Try to find larger graphs
    # We prioritize higher n.
    # We use a time slice for each n.
    targets = [35, 34, 33, 32, 31, 30, 28, 25, 22, 20]
    
    # Time allocation:
    # Give more time to higher n.
    # Total time ~ 110s.
    # 35: 30s, 34: 20s, 33: 15s, 32: 10s, 31: 10s, 30: 5s, ...
    
    time_slices = {
        35: 30,
        34: 20,
        33: 15,
        32: 10,
        31: 10,
        30: 8,
        28: 5,
        25: 3,
        22: 2,
        20: 2
    }

    for target_n in targets:
        if time.time() > deadline - 5:
            break
        
        slice_time = time_slices.get(target_n, 5)
        slice_end = time.time() + slice_time
        
        # Multiple restarts within the slice
        while time.time() < slice_end:
            # Initialize random graph
            # Density 0.5 is a good start.
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Run repair
            result_adj = repair_graph(target_n, adj, slice_end)
            
            # Verify
            if is_valid(target_n, result_adj):
                best_n = target_n
                best_adj = result_adj
                break # Found valid graph for this n, move to next (smaller) n? 
                      # No, we want MAX n. So if we found 35, we are done?
                      # Yes, if we found a valid graph for 35, that's the max possible (since 36 is open).
                      # So we can stop completely.
                return {'n': best_n, 'edges': bitmask_to_edges(best_n, best_adj)}
            
            # If not valid, try again with new random seed
            # Small pause to avoid infinite loop if repair is stuck?
            # The repair loop has its own time check.

    return {'n': best_n, 'edges': bitmask_to_edges(best_n, best_adj)}
