import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    n = 16
    
    # Helper: Check if graph is 3-colorable
    # adj[i] is a bitmask of neighbors of i
    def is_3_colorable(adj):
        # Backtracking 3-coloring
        # colors[i] is 0, 1, or 2
        colors = [-1] * n
        
        def backtrack(i):
            if i == n:
                return True
            
            # Determine available colors
            # A color is available if no neighbor has it
            # We can optimize by checking neighbors first
            
            # Get neighbors of i
            neigh_mask = adj[i]
            
            # Check colors of neighbors
            c0 = False
            c1 = False
            c2 = False
            
            # Iterate through set bits in neigh_mask
            m = neigh_mask
            while m:
                # Get least significant bit
                lsb = m & -m
                j = lsb.bit_length() - 1
                c = colors[j]
                if c == 0: c0 = True
                elif c == 1: c1 = True
                elif c == 2: c2 = True
                m ^= lsb
            
            # Try colors
            if not c0:
                colors[i] = 0
                if backtrack(i + 1):
                    return True
            if not c1:
                colors[i] = 1
                if backtrack(i + 1):
                    return True
            if not c2:
                colors[i] = 2
                if backtrack(i + 1):
                    return True
            
            colors[i] = -1
            return False
        
        # Order vertices by degree descending for better pruning
        # This requires re-indexing or just using the order 0..15 which is fine for n=16
        # But let's use a simple heuristic: try to color high degree vertices first?
        # For n=16, fixed order 0..15 is likely fast enough if we use bitsets.
        
        return backtrack(0)

    def create_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            m = adj[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m ^= lsb
        return edges

    # Initialize with Clebsch Graph
    # Clebsch graph vertices 0..15
    # It is the folded 5-cube.
    # Or we can define it by its adjacency list.
    # Clebsch graph is strongly regular (16, 5, 0, 2).
    # Let's construct it properly.
    # Vertices are 4-bit strings. Two vertices are adjacent if their Hamming distance is 2 or 4?
    # No, distance 2 or 4?
    # Standard Clebsch: distance 2.
    # Let's check: n=16, k=5.
    # If dist=2: number of vectors at dist 2 from 0000 is C(4,2) = 6. Too many.
    # If dist=4: number is 1.
    # If dist=2 and dist=4? 6+1=7. No.
    # The Clebsch graph is the complement of the Shrikhande graph? No.
    # Let's use a known list or generate it via Mycielski?
    # Actually, let's just use a random high-density triangle-free graph generator
    # and rely on local search to find the structure.
    
    # Better: Start with a few known good seeds.
    # Seed 1: Grötzsch graph (11 vertices) + 5 isolated.
    # Seed 2: Random maximal triangle-free graph.
    
    def groetzsch_edges():
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]
    
    best_edges_count = 0
    best_adj = None
    
    # Function to greedily complete a graph to maximal triangle-free
    def complete_to_maximal(adj):
        # Try to add all possible edges that don't create triangles
        # Repeat until no changes
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        # Check if adding (u,v) creates a triangle
                        # Triangle exists if there is w such that w is neighbor of u and w is neighbor of v
                        common = adj[u] & adj[v]
                        if common == 0:
                            # Safe to add
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
        return adj

    # Local Search Step: Swap an edge to increase density or change structure
    def local_search(adj, steps):
        current_adj = adj[:]
        current_is_3col = is_3_colorable(current_adj)
        current_edges = sum(bin(x).count("1") for x in current_adj) // 2
        
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            edges = edges_from_adj(current_adj)
            if not edges:
                break
            e1 = random.choice(edges)
            u1, v1 = e1
            
            # Remove it
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            
            # Pick a random non-edge to add
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (current_adj[u] >> v) & 1:
                        non_edges.append((u, v))
            
            if not non_edges:
                # Restore e1 if no non-edges
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                break
                
            e2 = random.choice(non_edges)
            u2, v2 = e2
            
            # Check if adding e2 creates a triangle
            common = current_adj[u2] & current_adj[v2]
            if common == 0:
                # Add e2
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                # Check 3-colorability
                new_is_3col = is_3_colorable(current_adj)
                new_edges = sum(bin(x).count("1") for x in current_adj) // 2
                
                # We want to maximize edges, but we MUST be not 3-colorable.
                # If the new graph is not 3-colorable, we accept it if it has more edges
                # or if it has same edges but we are exploring.
                # Actually, we should keep the best non-3-colorable graph found so far.
                
                if not new_is_3col:
                    # This is a valid candidate
                    if new_edges > best_edges_count:
                        best_edges_count = new_edges
                        best_adj = current_adj[:]
                else:
                    # If it is 3-colorable, we might still keep it for further search
                    # if it has more edges than the previous state?
                    # Or just accept the move if it doesn't decrease edges too much?
                    # Let's accept if edges >= current_edges or randomly to escape local minima
                    if new_edges >= current_edges or random.random() < 0.1:
                        current_edges = new_edges
                        current_is_3col = new_is_3col
                    else:
                        # Reject move, restore e1
                        current_adj[u1] |= (1 << v1)
                        current_adj[v1] |= (1 << u1)
                        # e2 is already removed
                # If we rejected, we need to restore the state properly
                # The logic above is a bit mixed. Let's simplify:
                # We always perform the swap. We check validity.
                # If valid (triangle-free), we check 3-color.
                # We update best if valid and not 3-color.
                # We decide whether to keep the state for the next iteration.
                # Heuristic: Keep state if it is not 3-colorable (good) or if it has more edges.
                if not new_is_3col:
                    pass # Keep
                elif new_edges > current_edges:
                    pass # Keep
                else:
                    # Revert
                    current_adj[u2] &= ~(1 << v2)
                    current_adj[v2] &= ~(1 << u2)
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
            else:
                # Triangle formed, revert
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)

        return current_adj, best_edges_count, best_adj

    # Main Loop
    # Try multiple restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial graph
        if r == 0:
            # Start with Grötzsch
            edges = groetzsch_edges()
            adj = create_adj_from_edges(edges)
        elif r == 1:
            # Start with Clebsch-like structure (random dense triangle-free)
            adj = [0] * n
            # Add a random bipartite graph
            part1 = list(range(8))
            part2 = list(range(8, 16))
            for u in part1:
                for v in part2:
                    if random.random() < 0.7:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            # Random sparse start
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.2:
                        # Check triangle
                        if (adj[u] & adj[v]) == 0:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        
        # Complete to maximal
        adj = complete_to_maximal(adj)
        
        # Check best
        if not is_3_colorable(adj):
            cnt = sum(bin(x).count("1") for x in adj) // 2
            if cnt > best_edges_count:
                best_edges_count = cnt
                best_adj = adj[:]
        
        # Local Search
        adj, best_edges_count, best_adj = local_search(adj, steps=200)
        
        # If we found a very good one, maybe do more search around it?
        # The local search function already updates best_adj.
        
    if best_adj is None:
        # Fallback to Grötzsch
        return groetzsch_edges()
    else:
        return edges_from_adj(best_adj)

# EVOLVE-BLOCK-END
