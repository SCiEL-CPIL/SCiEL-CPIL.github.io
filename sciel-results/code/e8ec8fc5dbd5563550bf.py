import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Construct the Schläfli graph (35 vertices) as complement of GQ(2,4) point graph
    # GQ(2,4) has 43 points = PG(3,2) \ hyperoval (21 - 8 = 13? No, 21 points in PG(3,2))
    # Actually PG(3,2) has 15 points, not 21. Let me reconsider.
    # PG(3,2): points = (2^4 - 1)/(2-1) = 15. That's not 43 either.
    #
    # Let me use a different approach: the Schläfli graph is a strongly regular graph
    # with parameters (35, 16, 7, 8). I'll construct it using the known fact that
    # it's the complement of the point graph of GQ(2,4).
    #
    # GQ(2,4) has (s+1)(st+1) = 3*9 = 27 points? No, (s+1)(st+1) = 3*(2*4+1) = 27.
    # That's not 35 either.
    #
    # Let me try yet another construction. The Schläfli graph can be constructed
    # from the 35 edges of K7. Two edges are adjacent if they share a vertex.
    # Wait, that gives the line graph of K7, which has 35 vertices.
    # The line graph of K7: vertices = edges of K7, two adjacent if edges share a vertex.
    # Line graph of K7 has degree 2*(7-2) = 10. But Schläfli has degree 16.
    #
    # Actually, the complement of the line graph of K7 might work.
    # In L(K7), each vertex (edge of K7) is adjacent to 2*(7-2) = 10 others.
    # So in complement, each vertex has degree 35-1-10 = 24. Still not 16.
    #
    # Let me use the correct construction: The Schläfli graph is the graph on the
    # 35 points of PG(3,4)? No, PG(3,4) has (4^4-1)/(4-1) = 273 points.
    #
    # I'll use the explicit construction from the 35 vertices as pairs {i,j} from {0,...,6}
    # with a specific adjacency rule. Actually, let me just hardcode the adjacency
    # from a verified source using the known SRG parameters.
    #
    # The Schläfli graph: 35 vertices, each vertex connected to 16 others.
    # It's the complement of the Clebsch graph? No, Clebsch has 16 vertices.
    #
    # Let me use the construction: vertices are the 35 lines of the projective plane
    # of order 4? PG(2,4) has (4^3-1)/(4-1) = 21 lines. Not 35.
    #
    # OK, I'll use the fact that the Schläfli graph is isomorphic to the graph
    # where vertices are the 35 2-element subsets of {1,...,7}, and two subsets
    # are adjacent if their symmetric difference has size 2 (i.e., they share exactly one element)
    # OR their union has size 3 (share one element) ... 
    #
    # Actually, let me just use a verified adjacency list. The Schläfli graph is unique.
    # I'll generate it using the following known property:
    # It is the complement of the point graph of the generalized quadrangle GQ(2,4).
    # GQ(2,4) has (s+1)(st+1) = 3*9 = 27 points. That's wrong.
    #
    # GQ(s,t) has (s+1)(st+1) points. For GQ(2,4): 3*(2*4+1) = 27. Not 35.
    #
    # I think I'm confusing constructions. Let me just hardcode the Schläfli graph
    # using the adjacency from the House of Graphs or similar verified source.
    #
    # The Schläfli graph adjacency (0-indexed, 35 vertices):
    # I'll use the known fact that it's a (35,16,7,8) SRG.
    # Let me construct it as follows: partition 35 vertices into 7 groups of 5.
    # Within each group: no edges (independent set of 5).
    # Between groups: each vertex connects to exactly 2 vertices in each other group.
    # This gives degree 6*2 = 12. Not 16.
    #
    # Let me try: 7 groups of 5. Each vertex connects to 3 in each of 5 other groups.
    # Degree = 5*3 = 15. Close but not 16.
    #
    # I'll use the explicit edge list from a verified source.
    # The Schläfli graph has 35*16/2 = 280 edges.
    
    # Verified construction: The Schläfli graph is the complement of the graph
    # where vertices are the 35 points of the projective geometry PG(3,2) minus
    # an oval? PG(3,2) has 15 points. 15 is not 35.
    
    # Let me just use the Paley graph on 17 as a safe fallback and try to
    # construct a larger graph using targeted repair.
    
    # Actually, I recall that the Schläfli graph can be constructed as follows:
    # Take the 35 vertices as the edges of K7. Two vertices (edges of K7) are
    # adjacent in the Schläfli graph if the corresponding edges in K7 are
    # NOT adjacent (i.e., they don't share a vertex).
    # In K7, each edge is disjoint from (7-2)*2 - 1 = 9 other edges? 
    # No: edge {a,b} is disjoint from edges {c,d} where c,d are from {0..6}\{a,b}.
    # Number of such edges: C(5,2) = 10.
    # So in this construction, each vertex has degree 10. Not 16.
    
    # The complement would have degree 35-1-10 = 24. Not 16.
    
    # I'll use the following verified construction from the literature:
    # The Schläfli graph is the point graph of the dual of GQ(2,4) or similar.
    
    # Given the difficulty of constructing it from scratch, I'll use the
    # targeted repair approach with a good initial guess.
    
    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_k4(n, adj):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        count += (adj[w] & c).bit_count()
        return count // 6

    def has_ind6(n, adj):
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask & ~(adj[i] | (1 << i))
        
        for i in range(n):
            if comp_adj[i].bit_count() < 5:
                return False

        def find_clique(candidates, size):
            if size == 0:
                return True
            if candidates.bit_count() < size:
                return False
            v = (candidates & -candidates).bit_length() - 1
            new_cands = candidates & ~(1 << v) & comp_adj[v]
            if find_clique(new_cands, size - 1):
                return True
            return find_clique(candidates & ~(1 << v), size)

        return find_clique((1 << n) - 1, 6)

    def is_valid(n, adj):
        if count_k4(n, adj) > 0:
            return False
        if has_ind6(n, adj):
            return False
        return True

    def targeted_repair(n, adj, time_limit):
        """Fix K4s and independent 6-sets by removing/adding edges."""
        current_adj = adj[:]
        
        def get_penalty(adj):
            k4 = count_k4(n, adj)
            ind6 = has_ind6(n, adj)
            return k4 * 10 + (100 if ind6 else 0)

        current_penalty = get_penalty(current_adj)
        if current_penalty == 0:
            return current_adj

        import random
        while time.time() < time_limit:
            if current_penalty == 0:
                return current_adj
            
            # Find a K4 or ind6 and try to fix it
            if count_k4(n, current_adj) > 0:
                # Find a K4 and remove an edge
                found = False
                for u in range(n):
                    if found:
                        break
                    for v in range(u + 1, n):
                        if found or not ((current_adj[u] >> v) & 1):
                            continue
                        common = current_adj[u] & current_adj[v]
                        c = common
                        while c:
                            lsb = c & -c
                            w = lsb.bit_length() - 1
                            c ^= lsb
                            if (current_adj[w] & (c & (1 << (w+1)))).bit_count() > 0:
                                # Found K4: u,v,w,x where x is in c & adj[w]
                                c2 = c & current_adj[w]
                                if c2:
                                    lsb2 = c2 & -c2
                                    x = lsb2.bit_length() - 1
                                    if x > w:
                                        # Remove edge (u,v) to break K4
                                        current_adj[u] &= ~(1 << v)
                                        current_adj[v] &= ~(1 << u)
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    # Random flip
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        if (current_adj[u] >> v) & 1:
                            current_adj[u] &= ~(1 << v)
                            current_adj[v] &= ~(1 << u)
                        else:
                            current_adj[u] |= (1 << v)
                            current_adj[v] |= (1 << u)
            else:
                # Fix ind6: add edges within the independent set
                # Find an independent 6-set and add a random edge within it
                mask = (1 << n) - 1
                comp_adj = [0] * n
                for i in range(n):
                    comp_adj[i] = mask & ~(current_adj[i] | (1 << i))
                
                def find_ind6_set(candidates, size, result):
                    if size == 0:
                        return True
                    if candidates.bit_count() < size:
                        return False
                    v = (candidates & -candidates).bit_length() - 1
                    result.append(v)
                    if find_ind6_set(candidates & ~(1 << v) & comp_adj[v], size - 1, result):
                        return True
                    result.pop()
                    return find_ind6_set(candidates & ~(1 << v), size, result)
                
                ind_set = []
                if find_ind6_set((1 << n) - 1, 6, ind_set):
                    # Add a random edge within the independent set
                    i1, i2 = random.sample(ind_set, 2)
                    current_adj[i1] |= (1 << i2)
                    current_adj[i2] |= (1 << i1)
                else:
                    # Random flip
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        if (current_adj[u] >> v) & 1:
                            current_adj[u] &= ~(1 << v)
                            current_adj[v] &= ~(1 << u)
                        else:
                            current_adj[u] |= (1 << v)
                            current_adj[v] |= (1 << u)
            
            current_penalty = get_penalty(current_adj)

        return current_adj

    # Best fallback: Paley graph on 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj17 = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj17[i] |= (1 << j)
                adj17[j] |= (1 << i)
    
    best_n = 17
    best_adj = adj17

    # Try to construct larger graphs using targeted repair
    import random
    targets = [35, 34, 33, 32, 31, 30, 28, 25, 22, 20]
    
    for target_n in targets:
        if time.time() > deadline - 15:
            break
        
        # Initialize with a random graph biased toward the right density
        # For n=35, we want ~280 edges (density ~0.457)
        target_density = 16 / (target_n - 1) if target_n > 1 else 0.5
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < target_density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        slice_end = time.time() + 12
        if slice_end > deadline - 5:
            slice_end = deadline - 5
            
        result_adj = targeted_repair(target_n, adj, slice_end)
        
        if is_valid(target_n, result_adj):
            best_n = target_n
            best_adj = result_adj
            break

    return {'n': best_n, 'edges': bitmask_to_edges(best_n, best_adj)}
