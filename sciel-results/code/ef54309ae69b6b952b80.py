import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 16
    # Adjacency matrix as list of bitmasks
    adj = [0] * n
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def has_q3_with_edge(u, v, adj):
        # Check if adding edge (u, v) creates a Q3.
        # A Q3 containing (u, v) must have u and v connected.
        # In Q3, u and v are adjacent.
        # The other 6 vertices form the rest of the cube.
        # Let's enumerate the possible roles.
        # u and v are adjacent in Q3.
        # Without loss of generality, let u=0, v=1 in the Q3 template.
        # Neighbors of 0 in Q3: 1, 2, 3.
        # Neighbors of 1 in Q3: 0, 4, 5.
        # So in the graph, u must be connected to 2 other vertices (let's call them A, B) which are the images of 2,3.
        # v must be connected to 2 other vertices (let's call them C, D) which are the images of 4,5.
        # The remaining 2 vertices (E, F) are images of 6,7.
        
        # Candidates for A, B: Neighbors of u (excluding v)
        # Candidates for C, D: Neighbors of v (excluding u)
        
        # Note: A, B, C, D, E, F must be distinct and not u, v.
        
        # Get neighbors of u
        # adj[u] is a bitmask.
        # Mask out u and v
        mask_uv = (1 << u) | (1 << v)
        neu = adj[u] & ~mask_uv
        nev = adj[v] & ~mask_uv
        
        # List of neighbor indices
        neu_list = [i for i in range(n) if (neu >> i) & 1]
        nev_list = [i for i in range(n) if (nev >> i) & 1]
        
        if len(neu_list) < 2 or len(nev_list) < 2:
            return False
            
        # We need to pick 2 from neu_list for {2,3} and 2 from nev_list for {4,5}.
        # They must be disjoint.
        
        for a, b in itertools.combinations(neu_list, 2):
            # a, b are candidates for 2, 3
            # Remaining candidates for 4, 5 must be from nev_list \ {a, b}
            # Actually, a and b are from neu, c and d from nev.
            # In Q3, {2,3} and {4,5} are disjoint.
            # So we pick c, d from nev_list such that c, d not in {a, b}.
            
            # Also, edges (2,4) and (3,5) or (2,5) and (3,4) must exist?
            # Q3 edges:
            # 0-1 (u-v)
            # 0-2 (u-a), 0-3 (u-b)
            # 1-4 (v-c), 1-5 (v-d)
            # 2-4 (a-c), 2-6 (a-e)
            # 3-5 (b-d), 3-6 (b-e)
            # 4-7 (c-f), 5-7 (d-f)
            # 6-7 (e-f)
            
            # Note: There is no edge between 2 and 3, nor 4 and 5, nor 2 and 5 (wait, 2-5 is not an edge in Q3? 
            # Q3 edges: 
            # 0: 1,2,3
            # 1: 0,4,5
            # 2: 0,4,6
            # 3: 0,5,6
            # 4: 1,2,7
            # 5: 1,3,7
            # 6: 2,3,7
            # 7: 4,5,6
            # So 2 is connected to 4 and 6. 3 is connected to 5 and 6.
            # 2 is NOT connected to 5. 3 is NOT connected to 4.
            
            # So we need:
            # adj[a][c] == 1
            # adj[b][d] == 1
            # OR
            # adj[a][d] == 1
            # adj[b][c] == 1
            
            # Let's check both cases.
            
            # Case 1: a-c and b-d
            # Case 2: a-d and b-c
            
            # For each valid pairing, we need to find e, f such that:
            # e is connected to a, b
            # f is connected to c, d
            # e is connected to f
            # e, f distinct, not in {u,v,a,b,c,d}
            
            used = {u, v, a, b}
            
            # Iterate c, d from nev_list
            for c, d in itertools.combinations(nev_list, 2):
                if c in used or d in used:
                    continue
                
                used_cd = used | {c, d}
                
                # Check connections a-c, b-d or a-d, b-c
                valid_pairing = False
                # We need to determine which c is "4" and which is "5".
                # If we assume c=4, d=5, then we need a-c and b-d.
                # If we assume c=5, d=4, then we need a-d and b-c.
                
                # Since we are iterating combinations (unordered c,d), we check both.
                
                # Option 1: c is 4, d is 5
                if (adj[a] & (1 << c)) and (adj[b] & (1 << d)):
                    valid_pairing = True
                    c_role = 4
                    d_role = 5
                    # e connected to a(2), b(3)
                    # f connected to c(4), d(5)
                elif (adj[a] & (1 << d)) and (adj[b] & (1 << c)):
                    valid_pairing = True
                    c_role = 5
                    d_role = 4
                    # e connected to a(2), b(3)
                    # f connected to d(4), c(5) -> f connected to c, d anyway.
                else:
                    continue
                
                if not valid_pairing:
                    continue
                
                # Find e, f
                # e in V \ used_cd
                # f in V \ used_cd
                # e != f
                # adj[e][a] and adj[e][b]
                # adj[f][c] and adj[f][d]
                # adj[e][f]
                
                # Precompute candidates for e and f
                # e must be neighbor of a and b
                # f must be neighbor of c and d
                
                # Iterate e
                for e in range(n):
                    if e in used_cd:
                        continue
                    if not ((adj[e] & (1 << a)) and (adj[e] & (1 << b))):
                        continue
                    
                    # Iterate f
                    for f in range(n):
                        if f in used_cd or f == e:
                            continue
                        if not ((adj[f] & (1 << c)) and (adj[f] & (1 << d))):
                            continue
                        if (adj[e] & (1 << f)):
                            return True
                            
        return False

    # Initialize with greedy
    # Start with empty graph? Or dense?
    # Let's start with the greedy result from the seed to have a good start.
    # But implementing the seed's greedy is redundant if we just run SA from a random dense graph.
    # However, a good start helps.
    
    # Let's try to build a dense graph quickly.
    # Random initialization
    random.seed(42)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Remove any Q3s initially
    # Simple way: check all edges, if removing one breaks a Q3, keep it?
    # Actually, just check if current graph has Q3. If yes, remove random edges until valid.
    def has_any_q3(adj):
        # Check all pairs of edges? No, check all 8-subsets is too slow.
        # Use the fact that we can check incrementally?
        # For initialization, we can just try to remove edges.
        # Or better: Start with K16 and remove edges?
        # Let's stick to the "Add/Kick" strategy.
        
        # To ensure we start valid, let's run a quick greedy add.
        # Or just assume we start with a graph that might have Q3s and fix it.
        pass

    # Let's restart with a proper initialization:
    # Start with K16 and remove edges until Q3-free?
    # Or start empty and add?
    # The seed used K3,13 + greedy.
    # Let's use a random dense graph and "fix" it by removing edges involved in Q3s.
    
    # To find if a graph has Q3, we can iterate over all 8-subsets.
    # 12870 subsets. For each, check isomorphism.
    # Isomorphism check for 8 vertices is fast.
    
    def is_q3_subset(vertices, adj):
        # vertices is a tuple of 8 indices
        # Map to 0..7
        # Check all 12 edges of Q3
        # Q3 edges:
        # (0,1), (0,2), (0,3), (1,4), (1,5), (2,4), (2,6), (3,5), (3,6), (4,7), (5,7), (6,7)
        # We need to check if there is a permutation of vertices that satisfies these.
        # Since 8! = 40320, and we have 12870 subsets, total ~500M checks. Too slow in Python.
        
        # Better: Use the structure.
        # Pick a vertex in the subset, say v0.
        # Its neighbors in the subset must be exactly 3.
        # If any vertex has degree != 3 in the induced subgraph, return False.
        
        # Induced subgraph degrees
        degs = [0]*8
        for i in range(8):
            for j in range(8):
                if i == j: continue
                if adj[vertices[i]] & (1 << vertices[j]):
                    degs[i] += 1
        if any(d != 3 for d in degs):
            return False
            
        # Now check bipartiteness? Q3 is bipartite.
        # And check specific structure.
        # Actually, if a 3-regular bipartite graph on 8 vertices is not Q3, it could be K4,4? No, K4,4 has degree 4.
        # The only 3-regular bipartite graph on 8 vertices is Q3?
        # Let's check.
        # 3-regular graphs on 8 vertices:
        # Q3 is one.
        # Are there others?
        # The complement of Q3 is 4-regular.
        # Yes, Q3 is the unique 3-regular bipartite graph on 8 vertices?
        # Wait, the cube Q3 is bipartite.
        # Any 3-regular bipartite graph on 8 vertices must be Q3?
        # Let's assume yes for now, but verify.
        # If it's 3-regular and bipartite, it's a candidate.
        # We should verify it's isomorphic to Q3.
        # Since it's small, we can just check if it contains a 4-cycle? Q3 has many.
        # Actually, the property "3-regular bipartite on 8 vertices" might be sufficient to identify Q3 in this context?
        # Let's double check.
        # There are 2 non-isomorphic 3-regular graphs on 8 vertices?
        # No, Q3 is the only 3-regular bipartite graph on 8 vertices?
        # Actually, there is another one: The "Möbius ladder" M4? No, M4 is 3-regular on 8 vertices but not bipartite (has odd cycles).
        # So if it's bipartite and 3-regular, it is Q3.
        
        # Check bipartiteness
        color = [-1]*8
        color[0] = 0
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in range(8):
                if adj[vertices[u]] & (1 << vertices[v]):
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
        return True

    def has_any_q3_fast(adj):
        for subset in itertools.combinations(range(n), 8):
            if is_q3_subset(subset, adj):
                return True
        return False

    # Initialization: Start with K16 and remove edges?
    # K16 has many Q3s.
    # Let's start with the greedy result from the seed logic to be safe.
    # But we can just start with a random graph and remove edges until valid.
    
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.9:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Remove edges to make it Q3-free
    # This is a heuristic initialization.
    # While has_any_q3_fast(adj):
    #    Find a Q3, remove a random edge from it.
    # This is slow.
    
    # Better: Use the greedy add strategy from seed but with bitmasks.
    # Start empty.
    adj = [0] * n
    edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if time.time() > deadline: break
        # Check if adding (u,v) creates Q3
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if has_q3_with_edge(u, v, adj):
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            # Keep
            pass
            
    # Now we have a valid graph.
    # Run Local Search
    best_edges = sum(bin(adj[i]).count('1') for i in range(n)) // 2
    current_edges = best_edges
    best_adj = adj[:]
    
    while time.time() < deadline:
        # Try to add a random missing edge
        added = False
        for _ in range(100):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if adj[u] & (1 << v): continue
            
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(u, v, adj):
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Keep
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                added = True
                break
        
        if not added:
            # Local optimum. Kick: remove a random edge
            # Pick a random edge that exists
            edges_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges_list.append((i, j))
            if edges_list:
                u, v = random.choice(edges_list)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
            else:
                break # Empty graph

    # Return best found
    return get_edges(best_adj)
