import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n = 16
    
    # Precompute all pairs (edges)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute neighbors for potential Q3 checks
    # A Q3 is an 8-vertex subgraph isomorphic to the 3-cube.
    # The 3-cube has vertices {000, 001, 010, 011, 100, 101, 110, 111}.
    # Edges connect vertices differing in exactly one bit.
    
    # To check if a set of 8 vertices S contains a Q3, we need to check if there is an isomorphism.
    # However, for local search, we check if adding edge (u,v) creates a Q3.
    # This happens if there exists a set of 6 other vertices such that the induced subgraph on these 8 is Q3.
    
    # Efficient Q3 detection for a specific graph:
    # Iterate over all 8-vertex subsets? No, too slow (C(16,8) = 12870).
    # Better: Iterate over all possible "centers" or specific structures.
    
    # Alternative approach: 
    # Start with a dense graph and remove edges that are part of a Q3.
    # Or start sparse and add edges.
    
    # Let's try a hybrid:
    # 1. Initialize with a random graph with high density (e.g., p=0.6).
    # 2. Identify all Q3s.
    # 3. Solve a hitting set / set cover problem to remove minimum edges to break all Q3s.
    #    This is NP-hard, but for 16 vertices, we can use a greedy approximation or local search.
    
    # Actually, let's try a different structural insight:
    # The 3-cube Q3 is bipartite. Its vertices can be colored black/white based on parity of bits.
    # 4 black, 4 white.
    # Any Q3 in the graph must have 4 vertices in one part and 4 in the other, with specific connections.
    
    # Let's use a bitmask representation for the graph.
    # adj[i] is a 16-bit integer where bit j is 1 if (i,j) is an edge.
    
    def check_q3_in_graph(adj):
        """
        Returns True if the graph represented by adj (list of 16 bitmasks) contains a Q3.
        A Q3 can be identified by its 8 vertices.
        We can enumerate all 8-vertex subsets and check if they form a Q3.
        But C(16,8) is 12870. For each, checking isomorphism is expensive.
        
        Better: Enumerate all possible embeddings of Q3?
        There are 16! / (8! * 8!) * 8! ways... no.
        
        Let's stick to the "find a specific Q3 structure" method.
        A Q3 consists of two disjoint 4-cycles connected by 4 edges? No.
        A Q3 has 8 vertices and 12 edges.
        
        Fast check:
        For each vertex v, look at its neighbors.
        If v is in a Q3, it has 3 neighbors in the Q3.
        Let these be a, b, c.
        a, b, c must be pairwise non-adjacent.
        Then there must be vertices x, y, z, w such that:
        x ~ a, b
        y ~ a, c
        z ~ b, c
        w ~ x, y, z
        And x,y,z are pairwise non-adjacent.
        And w is not adjacent to a,b,c.
        
        This is the same logic as the previous program, but we can make it faster with bitmasks.
        """
        for v in range(n):
            # Get neighbors of v
            nb_v = adj[v]
            # If degree < 3, skip
            if nb_v.bit_count() < 3:
                continue
            
            # Get list of neighbors
            neigh_list = []
            for i in range(n):
                if nb_v & (1 << i):
                    neigh_list.append(i)
            
            # Iterate over triples of neighbors
            for i in range(len(neigh_list)):
                for j in range(i+1, len(neigh_list)):
                    for k in range(j+1, len(neigh_list)):
                        a = neigh_list[i]
                        b = neigh_list[j]
                        c = neigh_list[k]
                        
                        # a, b, c must be pairwise non-adjacent
                        if adj[a] & (1 << b): continue
                        if adj[a] & (1 << c): continue
                        if adj[b] & (1 << c): continue
                        
                        # Find x: neighbor of a and b
                        # x != v, a, b, c
                        x_candidates = (adj[a] & adj[b]) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                        
                        # Find y: neighbor of a and c
                        y_candidates = (adj[a] & adj[c]) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                        
                        # Find z: neighbor of b and c
                        z_candidates = (adj[b] & adj[c]) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                        
                        # Iterate through candidates
                        # To speed up, we can just pick one candidate for x, y, z?
                        # No, we need to check if ANY combination works.
                        
                        # Let's get lists
                        xs = [x for x in range(n) if x_candidates & (1 << x)]
                        ys = [y for y in range(n) if y_candidates & (1 << y)]
                        zs = [z for z in range(n) if z_candidates & (1 << z)]
                        
                        for x in xs:
                            for y in ys:
                                for z in zs:
                                    if x == y or y == z or x == z:
                                        continue
                                    
                                    # x, y, z must be pairwise non-adjacent
                                    if adj[x] & (1 << y): continue
                                    if adj[x] & (1 << z): continue
                                    if adj[y] & (1 << z): continue
                                    
                                    # Find w: neighbor of x, y, z
                                    # w != v, a, b, c, x, y, z
                                    w_candidates = (adj[x] & adj[y] & adj[z])
                                    mask_forbidden = (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << x) | (1 << y) | (1 << z)
                                    w_candidates &= ~mask_forbidden
                                    
                                    # w must NOT be adjacent to a, b, c
                                    # Check for any w in w_candidates
                                    for w in range(n):
                                        if w_candidates & (1 << w):
                                            # Check w not adjacent to a, b, c
                                            if adj[w] & (1 << a): continue
                                            if adj[w] & (1 << b): continue
                                            if adj[w] & (1 << c): continue
                                            
                                            # Found a Q3
                                            return True
        return False

    # Strategy:
    # 1. Start with a complete graph.
    # 2. While Q3 exists, find one and remove an edge from it.
    #    To maximize edges, we should remove edges that are in MANY Q3s (high centrality).
    #    This is a greedy set cover.
    
    def find_q3_and_edges(adj):
        """
        Returns a list of edges in the first found Q3, or None.
        """
        for v in range(n):
            nb_v = adj[v]
            if nb_v.bit_count() < 3:
                continue
            neigh_list = [i for i in range(n) if nb_v & (1 << i)]
            for i in range(len(neigh_list)):
                for j in range(i+1, len(neigh_list)):
                    for k in range(j+1, len(neigh_list)):
                        a = neigh_list[i]
                        b = neigh_list[j]
                        c = neigh_list[k]
                        if adj[a] & (1 << b): continue
                        if adj[a] & (1 << c): continue
                        if adj[b] & (1 << c): continue
                        
                        x_candidates = (adj[a] & adj[b]) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                        y_candidates = (adj[a] & adj[c]) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                        z_candidates = (adj[b] & adj[c]) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                        
                        xs = [x for x in range(n) if x_candidates & (1 << x)]
                        ys = [y for y in range(n) if y_candidates & (1 << y)]
                        zs = [z for z in range(n) if z_candidates & (1 << z)]
                        
                        for x in xs:
                            for y in ys:
                                for z in zs:
                                    if x == y or y == z or x == z: continue
                                    if adj[x] & (1 << y): continue
                                    if adj[x] & (1 << z): continue
                                    if adj[y] & (1 << z): continue
                                    
                                    w_candidates = (adj[x] & adj[y] & adj[z])
                                    mask_forbidden = (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << x) | (1 << y) | (1 << z)
                                    w_candidates &= ~mask_forbidden
                                    
                                    for w in range(n):
                                        if w_candidates & (1 << w):
                                            if adj[w] & (1 << a): continue
                                            if adj[w] & (1 << b): continue
                                            if adj[w] & (1 << c): continue
                                            
                                            # Found Q3 with vertices {v, a, b, c, x, y, z, w}
                                            # Edges:
                                            # (v,a), (v,b), (v,c)
                                            # (a,x), (b,x)
                                            # (a,y), (c,y)
                                            # (b,z), (c,z)
                                            # (x,w), (y,w), (z,w)
                                            
                                            edges = [
                                                (v, a), (v, b), (v, c),
                                                (a, x), (b, x),
                                                (a, y), (c, y),
                                                (b, z), (c, z),
                                                (x, w), (y, w), (z, w)
                                            ]
                                            # Normalize
                                            normalized = []
                                            for u, v_e in edges:
                                                if u > v_e:
                                                    u, v_e = v_e, u
                                                normalized.append((u, v_e))
                                            return normalized
        return None

    # Greedy removal
    def greedy_remove_q3s(adj):
        # Copy adj
        adj = adj[:]
        
        while True:
            if time.time() > deadline - 5:
                break
            q3_edges = find_q3_and_edges(adj)
            if q3_edges is None:
                break
            
            # Count frequency of each edge in this Q3? No, just pick one.
            # To be smarter, we could count how many Q3s each edge is in, but that's expensive.
            # Let's just pick the first one or a random one.
            # Actually, let's pick the edge that appears in the most Q3s?
            # For now, random choice from the 12 edges.
            e_to_remove = random.choice(q3_edges)
            u, v = e_to_remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
    # Local Search: Add edges back
    def local_search(adj):
        # Try to add all missing edges
        missing_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    missing_edges.append((i, j))
        
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline - 5:
                break
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if Q3 created
            if check_q3_in_graph(adj):
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            # Else keep it
            
    best_edges = []
    best_count = 0
    
    # Multiple restarts
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        
        # Initialize with complete graph
        adj = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i] |= (1 << j)
        
        # Remove Q3s
        greedy_remove_q3s(adj)
        
        # Count edges
        edge_count = 0
        for i in range(n):
            edge_count += adj[i].bit_count()
        edge_count //= 2
        
        if edge_count > best_count:
            best_count = edge_count
            # Convert to list of edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            best_edges = edges
            
        # Local search
        local_search(adj)
        
        # Count edges again
        edge_count = 0
        for i in range(n):
            edge_count += adj[i].bit_count()
        edge_count //= 2
        
        if edge_count > best_count:
            best_count = edge_count
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            best_edges = edges

    return best_edges

# EVOLVE-BLOCK-END
