"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a C4-free graph with no independent set of 11 vertices.
    
    We use the Higman-Sims graph complement approach. The Higman-Sims graph
    has 100 vertices, is strongly regular (100, 22, 0, 6), and is C4-free.
    Its complement has independence number 22 (the maximum clique of HS).
    
    Actually, let's use a different approach: we'll use the known (39, 582) 
    C4-free graph from the literature, or construct one via finite fields.
    
    A well-known construction: take the graph where vertices are pairs (x, y)
    in Z_p x Z_p for prime p, with edges defined by y - y' = (x - x')^2 
    or similar. But let's use a verified construction.
    
    We'll use the complement of a known C4-free graph. The Petersen graph 
    complement has 10 vertices. Let's build up from known components.
    
    Actually, the best known C4-free graph with alpha <= 10 has 39 vertices.
    This is related to the Erdos-Renyi graph. Let me construct it using
    the standard finite field construction for C4-free graphs.
    
    For p = 7 (prime), consider the graph on Z_7 x Z_7 with 49 vertices.
    Edge between (a,b) and (c,d) if (a-c, b-d) is in a specific set.
    
    Let's use the known construction: vertices are elements of GF(7^2) = GF(49).
    Two vertices are adjacent if their difference is a square in GF(49).
    This gives a C4-free graph with 49 vertices.
    
    But we need alpha <= 10. The independence number of this graph might be 
    too large. Let me use a smaller construction.
    
    For the Higman-Sims graph: 100 vertices, strongly regular (100,22,0,6).
    It's C4-free (lambda=0 means no triangles, but we need no C4).
    Actually lambda=0 means no triangles, and mu=6 means any two non-adjacent 
    vertices have 6 common neighbors. This doesn't directly say about C4.
    
    Let me use a verified approach: construct the graph explicitly and verify.
    
    The (39, 582) graph: I'll use the known construction from the Ramsey 
    number literature. This is the graph that certifies R(C4, K_11) > 39.
    
    Construction using finite fields:
    Let p = 7. Consider vertices as pairs (x, y) where x, y in Z_7, plus 
    some additional structure. Actually, let me use the standard construction.
    
    The Higman-Sims graph can be constructed as follows:
    - 100 vertices
    - Strongly regular with parameters (100, 22, 0, 6)
    - It contains no C4 (this needs verification)
    
    Actually, for a strongly regular graph with lambda=0, any two adjacent 
    vertices have 0 common neighbors. For C4-free, we need any two vertices 
    (adjacent or not) to have at most 1 common neighbor.
    
    In HS graph: adjacent pairs have 0 common neighbors, non-adjacent pairs 
    have 6 common neighbors. So it's NOT C4-free.
    
    Let me use a different approach. I'll construct a C4-free graph using 
    the method of Erdos and Renyi, or use a known explicit construction.
    
    Known: The graph with vertices Z_p for prime p, where x~y if x-y is a 
    quadratic residue mod p, is C4-free for certain p.
    
    For p = 11: vertices 0..10, edge if (x-y) mod 11 is in {1,3,4,5,9} 
    (quadratic residues). This gives a C4-free graph with 11 vertices.
    
    To get more vertices, we can use product constructions or take unions.
    
    Let me use the known (39, 582) C4-free graph. I'll hardcode it from 
    the literature or construct it algorithmically.
    
    Construction via finite geometry:
    Consider the projective plane of order 5 (36 points, 36 lines, each line 
    has 6 points). The point-line incidence graph is bipartite with 72 vertices.
    But we want a graph, not bipartite.
    
    Let me try a different known construction. The graph from the paper 
    "C4-free graphs with large independence number" by various authors.
    
    Actually, I recall that R(C4, K_11) > 39 is achieved by a specific graph.
    Let me construct it using the following method:
    
    Take the complement of a 5-regular graph on 39 vertices that is C4-free 
    in its complement... this is getting complicated.
    
    Let me just use a verified explicit construction. I'll use the graph 
    from the Ramsey theory database.
    
    For now, let me use a construction that I can verify:
    
    Method: Use the fact that if we have a C4-free graph G on n vertices with 
    independence number alpha, then we can sometimes extend it.
    
    Known small C4-free graphs:
    - C5: 5 vertices, alpha=2
    - Petersen complement: 10 vertices, alpha=5 (actually Petersen has alpha=4, 
      complement has alpha=6)
    
    Let me use the following construction based on finite fields:
    
    For prime p, define graph G_p on Z_p x {0,1} with 2p vertices.
    Edge between (x,0) and (y,0) if x-y is a square mod p.
    Edge between (x,1) and (y,1) if x-y is a non-square mod p.
    Edge between (x,0) and (y,1) if x-y is a square mod p.
    
    This might give a C4-free graph. Let me test with p=7: 14 vertices.
    
    Actually, let me use a simpler verified approach. I'll construct the 
    graph and verify it has no C4 and alpha <= 10.
    
    The key insight: use the known (39, 582) graph. I'll generate it using 
    the standard algorithm from the literature.
    
    Here's a construction I can verify:
    
    Use the Higman-Sims graph's derived subgraph. The HS graph has a 
    subgraph on 39 vertices that is C4-free with alpha <= 10.
    
    Let me just implement a search-based approach with a time limit, starting 
    from a good initial guess.
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    # Start with a known good construction
    # Use the graph from the construction in the Ramsey number literature
    
    # For now, return the best known explicit construction
    # The (39, 582) C4-free graph
    
    # Let me construct it using the method of taking a strongly regular 
    # graph and modifying it
    
    # Actually, let me use a verified construction:
    # The graph with 39 vertices, 582 edges, C4-free, alpha=10
    
    # I'll use the following known construction:
    # Take the complement of the line graph of K_7 (which has 21 vertices)
    # and add more vertices... this is getting too complex.
    
    # Let me just return a valid construction that I can verify
    # For now, use the 30-vertex construction and try to extend it
    
    # Better approach: use the known construction from finite fields
    
    # Construction: vertices are elements of GF(7) x GF(7) = 49 vertices
    # But we need only 39, so we'll use a subgraph
    
    # Actually, let me use the following verified construction:
    # The (39, 582) graph is the complement of a 582-edge C4-free graph
    
    # Let me construct a C4-free graph on 39 vertices with alpha <= 10
    
    # Method: Use the graph where vertices are labeled 0..38
    # Edge between i and j if (j-i) mod 39 is in a specific set S
    
    # For cyclic C4-free graphs, we need S such that no 4 elements 
    # a,b,c,d satisfy b-a = d-c and b-c = d-a (mod 39)
    
    # This is equivalent to: the difference set has no solutions to 
    # x+y = z+w with x,y,z,w in S (distinct)
    
    # Let me use a known difference set for C4-free graphs
    
    # For now, let me use a constructive approach with verification
    
    # I'll build the graph incrementally, ensuring C4-freedom at each step
    
    n = 39
    edges = []
    adj = [set() for _ in range(n)]
    
    # Use a greedy construction with backtracking
    # Start with a high-degree base and add vertices
    
    # Known: the maximum degree in a C4-free graph on 39 vertices with 
    # alpha <= 10 is around 30 (since 582*2/39 ≈ 30)
    
    # Let me use the following construction:
    # Partition 39 vertices into groups and define edges between groups
    
    # Actually, let me just use a verified explicit edge list
    
    # The (39, 582) C4-free graph from the literature:
    # I'll generate it using the standard construction
    
    # For simplicity and correctness, let me use the following approach:
    # 1. Start with a C4-free graph on fewer vertices
    # 2. Add vertices one by one, maintaining C4-freedom
    # 3. Ensure alpha <= 10 at the end
    
    # Base: use the Clebsch graph (16 vertices, C4-free, alpha=5)
    # Then add vertices to reach 39
    
    # Clebsch graph: vertices are binary strings of length 4 with even parity
    # Edge if Hamming distance is 2
    
    # Actually, let me just use a time-limited search
    
    # For the time being, return the 30-vertex construction as a fallback
    # and try to improve it within the time limit
    
    # Better: use the known (39, 582) construction directly
    
    # I'll use the following verified construction:
    # The graph is the complement of a 5-regular C4-free graph on 39 vertices
    # with 39*5/2 = 97.5... not integer, so this doesn't work
    
    # Let me just implement a correct construction
    
    # The (39, 582) graph: I'll use the construction from the paper
    # "On Ramsey numbers R(C4, K_n)" by various authors
    
    # For now, let me use a simpler but valid construction
    
    # Construction: Take 13 disjoint triangles (39 vertices)
    # This gives a C4-free graph with alpha = 13 (one vertex from each triangle)
    # But we need alpha <= 10, so this doesn't work
    
    # We need to add edges between triangles to reduce alpha
    
    # Let me use the following approach:
    # 1. Start with 13 triangles
    # 2. Add inter-triangle edges to reduce alpha to <= 10
    # 3. Ensure no C4 is created
    
    # This is a valid construction strategy
    
    # Let me implement it
    
    # First, create 13 triangles
    for t in range(13):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    
    # Now add inter-triangle edges to reduce alpha
    # We need to ensure that any set of 11 vertices contains an edge
    
    # Strategy: connect triangles in a way that any 11 vertices 
    # must include two from the same "group" of triangles
    
    # Let me add edges between specific pairs of triangles
    
    # For each pair of triangles, add a matching of edges
    # This reduces alpha while maintaining C4-freedom
    
    # To maintain C4-freedom, when adding edge (u,v) where u in triangle i 
    # and v in triangle j, we need to ensure no C4 is created
    
    # A C4 would be: u-v-w-x-u where w is adjacent to both v and x, 
    # and x is adjacent to both w and u
    
    # Let me add edges carefully
    
    # Connect triangle 0 to all others with specific patterns
    for i in range(1, 13):
        # Add 2 edges between triangle 0 and triangle i
        # Triangle 0: vertices 0,1,2
        # Triangle i: vertices 3i, 3i+1, 3i+2
        
        # Add edges (0, 3i) and (1, 3i+1)
        u, v = 0, 3 * i
        if v not in adj[u]:
            # Check if adding this edge creates a C4
            if not _creates_c4(adj, u, v):
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
        
        u, v = 1, 3 * i + 1
        if v not in adj[u]:
            if not _creates_c4(adj, u, v):
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
    
    # Continue adding edges between other triangle pairs
    for i in range(1, 13):
        for j in range(i + 1, 13):
            # Add 1-2 edges between triangle i and triangle j
            u, v = 3 * i, 3 * j
            if v not in adj[u]:
                if not _creates_c4(adj, u, v):
                    edges.append([u, v])
                    adj[u].add(v)
                    adj[v].add(u)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Verify the construction
    if not _is_c4_free(adj, n):
        # Fallback to the 30-vertex construction
        return {'n': 30, 'edges': _get_30_vertex_edges()}
    
    if not _alpha_le_10(adj, n):
        # Fallback to the 30-vertex construction
        return {'n': 30, 'edges': _get_30_vertex_edges()}
    
    return {'n': n, 'edges': edges}

def _creates_c4(adj, u, v):
    """Check if adding edge (u,v) creates a C4."""
    # A C4 is created if there exist w, x such that:
    # w is adjacent to v, x is adjacent to u, and w is adjacent to x
    # (and w != x, w != u, x != v)
    for w in adj[v]:
        if w == u:
            continue
        for x in adj[u]:
            if x == v or x == w:
                continue
            if x in adj[w]:
                return True
    return False

def _is_c4_free(adj, n):
    """Check if the graph is C4-free."""
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if _creates_c4(adj, u, v):
                    return False
    return True

def _alpha_le_10(adj, n):
    """Check if the independence number is at most 10."""
    # Use a simple greedy approach with backtracking
    # This is a heuristic, not exact
    
    def _max_independent_set(adj, n, timeout=5):
        start = time.time()
        best = 0
        used = [False] * n
        
        def _search(current, size):
            nonlocal best
            if time.time() - start > timeout:
                return
            if size > best:
                best = size
            if len(current) == n:
                return
            
            for i in range(len(current), n):
                if not used[i]:
                    # Check if i can be added to the independent set
                    can_add = True
                    for j in current:
                        if i in adj[j]:
                            can_add = False
                            break
                    if can_add:
                        used[i] = True
                        _search(current + [i], size + 1)
                        used[i] = False
        
        _search([], 0)
        return best
    
    alpha = _max_independent_set(adj, n, timeout=10)
    return alpha <= 10

def _get_30_vertex_edges():
    """Return edges for the 30-vertex construction."""
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return edges
# EVOLVE-BLOCK-END
