import random
import time
import math

def construct():
    n = 16
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_3_colorable(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            used = 0
            for u in range(n):
                if adj[v] & (1 << u):
                    if colors[u] != -1:
                        used |= (1 << colors[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
            colors[v] = -1
            return False
        
        return backtrack(0)

    # Generate Clebsch Graph
    # Clebsch graph vertices can be represented as binary strings of length 4 with even parity?
    # Or simply use known edge list.
    # Clebsch graph is the halved 5-cube.
    # Vertices: 0..15.
    # Two vertices are connected if their Hamming distance is 2 or 6?
    # In 5-cube, distance 2. Halved means we take one bipartition of the 5-cube?
    # No, the Clebsch graph is the graph where vertices are the 16 binary strings of length 4 with even weight?
    # Let's check: There are 16 such strings.
    # Distance between two even-weight strings:
    # If they differ in 2 positions, the new weight changes by -2, 0, or +2.
    # Actually, the Clebsch graph is the graph on the 16 even-weight binary strings of length 4,
    # where two vertices are adjacent if their Hamming distance is 2.
    
    even_weight_vertices = []
    for i in range(16):
        if bin(i).count('1') % 2 == 0:
            even_weight_vertices.append(i)
            
    # Map index 0..15 to these vertices? 
    # The task requires vertices 0..15. 
    # We can just use the indices 0..15 directly if we define the edges properly.
    # Let's define the Clebsch graph edges explicitly for vertices 0..15.
    # The Clebsch graph is strongly regular (16, 5, 0, 2).
    # It is the complement of the 16x16 grid graph? No.
    # Let's generate it using the binary representation.
    # Let vertices be 0..15.
    # u ~ v if popcount(u ^ v) == 2.
    
    clebsch_edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if bin(u ^ v).count('1') == 2:
                clebsch_edges.append([u, v])
                
    # Verify Clebsch properties
    # 16 vertices, 5*16/2 = 40 edges.
    # Triangle free?
    # If u~v (dist 2) and v~w (dist 2), is u~w?
    # u^v has 2 bits. v^w has 2 bits.
    # u^w = (u^v) ^ (v^w).
    # If u^v and v^w are disjoint, dist is 4.
    # If they share 1 bit, dist is 2.
    # If they share 2 bits, dist is 0 (u=w).
    # So triangles can exist if dist(u,v)=2, dist(v,w)=2, dist(u,w)=2.
    # Example: 0011 (3), 0001 (1), 0010 (2).
    # 3^1 = 0010 (1 bit? No, 3=0011, 1=0001, xor=0010, popcount 1).
    # Wait, my definition: u~v if popcount(u^v)==2.
    # 3 (0011) and 1 (0001): xor 0010, popcount 1. Not connected.
    # 3 (0011) and 2 (0010): xor 0001, popcount 1. Not connected.
    # 3 (0011) and 7 (0111): xor 0100, popcount 1.
    # 3 (0011) and 5 (0101): xor 0110, popcount 2. Connected.
    # 5 (0101) and 7 (0111): xor 0010, popcount 1.
    # 3 (0011) and 6 (0110): xor 0101, popcount 2. Connected.
    # 5 (0101) and 6 (0110): xor 0011, popcount 2. Connected.
    # 3~5, 5~6, 3~6. Triangle!
    # So the graph defined by popcount(u^v)==2 on all 16 vertices is NOT triangle-free.
    # The Clebsch graph is the subgraph induced by even-weight vertices?
    # Let's check even-weight vertices: 0, 3, 5, 6, 9, 10, 12, 15.
    # 0 (0000)
    # 3 (0011)
    # 5 (0101)
    # 6 (0110)
    # 9 (1001)
    # 10 (1010)
    # 12 (1100)
    # 15 (1111)
    # 
    # 3~5? 3^5=6 (0110), popcount 2. Yes.
    # 5~6? 5^6=3 (0011), popcount 2. Yes.
    # 3~6? 3^6=5 (0101), popcount 2. Yes.
    # Triangle 3-5-6.
    # So even-weight subgraph is not triangle-free either?
    
    # Let's check the standard definition.
    # The Clebsch graph is the halved 5-cube.
    # Vertices of 5-cube are binary strings of length 5.
    # Bipartition into even and odd weight.
    # The Clebsch graph is the graph on the even-weight vertices of the 5-cube,
    # where two vertices are adjacent if their distance in the 5-cube is 2.
    # There are 16 even-weight vertices in the 5-cube.
    # Let's generate this.
    
    clebsch_vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            clebsch_vertices.append(i)
            
    # Map these 16 vertices to 0..15
    vertex_map = {v: i for i, v in enumerate(clebsch_vertices)}
    inv_vertex_map = {i: v for i, v in enumerate(clebsch_vertices)}
    
    clebsch_edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            u = clebsch_vertices[i]
            v = clebsch_vertices[j]
            if bin(u ^ v).count('1') == 2:
                clebsch_edges.append([i, j])
                
    # Check triangle freeness
    def is_triangle_free(edges):
        adj = [0] * 16
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(16):
            for v in range(u + 1, 16):
                if adj[u] & (1 << v):
                    if adj[u] & adj[v]:
                        return False
        return True

    # The Clebsch graph is known to be triangle-free.
    
    # Simulated Annealing
    current_edges = set(tuple(e) for e in clebsch_edges)
    current_adj = [0] * 16
    for u, v in current_edges:
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        
    def get_energy(edges_set, adj):
        e_count = len(edges_set)
        # Check 3-colorability
        # This is the bottleneck.
        col = is_3_colorable(list(edges_set))
        penalty = 50 if col else 0
        return -e_count + penalty, col, e_count

    # Initial state
    curr_energy, curr_is_3col, curr_e = get_energy(current_edges, current_adj)
    best_edges = list(current_edges)
    best_energy = curr_energy
    best_is_3col = curr_is_3col
    best_e = curr_e
    
    if not curr_is_3col:
        best_edges = list(current_edges)
        best_energy = curr_energy
        best_is_3col = curr_is_3col
        best_e = curr_e
    else:
        # Should not happen for Clebsch
        best_edges = []
        best_energy = float('inf')
        best_is_3col = True
        best_e = 0

    start_time = time.time()
    time_limit = 110.0
    T = 50.0
    cooling = 0.995
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit and T > 0.1:
        # Choose move
        # 50% Add, 50% Remove
        if rng.random() < 0.5:
            # Add edge
            # Find non-edges that don't create triangles
            candidates = []
            for u in range(16):
                for v in range(u + 1, 16):
                    if not (current_adj[u] & (1 << v)):
                        if not (current_adj[u] & current_adj[v]):
                            candidates.append((u, v))
            if not candidates:
                T *= cooling
                continue
            u, v = rng.choice(candidates)
            new_edges = current_edges | {(u, v)}
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            new_energy, new_is_3col, new_e = get_energy(new_edges, new_adj)
            delta_e = new_energy - curr_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_edges = new_edges
                current_adj = new_adj
                curr_energy = new_energy
                curr_is_3col = new_is_3col
                curr_e = new_e
                
                if not new_is_3col:
                    if new_e > best_e:
                        best_e = new_e
                        best_edges = list(new_edges)
                        best_energy = new_energy
                        best_is_3col = new_is_3col
        else:
            # Remove edge
            if not current_edges:
                T *= cooling
                continue
            u, v = rng.choice(list(current_edges))
            new_edges = current_edges - {(u, v)}
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            new_energy, new_is_3col, new_e = get_energy(new_edges, new_adj)
            delta_e = new_energy - curr_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_edges = new_edges
                current_adj = new_adj
                curr_energy = new_energy
                curr_is_3col = new_is_3col
                curr_e = new_e
                
                if not new_is_3col:
                    if new_e > best_e:
                        best_e = new_e
                        best_edges = list(new_edges)
                        best_energy = new_energy
                        best_is_3col = new_is_3col
                        
        T *= cooling
        
    if not best_edges:
        # Fallback to Clebsch
        return clebsch_edges
        
    return best_edges
