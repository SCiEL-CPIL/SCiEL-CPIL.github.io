import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 18
    all_vertices_mask = (1 << n) - 1
    
    # Helper to check if graph is K4-free using bitmask
    def is_k4_free(adj_bits):
        # adj_bits[v] is a bitmask of neighbors of v
        # Check all 4-cliques
        for quad in itertools.combinations(range(n), 4):
            u, v, w, x = quad
            if ((adj_bits[u] >> v) & 1 and (adj_bits[u] >> w) & 1 and (adj_bits[u] >> x) & 1 and
                (adj_bits[v] >> w) & 1 and (adj_bits[v] >> x) & 1 and (adj_bits[w] >> x) & 1):
                return False
        return True

    # Helper to calculate score: largest triangle-free induced subgraph
    def calc_score(adj_bits):
        # Branch and bound for max triangle-free set
        best_size = 0
        
        # Precompute: for each vertex, its neighbors as bitmask
        # adj_bits[v] already gives us this
        
        def has_triangle_in_mask(mask):
            # Check if induced subgraph on vertices in mask has a triangle
            if mask.bit_count() < 3:
                return False
            # Iterate through vertices in mask
            verts = [v for v in range(n) if (mask >> v) & 1]
            for i in range(len(verts)):
                u = verts[i]
                # For each pair of neighbors of u that are both in mask
                neigh_u_in_mask = adj_bits[u] & mask
                # Check if any two neighbors of u are connected
                temp = neigh_u_in_mask
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    # Check if v is connected to any other neighbor of u in mask
                    if (adj_bits[v] & neigh_u_in_mask) & ~(1 << v):
                        return True
            return False
        
        def search(mask):
            nonlocal best_size
            size = mask.bit_count()
            if size <= best_size:
                return
            
            if not has_triangle_in_mask(mask):
                best_size = size
                return
            
            # Find a triangle to branch on
            verts = [v for v in range(n) if (mask >> v) & 1]
            tri = None
            for i in range(len(verts)):
                u = verts[i]
                neigh_u_in_mask = adj_bits[u] & mask
                temp = neigh_u_in_mask
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    # Check if v is connected to any other neighbor of u in mask
                    common = (adj_bits[v] & neigh_u_in_mask) & ~(1 << v)
                    if common:
                        w = (common & -common).bit_length() - 1
                        tri = (u, v, w)
                        break
                if tri:
                    break
            
            if not tri:
                best_size = max(best_size, size)
                return
                
            u, v, w = tri
            # Branch: remove u, or remove v, or remove w
            search(mask & ~(1 << u))
            search(mask & ~(1 << v))
            search(mask & ~(1 << w))
            
        search(all_vertices_mask)
        return best_size

    # Helper to convert adj_bits to edge list
    def adj_to_edges(adj_bits):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Helper to convert edge list to adj_bits
    def edges_to_adj(edges):
        adj_bits = [0] * n
        for u, v in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj_bits

    # Strategy: Start from complement of a 3-regular graph
    # The 4-dimensional hypercube Q4 has 16 vertices, but we need 18
    # Let's use a random 3-regular graph on 18 vertices
    
    def make_random_3regular(n):
        # Random 3-regular graph using configuration model
        while True:
            # Create half-edges: each vertex has 3 stubs
            stubs = []
            for v in range(n):
                for _ in range(3):
                    stubs.append(v)
            
            random.shuffle(stubs)
            
            # Pair up stubs
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if valid:
                # Verify it's 3-regular
                if all(len(adj[v]) == 3 for v in range(n)):
                    return adj
            # If not valid, retry
    
    # Generate several random 3-regular graphs and take complements
    best_adj = None
    best_score = n + 1
    
    for trial in range(50):
        # Generate random 3-regular graph
        reg_adj = make_random_3regular(n)
        
        # Complement: edge exists in complement if NOT in regular graph
        comp_adj = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v and v not in reg_adj[u]:
                    comp_adj[u] |= (1 << v)
        
        # Check K4-free
        if is_k4_free(comp_adj):
            score = calc_score(comp_adj)
            if score < best_score:
                best_score = score
                best_adj = comp_adj[:]
        
        if time.time() > deadline - 30:
            break
    
    # If no good solution found, fall back to K6,6,6
    if best_adj is None:
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        initial_edges = []
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        initial_edges.append([a, b])
        best_adj = edges_to_adj(initial_edges)
        best_score = calc_score(best_adj)
    
    # Local search to improve
    current_adj = best_adj[:]
    current_score = best_score
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random edge to toggle
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        # Check if edge is currently present
        present = (current_adj[u] >> v) & 1
        
        if present:
            # Remove edge
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            # Always K4-free
            new_score = calc_score(new_adj)
            
            if new_score < current_score:
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
            else:
                delta = new_score - current_score
                if delta > 0:
                    if random.random() < pow(2.718, -delta / T):
                        current_adj = new_adj
                        current_score = new_score
                elif random.random() < 0.5:
                    current_adj = new_adj
                    current_score = new_score
        else:
            # Add edge
            # Check if adding (u,v) creates a K4
            # Need to check all pairs (w,x) such that w,x != u,v
            # And (u,w),(u,x),(v,w),(v,x),(w,x) are all present
            
            creates_k4 = False
            # Iterate w in neighbors of u
            w_mask = current_adj[u] & ~(1 << v)
            while w_mask:
                w = (w_mask & -w_mask).bit_length() - 1
                w_mask &= w_mask - 1
                # Iterate x in neighbors of v
                x_mask = current_adj[v] & ~(1 << u) & ~(1 << w)
                while x_mask:
                    x = (x_mask & -x_mask).bit_length() - 1
                    x_mask &= x_mask - 1
                    # Check if w,x connected and u,x connected and v,w connected
                    if ((current_adj[w] >> x) & 1 and (current_adj[u] >> x) & 1 and (current_adj[v] >> w) & 1):
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if creates_k4:
                continue
                
            # Add edge
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_score = calc_score(new_adj)
            
            if new_score < current_score:
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
            else:
                delta = new_score - current_score
                if delta > 0:
                    if random.random() < pow(2.718, -delta / T):
                        current_adj = new_adj
                        current_score = new_score
                elif random.random() < 0.5:
                    current_adj = new_adj
                    current_score = new_score
                    
        T *= cooling
    
    return adj_to_edges(best_adj)

# EVOLVE-BLOCK-END
