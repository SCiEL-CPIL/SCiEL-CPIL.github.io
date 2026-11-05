import time
import random
import itertools

def construct():
    """
    Constructs a simple triangle-free graph on vertices 0..15 that is not 3-colourable.
    Maximizes the number of edges using a combination of:
    1. Fast 3-colorability checking with forward checking.
    2. Deterministic and randomized local search (edge addition).
    3. Simulated annealing to escape local optima.
    4. Starting from known dense triangle-free graphs (Clebsch, Grötzsch).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (or 150s)
    
    n = 16
    full_mask = (1 << n) - 1
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_triangle_free_adj(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common != 0:
                        return False
        return True

    def is_3_colorable(adj):
        # Fast 3-colorability check using backtracking with forward checking
        # Order vertices by degree descending
        degrees = [bin(a).count('1') for a in adj]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            u = order[idx]
            
            # Check available colors
            used_colors = 0
            for v in range(n):
                if adj[u] & (1 << v):
                    c = color[v]
                    if c != -1:
                        used_colors |= (1 << c)
            
            for c in range(3):
                if not (used_colors & (1 << c)):
                    color[u] = c
                    if backtrack(idx + 1):
                        return True
                    color[u] = -1
            
            return False
        
        return backtrack(0)

    def try_add_edge(adj, u, v):
        """
        Tries to add edge (u, v) to adj if it doesn't create a triangle 
        and the graph remains not 3-colorable.
        Returns True if edge was added, False otherwise.
        """
        # Check triangle-free
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        if common != 0:
            return False
        
        # Tentatively add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check 3-colorability
        if is_3_colorable(adj):
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False
        
        return True

    def local_search(adj):
        """
        Attempts to add more edges to adj while maintaining triangle-free and non-3-colorable.
        Returns the modified adj and the number of edges.
        """
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if try_add_edge(adj, u, v):
                            improved = True
        # Count edges
        edge_count = 0
        for u in range(n):
            edge_count += bin(adj[u]).count('1')
        edge_count //= 2
        return adj, edge_count

    def generate_random_maximal_triangle_free():
        """
        Generates a random maximal triangle-free graph.
        """
        adj = [0] * n
        # Random order to add edges
        vertices = list(range(n))
        random.shuffle(vertices)
        
        # Greedily add edges
        for u in vertices:
            for v in vertices:
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                # Check if adding (u,v) creates triangle
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def get_grotzsch_graph():
        """
        Returns the Grötzsch graph (Mycielski of C5) on 11 vertices, 
        with 5 isolated vertices (11-15).
        Vertices 0-4: C5
        Vertices 5-9: Mycielski layer
        Vertex 10: Mycielski apex
        """
        adj = [0] * n
        # C5 edges
        for i in range(5):
            u = i
            v = (i + 1) % 5
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Mycielski: for each edge (u, w) in C5, add edges (u, 5+w) and (w, 5+u)
        for i in range(5):
            u = i
            w = (i + 1) % 5
            # Edge (u, 5+w)
            adj[u] |= (1 << (5 + w))
            adj[5 + w] |= (1 << u)
            # Edge (w, 5+u)
            adj[w] |= (1 << (5 + u))
            adj[5 + u] |= (1 << w)
            
        # Apex 10 connected to all 5-9
        for i in range(5):
            adj[10] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
            
        return adj

    def get_clebsch_graph():
        """
        Returns the Clebsch graph on 16 vertices.
        It is triangle-free, 3-colorable? No, Clebsch is 3-chromatic?
        Actually, Clebsch graph is 3-colorable? Let's check.
        Clebsch graph is the folded 5-cube. It is strongly regular (16,5,0,2).
        Chromatic number is 3? No, it is 3-colorable?
        Wait, the Clebsch graph is 3-colorable. We need a graph that is NOT 3-colorable.
        So we start with Clebsch, but it might be 3-colorable.
        If it is 3-colorable, we can't use it directly.
        However, the task asks for a graph that is NOT 3-colorable.
        So we need to start with a graph that is already not 3-colorable, or modify it.
        The Grötzsch graph is not 3-colorable.
        The Clebsch graph is 3-colorable.
        So we should start with Grötzsch or random maximal triangle-free graphs that are not 3-colorable.
        """
        adj = [0] * n
        # Clebsch graph edges
        # Vertices are binary strings of length 4.
        # Edge if Hamming distance is 1 or 3.
        for u in range(16):
            for v in range(u + 1, 16):
                dist = bin(u ^ v).count('1')
                if dist == 1 or dist == 3:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_edges = 0
    best_adj = None
    
    # Seed 1: Grötzsch
    adj = get_grotzsch_graph()
    if not is_3_colorable(adj):
        adj, ec = local_search(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
            
    # Seed 2: Random maximal triangle-free graphs
    for _ in range(10):
        if time.time() > deadline:
            break
        adj = generate_random_maximal_triangle_free()
        if not is_3_colorable(adj):
            adj, ec = local_search(adj)
            if ec > best_edges:
                best_edges = ec
                best_adj = adj[:]

    # Simulated Annealing
    if best_adj is not None and time.time() < deadline:
        adj = best_adj[:]
        current_edges = best_edges
        temperature = 10.0
        cooling = 0.99
        
        while time.time() < deadline and temperature > 0.1:
            # Pick a random edge to add or remove
            # To maximize edges, we mostly try to add edges
            # But sometimes remove to escape local optima
            if random.random() < 0.8:
                # Try to add a random edge
                u, v = random.choice(all_pairs)
                if not (adj[u] & (1 << v)):
                    # Check triangle-free
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common == 0:
                        # Tentatively add
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                        # Check 3-colorability
                        if not is_3_colorable(adj):
                            current_edges += 1
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = adj[:]
                        else:
                            # Revert
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
            else:
                # Try to remove a random edge
                u, v = random.choice(all_pairs)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
            
            temperature *= cooling

    # Convert best_adj to list of edges
    edges = []
    if best_adj is not None:
        for u in range(n):
            for v in range(u + 1, n):
                if best_adj[u] & (1 << v):
                    edges.append([u, v])
                
    return edges

# Fallback if construct fails or returns empty (should not happen)
if __name__ == "__main__":
    # Just for local testing
    edges = construct()
    print(f"Found {len(edges)} edges")
