import time
import random

def construct():
    """
    Constructs a simple triangle-free graph on vertices 0..15 that is not 3-colourable.
    Maximizes the number of edges using random restarts and local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    full_mask = (1 << n) - 1
    
    def is_triangle_free_adj(adj):
        # adj is a list of bitmasks
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Check if u and v share a common neighbor
                    # Common neighbors = adj[u] & adj[v]
                    # We must exclude u and v themselves from the check if they are in each other's mask?
                    # No, adj[u] contains v if edge exists. adj[v] contains u.
                    # Common neighbors w such that w!=u, w!=v.
                    common = adj[u] & adj[v]
                    # Remove u and v from common
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common != 0:
                        return False
        return True

    def is_3_colorable(adj):
        # Static ordering: sort vertices by degree descending
        degrees = [bin(a).count('1') for a in adj]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            u = order[idx]
            
            # Check available colors
            # Colors used by neighbors of u
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
        # If u and v have a common neighbor, adding edge creates triangle
        common = adj[u] & adj[v]
        # Exclude u and v themselves
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

    best_edges = 0
    best_adj = None
    
    # Seed 1: Grötzsch
    adj = get_grotzsch_graph()
    if not is_3_colorable(adj):
        adj, ec = local_search(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
            
    # Random restarts
    while time.time() < deadline:
        adj = generate_random_maximal_triangle_free()
        
        # Check if it's not 3-colorable
        if is_3_colorable(adj):
            continue
            
        # Local search
        adj, ec = local_search(adj)
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
            
            # If we reach a very high number, we might be near optimal
            # Max possible is 64 (K8,8), but that is 3-colorable.
            # Known bounds for f4(16) are likely around 30-40.
            if best_edges > 40:
                # Try to break early if we have a very strong result? 
                # No, keep searching for a bit more.
                pass

    # Convert best_adj to list of edges
    edges = []
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
