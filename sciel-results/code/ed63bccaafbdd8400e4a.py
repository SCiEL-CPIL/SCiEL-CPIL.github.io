import time
import random
import numpy as np

def _compute_max_book(adj, n=40):
    """Compute max book size using matrix multiplication."""
    tri = (adj @ adj).astype(np.int32)
    max_book = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                if tri[u, v] > max_book:
                    max_book = tri[u, v]
    return max_book

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 105.0  # Safety margin
    
    random.seed(42)
    
    # Initial construction: K_{20,20} + 1 edge
    # This gives a very regular graph with max book = 20 (each cross edge in 20 triangles)
    # Wait, that's too high. Let me think again.
    # In K_{20,20}, each edge (u,v) with u in left, v in right has 0 common neighbors 
    # (bipartite, no triangles). So max book = 0 for K_{20,20}.
    # Adding one edge (0,1) within left: this edge has common neighbors = all 20 right vertices.
    # So max book = 20. That's terrible.
    
    # Better: Use K_{10,30} structure from current program.
    # K_{10,30} has 300 edges, max book = 0 (bipartite).
    # Need 101 more edges within the 30-vertex part.
    # If we add edges within B (30 vertices), each such edge (u,v) in B has common neighbors
    # = |N(u) ∩ N(v)|. Since all 10 A vertices connect to both u and v, 
    # every B-B edge has at least 10 common neighbors.
    # So max book >= 10 for any B-B edge.
    # Can we do better? What if we use a different split?
    
    # Actually, let's use the K_{20,20} + 1 edge but with local search to reduce max book.
    # The issue is that the single intra-part edge has book size 20.
    # Local search should be able to move edges to reduce this.
    
    # Alternative: Start with a random graph with 401 edges, then optimize.
    # But structured start is better.
    
    # Let me try: K_{10,30} + 101 random edges in B, then optimize.
    # Max book starts at ~10, local search can reduce it.
    
    a_size = 10
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    adj = np.zeros((n, n), dtype=np.uint8)
    edges_set = set()
    
    # Add all cross edges A-B
    for u in A:
        for v in B:
            adj[u, v] = 1
            adj[v, u] = 1
            edges_set.add((u, v))
    
    num_cross = a_size * (n - a_size)  # 300
    edges_needed = target_edges - num_cross  # 101
    
    # Generate G_B with low degree to minimize triangles
    b_indices = list(range(len(B)))
    possible_edges = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            possible_edges.append((i, j))
    random.shuffle(possible_edges)
    
    deg = [0] * len(B)
    max_deg_target = 7  # Keep degrees low
    
    for i, j in possible_edges:
        if len(edges_set) == target_edges:
            break
        if deg[i] < max_deg_target and deg[j] < max_deg_target:
            u, v = B[i], B[j]
            adj[u, v] = 1
            adj[v, u] = 1
            edges_set.add((u, v))
            deg[i] += 1
            deg[j] += 1
    
    # If we didn't reach target, add more edges
    if len(edges_set) < target_edges:
        for i, j in possible_edges:
            if len(edges_set) == target_edges:
                break
            u, v = B[i], B[j]
            if (u, v) not in edges_set and (v, u) not in edges_set:
                adj[u, v] = 1
                adj[v, u] = 1
                edges_set.add((u, v))
    
    # Ensure exactly target_edges
    while len(edges_set) > target_edges:
        # Remove a random edge from B-B
        b_edges = [(u, v) for u, v in edges_set if u >= a_size and v >= a_size]
        if b_edges:
            u, v = random.choice(b_edges)
            edges_set.discard((u, v))
            edges_set.discard((v, u))
            adj[u, v] = 0
            adj[v, u] = 0
    
    current_max = _compute_max_book(adj, n)
    best_adj = adj.copy()
    best_max = current_max
    
    # Local search with NumPy
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Pick an edge to remove - bias toward high-book edges
        # Get all edges and their book sizes
        tri = (adj @ adj).astype(np.int32)
        
        # Find edges with high book size
        edge_books = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edge_books.append((tri[u, v], u, v))
        
        if not edge_books:
            break
        
        # Sort by book size descending, pick from top 20%
        edge_books.sort(reverse=True)
        top_k = max(1, len(edge_books) // 5)
        u_rem, v_rem = random.choice(edge_books[:top_k])
        
        # Try several random swaps
        found_improvement = False
        for _ in range(20):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) == (u_rem, v_rem) or (x, y) == (v_rem, u_rem):
                continue
            if adj[x, y]:
                continue
            
            # Tentative swap
            adj[u_rem, v_rem] = 0
            adj[v_rem, u_rem] = 0
            adj[x, y] = 1
            adj[y, x] = 1
            
            new_max = _compute_max_book(adj, n)
            
            if new_max < best_max:
                best_max = new_max
                best_adj = adj.copy()
                found_improvement = True
            else:
                # Revert
                adj[u_rem, v_rem] = 1
                adj[v_rem, u_rem] = 1
                adj[x, y] = 0
                adj[y, x] = 0
            
            if found_improvement:
                break
        
        if found_improvement:
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Perturb: do several random swaps
            for _ in range(5):
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 == v1 or not adj[u1, v1]:
                    continue
                if u1 > v1:
                    u1, v1 = v1, u1
                
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2 or adj[u2, v2]:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                
                if (u1, v1) == (u2, v2):
                    continue
                
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
            
            no_improve = 0
    
    # Convert best_adj to list of edges
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                final_edges.append([u, v])
    
    # Ensure exactly target_edges
    if len(final_edges) != target_edges:
        # Fallback
        final_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    final_edges.append([u, v])
    
    return final_edges
