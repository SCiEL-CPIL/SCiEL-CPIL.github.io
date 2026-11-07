"""Rewritten program for spectral graph lifts task."""
# EVOLVE-BLOCK-START
import numpy as np

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Identify the structure:
    # Two m-cycles: vertices 0..m-1 and m..2m-1
    # Matching edges: (i, i+m) for i in 0..m-1
    # 
    # For a prism graph, the edges are:
    # - Top cycle: (i, i+1 mod m) for i in 0..m-1
    # - Bottom cycle: (m+i, m+(i+1 mod m)) for i in 0..m-1
    # - Matching: (i, m+i) for i in 0..m-1
    # Total: 3m edges
    #
    # The lift is 3-regular and connected iff it's a valid 2-lift.
    # 
    # Strategy: Exploit gauge invariance. Fix signs of a spanning tree to +1.
    # The graph has 2m vertices and 3m edges. A spanning tree has 2m-1 edges.
    # So there are 3m - (2m-1) = m+1 non-tree edges (chords).
    # 
    # For m <= 9, 2^(m+1) <= 2^10 = 1024, which is trivially fast.
    #
    # We need to:
    # 1. Build the adjacency structure
    # 2. Find a spanning tree
    # 3. Identify the m+1 chord edges
    # 4. Enumerate all 2^(m+1) sign assignments for chords
    # 5. For each, check if the lift is valid (connected, 3-regular)
    # 6. Compute spectral radius of signed adjacency matrix
    # 7. Return the best valid assignment
    
    # Build edge list with indices
    # edges is a list of (u, v) tuples
    
    # Find a spanning tree using BFS/DFS on the base graph
    # Vertices: 0 to n-1
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # BFS to find spanning tree
    visited = [False] * n
    parent_edge = [-1] * n  # edge index used to reach this vertex
    tree_edges = set()
    from collections import deque
    queue = deque([0])
    visited[0] = True
    while queue:
        u = queue.popleft()
        for v, eidx in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent_edge[v] = eidx
                tree_edges.add(eidx)
                queue.append(v)
    
    # Chord edges: all edges not in the spanning tree
    chord_edges = [i for i in range(num_edges) if i not in tree_edges]
    num_chords = len(chord_edges)
    
    # Build signed adjacency matrix function
    # The signed adjacency matrix A_s has entries:
    # A_s[u][v] = sign(e) if (u,v) is an edge with sign(e), else 0
    # For undirected edges, A_s is symmetric: A_s[u][v] = A_s[v][u] = sign(e)
    
    # Precompute the base structure
    # For each chord assignment, we compute the full sign vector and the spectral radius
    
    def compute_spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        # Use eigvalsh for symmetric matrix
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the 2-lift defined by signs is connected and 3-regular."""
        # The 2-lift has vertices (v, 0) and (v, 1) for v in 0..n-1
        # Total 2n vertices
        # Edge (u, v) with sign s:
        #   if s = +1: connects (u,0)-(v,0) and (u,1)-(v,1)
        #   if s = -1: connects (u,0)-(v,1) and (u,1)-(v,0)
        # 
        # Each vertex in the lift has degree 3 (same as base graph)
        # So 3-regular is automatic if the base graph is 3-regular.
        # We just need to check connectivity.
        
        num_lift_vertices = 2 * n
        lift_adj = [[] for _ in range(num_lift_vertices)]
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # Connect (u,0)-(v,0) and (u,1)-(v,1)
                lift_adj[u].append(v)
                lift_adj[v].append(u)
                lift_adj[u + n].append(v + n)
                lift_adj[v + n].append(u + n)
            else:
                # Connect (u,0)-(v,1) and (u,1)-(v,0)
                lift_adj[u].append(v + n)
                lift_adj[v + n].append(u)
                lift_adj[u + n].append(v)
                lift_adj[v].append(u + n)
        
        # BFS to check connectivity
        visited = [False] * num_lift_vertices
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            u = queue.popleft()
            for v in lift_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        
        return count == num_lift_vertices
    
    # Enumerate all 2^num_chords assignments
    best_radius = float('inf')
    best_signs = None
    
    # Start with all tree edges = +1
    base_signs = np.ones(num_edges, dtype=np.int8)
    
    for mask in range(1 << num_chords):
        signs = base_signs.copy()
        for i, chord_idx in enumerate(chord_edges):
            if (mask >> i) & 1:
                signs[chord_idx] = -1
            else:
                signs[chord_idx] = 1
        
        if is_valid_lift(signs):
            radius = compute_spectral_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: should not happen for prism graphs, but just in case
        # Return all +1s
        return np.ones(num_edges, dtype=np.int8)
    
    return best_signs
# EVOLVE-BLOCK-END
