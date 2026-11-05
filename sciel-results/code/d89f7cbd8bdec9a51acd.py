import time
import numpy as np
import random

def construct():
    """
    Constructs a C6-free graph on 40 vertices with many edges.
    Uses simulated annealing with 2-opt moves (remove one edge, add another).
    """
    n = 40
    deadline = 115.0  # Seconds, leave safety margin
    start_time = time.time()
    
    def has_c6(adj):
        """Check if graph contains a C6 by checking all 6-cycles."""
        # A C6 exists if there are vertices v0,v1,v2,v3,v4,v5 such that
        # all consecutive pairs are edges and v0-v5 is NOT an edge (to avoid shorter cycles)
        # More efficiently: check if A^5 has any entry (i,j) where i!=j and A[i,j]==0
        # but we need to be careful about paths that revisit vertices
        
        # Use matrix approach: A^5[i,j] counts paths of length 5 from i to j
        # If A^5[i,j] > 0 and A[i,j] == 0 and i != j, there might be a C6
        # But this isn't sufficient because the path might revisit vertices
        # For small graphs, we can do a more careful check
        
        # Better approach: for each pair (u,v) that is NOT an edge, check if there's
        # a path of length 5 between them that uses distinct intermediate vertices
        # This is expensive, so let's use a heuristic check first
        
        A2 = adj @ adj
        A3 = A2 @ adj
        A5 = A3 @ (A2)
        
        # For each non-edge (i,j), if A5[i,j] > 0, there's a path of length 5
        # But we need to ensure it's a simple path (no repeated vertices)
        # For our purposes, we'll accept this as a sufficient condition for C6 detection
        # since in dense graphs, repeated vertices in short paths are less likely
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0 and i != j:
                    if A5[i, j] > 0:
                        # Potential C6 - verify with BFS
                        if _verify_c6(adj, i, j):
                            return True
        return False
    
    def _verify_c6(adj, start, end):
        """Verify there's a simple path of length 5 from start to end."""
        # BFS with path tracking
        from collections import deque
        
        # State: (current_vertex, visited_set, path_length)
        queue = deque()
        queue.append((start, frozenset([start]), 0))
        
        while queue:
            curr, visited, length = queue.popleft()
            if length == 5:
                if curr == end and end not in visited:
                    return True
                continue
            if length >= 5:
                continue
            for neighbor in range(n):
                if adj[curr, neighbor] == 1 and neighbor not in visited:
                    new_visited = visited | frozenset([neighbor])
                    queue.append((neighbor, new_visited, length + 1))
        return False
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6."""
        # Adding (u,v) creates a C6 iff there's a path of length 5 from u to v
        # in the current graph
        # Use A^5 check
        A2 = adj @ adj
        A3 = A2 @ adj
        A5 = A3 @ A2
        return A5[u, v] == 0
    
    def count_edges(adj):
        return int(adj.sum() // 2)
    
    def get_all_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))
        return non_edges
    
    def get_all_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    edges.append((i, j))
        return edges
    
    # Initialize: start with random dense graph and remove C6s
    np.random.seed(42)
    random.seed(42)
    
    # Start with a moderately dense random graph
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < 0.4:
                A[i, j] = 1
                A[j, i] = 1
    
    # Remove edges that create C6s (greedy removal)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    # Temporarily remove
                    A[i, j] = 0
                    A[j, i] = 0
                    # Check if graph is C6-free
                    A2 = A @ A
                    A3 = A2 @ A
                    A5 = A3 @ A2
                    # Check if any non-edge has path length 5
                    has_c6_now = False
                    for x in range(n):
                        for y in range(x + 1, n):
                            if A[x, y] == 0 and x != y:
                                if A5[x, y] > 0:
                                    has_c6_now = True
                                    break
                        if has_c6_now:
                            break
                    if has_c6_now:
                        # Keep edge removed
                        changed = True
                    else:
                        # Restore edge
                        A[i, j] = 1
                        A[j, i] = 1
    
    num_edges = count_edges(A)
    best_num_edges = num_edges
    best_A = A.copy()
    
    # Simulated annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() - start_time < deadline and T > min_T:
        # Pick a random move: remove one edge and add another (swap)
        edges_list = get_all_edges(A)
        non_edges_list = get_all_non_edges(A)
        
        if not edges_list or not non_edges_list:
            break
        
        u_out, v_out = edges_list[random.randint(0, len(edges_list) - 1)]
        u_in, v_in = non_edges_list[random.randint(0, len(non_edges_list) - 1)]
        
        # Temporarily remove edge
        A[u_out, v_out] = 0
        A[v_out, u_out] = 0
        
        # Check if we can add the new edge
        if can_add_edge(A, u_in, v_in):
            # Compute new edge count (same as before since it's a swap)
            A[u_in, v_in] = 1
            A[v_in, u_in] = 1
            
            # Check if this creates a C6 (it shouldn't, but verify)
            # Actually, can_add_edge already ensures no C6 from the new edge
            # But the removal might have broken existing C6s, so we're fine
            
            delta_edges = 0  # Swap keeps edge count same
            # Accept with probability 1 for same edge count (neutral move)
            # This helps escape local minima
        else:
            # Revert
            A[u_out, v_out] = 1
            A[v_out, u_out] = 1
            T *= cooling
            continue
        
        # Try to improve: after the swap, try to add more edges
        improved = False
        for _ in range(10):  # Try a few greedy additions
            non_edges_now = get_all_non_edges(A)
            if not non_edges_now:
                break
            u, v = non_edges_now[random.randint(0, len(non_edges_now) - 1)]
            if can_add_edge(A, u, v):
                A[u, v] = 1
                A[v, u] = 1
                num_edges += 1
                improved = True
                if num_edges > best_num_edges:
                    best_num_edges = num_edges
                    best_A = A.copy()
        
        if not improved:
            # Try removing an edge to allow future additions
            if random.random() < 0.3:
                edges_now = get_all_edges(A)
                if edges_now:
                    u_r, v_r = edges_now[random.randint(0, len(edges_now) - 1)]
                    A[u_r, v_r] = 0
                    A[v_r, u_r] = 0
                    num_edges -= 1
        
        T *= cooling
    
    # Final greedy improvement
    for _ in range(100):
        non_edges_now = get_all_non_edges(best_A)
        if not non_edges_now:
            break
        u, v = non_edges_now[random.randint(0, len(non_edges_now) - 1)]
        if can_add_edge(best_A, u, v):
            best_A[u, v] = 1
            best_A[v, u] = 1
            best_num_edges += 1
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                result.append([i, j])
    
    return result
