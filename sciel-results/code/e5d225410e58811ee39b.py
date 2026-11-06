import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n = 18
    # Bitmask-based adjacency: adj_mask[i] is a bitmask of neighbors of vertex i
    adj_mask = [0] * n
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        if (adj_mask[u] >> v) & 1:
            return False
        # Common neighbors of u and v (excluding u and v themselves)
        common = adj_mask[u] & adj_mask[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        # Check if any pair in common has an edge (would form K4 with u,v)
        # For each w in common, check if w has any neighbor in common
        for w in range(n):
            if (common >> w) & 1:
                # Check if w has any neighbor in common
                if adj_mask[w] & common:
                    return False
        return True
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def is_triangle_free(subset_mask):
        """Check if the induced subgraph on subset_mask is triangle-free."""
        # A triangle exists if there are vertices a,b,c in subset where all three edges exist
        # For each vertex a in subset, check if any two neighbors of a are also in subset and connected
        m = subset_mask
        while m:
            a_bit = m & (-m)
            a = a_bit.bit_length() - 1
            m ^= a_bit
            # Neighbors of a within subset
            neighbors_in_subset = adj_mask[a] & subset_mask
            neighbors_in_subset &= ~a_bit
            # Check if any two neighbors are connected (forming triangle)
            nn = neighbors_in_subset
            while nn:
                b_bit = nn & (-nn)
                b = b_bit.bit_length() - 1
                nn ^= b_bit
                # Check if b has any neighbor in neighbors_in_subset
                if adj_mask[b] & neighbors_in_subset:
                    return False
        return True
    
    def get_max_triangle_free_size():
        """Find largest k such that there is a triangle-free induced subgraph of size k."""
        # Use branch and bound: try subsets of size k from n down
        for k in range(n, 0, -1):
            # Check all subsets of size k
            for combo in itertools.combinations(range(n), k):
                subset_mask = 0
                for v in combo:
                    subset_mask |= (1 << v)
                if is_triangle_free(subset_mask):
                    return k
        return 0
    
    def get_max_triangle_free_size_and_set():
        """Returns (size, subset_mask) of largest triangle-free induced subgraph."""
        for k in range(n, 0, -1):
            for combo in itertools.combinations(range(n), k):
                subset_mask = 0
                for v in combo:
                    subset_mask |= (1 << v)
                if is_triangle_free(subset_mask):
                    return k, subset_mask
        return 0, 0
    
    best_adj_mask = [0] * n
    best_score = n
    
    # Random restarts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Reset graph
        adj_mask = [0] * n
        
        # Add random edges to make it "dense" but K4-free
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if can_add_edge(u, v):
                add_edge(u, v)
        
        current_score, current_set = get_max_triangle_free_size_and_set()
        
        if current_score < best_score:
            best_score = current_score
            best_adj_mask = adj_mask[:]
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_score, current_set = get_max_triangle_free_size_and_set()
            
            if current_set == 0:
                break
            
            # Try to add an edge within current_set
            # Get list of vertices in current_set
            vertices_in_set = [v for v in range(n) if (current_set >> v) & 1]
            
            # Try all pairs in the set that don't have edges yet
            for i in range(len(vertices_in_set)):
                for j in range(i+1, len(vertices_in_set)):
                    u = vertices_in_set[i]
                    v = vertices_in_set[j]
                    if not ((adj_mask[u] >> v) & 1) and can_add_edge(u, v):
                        # Add edge
                        add_edge(u, v)
                        
                        new_score, _ = get_max_triangle_free_size_and_set()
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            # Revert
                            remove_edge(u, v)
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_adj_mask = adj_mask[:]

    # Convert best_adj_mask to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_mask[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
