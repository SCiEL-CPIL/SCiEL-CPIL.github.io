import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    n = 16
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    
    best_edges = 0
    best_graph = set()
    
    # Helper to check if adding edge (u, v) creates Q3
    # Uses bitmask adjacency for speed
    def creates_q3(adj, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency bitmasks adj creates a Q3.
        adj already includes the (u,v) edge when called.
        """
        nu = adj[u]
        nv = adj[v]
        
        # Find all (a,b) pairs forming a square through (u,v)
        # a is neighbor of v (not u), b is neighbor of u (not v), a~b
        # In the square u-v-a-b-u: u~v, v~a, a~b, b~u
        pairs = []
        # Iterate over neighbors of v (excluding u)
        a_mask = nv & ~(1 << u)
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # b must be neighbor of u (excluding v), and a~b
            b_mask = (nu & ~(1 << v)) & adj[a]
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                if b != a:
                    pairs.append((a, b))
        
        if len(pairs) < 2:
            return False
        
        # Try all pairs of distinct pairs
        for i in range(len(pairs)):
            a, b = pairs[i]
            for j in range(i+1, len(pairs)):
                c, d = pairs[j]
                # Check distinctness
                if a == c or a == d or b == c or b == d:
                    continue
                
                used = {u, v, a, b, c, d}
                used_mask = 0
                for x in used:
                    used_mask |= (1 << x)
                
                # Find e: adjacent to a and c, not in used
                e_mask = adj[a] & adj[c] & ~used_mask
                while e_mask:
                    e_bit = e_mask & (-e_mask)
                    e = e_bit.bit_length() - 1
                    e_mask ^= e_bit
                    
                    # Find f: adjacent to b and d, not in used, not e, and e~f
                    f_mask = adj[b] & adj[d] & ~used_mask & ~(1 << e) & adj[e]
                    if f_mask:
                        return True
        
        return False
    
    # Multi-restart greedy with local search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        random.shuffle(all_edges)
        
        # Build adjacency bitmask
        adj = [0] * n
        current_graph = set()
        
        for edge in all_edges:
            u, v = edge
            # Temporarily add u-v to adj for checking
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if creates_q3(adj, u, v):
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                current_graph.add((u, v))
        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
        
        # Local search: try to add any missing edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for edge in all_edges:
                if edge in current_graph:
                    continue
                u, v = edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not creates_q3(adj, u, v):
                    current_graph.add(edge)
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
        
        # Edge swap local search: remove one edge, try to add two others
        # This helps escape local optima
        if time.time() < deadline:
            # Rebuild adj from current_graph
            adj = [0] * n
            for u, v in current_graph:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Try removing each edge and see if we can add more
            edges_list = list(current_graph)
            random.shuffle(edges_list)
            
            for rem_edge in edges_list:
                if time.time() > deadline:
                    break
                ru, rv = rem_edge
                # Remove
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                current_graph.discard(rem_edge)
                
                # Try to add any missing edge
                added_any = False
                for add_edge in all_edges:
                    if add_edge in current_graph:
                        continue
                    au, av = add_edge
                    adj[au] |= (1 << av)
                    adj[av] |= (1 << au)
                    if not creates_q3(adj, au, av):
                        current_graph.add(add_edge)
                        added_any = True
                    else:
                        adj[au] &= ~(1 << av)
                        adj[av] &= ~(1 << au)
                
                if len(current_graph) > best_edges:
                    best_edges = len(current_graph)
                    best_graph = set(current_graph)
                
                # If we couldn't add more, put the edge back
                if not added_any or len(current_graph) <= best_edges:
                    # Put back
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    current_graph.add(rem_edge)
        
        # Rebuild adj for next iteration
        adj = [0] * n
        for u, v in current_graph:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Safety: limit total restarts to avoid infinite loop
        if num_restarts > 2000:
            break
    
    # Convert to list of lists
    return [[u, v] for u, v in best_graph]
