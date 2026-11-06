import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    
    # Initialize with K_20,20 plus one edge (0,1)
    # Vertices 0..19 are A, 20..39 are B
    adj = [0] * n
    edges = set()
    
    # Add all edges between A and B
    for u in range(20):
        for v in range(20, 40):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add((u, v))
            
    # Add edge (0, 1)
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    edges.add((0, 1))
    
    # Helper to get max book
    def get_max_book():
        max_b = 0
        # Iterate over all edges
        # We can just iterate over all pairs and check if edge exists
        # Or maintain a list of edges
        # Since n is small, iterating over all pairs is fine
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Edge exists
                    b = (adj[u] & adj[v]).bit_count()
                    if b > max_b:
                        max_b = b
        return max_b

    current_max_book = get_max_book()
    
    # Local Search
    # We want to reduce current_max_book
    # Strategy: Random swaps
    # To make it faster, we can try to specifically target edges with high book size
    # But random swaps might explore the space well enough.
    
    # List of all possible edges for random selection
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Pre-compute list of edges for removal
    # We need to pick a random edge to remove and a random non-edge to add
    
    best_max_book = current_max_book
    best_adj = adj[:]
    best_edges = set(edges)
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    steps = 20000
    
    for step in range(steps):
        # Pick random edge to remove
        # We have ~401 edges. 
        # To pick uniformly, we can maintain a list of current edges.
        # But set is okay, we can convert to list or use random.choice on list(edges)
        # For speed, let's maintain a list of edges.
        # Actually, just picking random u, v and checking if edge exists is O(1) avg if we try a few times?
        # No, better to maintain a list of current edges.
        
        # Let's update the list of edges
        # This is getting complex. Let's just use the set and convert to list for choice.
        # 20000 steps * len(edges) conversion might be slow.
        # Let's maintain a list `cur_edges_list`.
        
        # Re-initialize list from set initially
        if 'cur_edges_list' not in locals():
            cur_edges_list = list(edges)
        
        # Pick random edge to remove
        idx_rem = random.randint(0, len(cur_edges_list) - 1)
        u_rem, v_rem = cur_edges_list[idx_rem]
        
        # Pick random non-edge to add
        # We need to find a non-edge.
        # Try random pairs until we find one not in edges.
        # Since graph is dense (401/780 ~ 50%), this takes ~2 tries on average.
        while True:
            u_add = random.randint(0, n - 1)
            v_add = random.randint(0, n - 1)
            if u_add == v_add:
                continue
            if u_add > v_add:
                u_add, v_add = v_add, u_add
            if (u_add, v_add) not in edges:
                break
        
        # Perform swap
        # Remove (u_rem, v_rem)
        adj[u_rem] &= ~(1 << v_rem)
        adj[v_rem] &= ~(1 << u_rem)
        edges.remove((u_rem, v_rem))
        cur_edges_list.pop(idx_rem)
        
        # Add (u_add, v_add)
        adj[u_add] |= (1 << v_add)
        adj[v_add] |= (1 << u_add)
        edges.add((u_add, v_add))
        cur_edges_list.append((u_add, v_add))
        
        # Calculate new max book
        new_max_book = get_max_book()
        
        # Acceptance criterion
        if new_max_book <= current_max_book:
            # Accept
            current_max_book = new_max_book
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_adj = adj[:]
                best_edges = set(edges)
        else:
            # Reject with probability
            # Simple hill climbing first? 
            # If we only accept improvements, we might get stuck.
            # Let's allow some worsening moves if T is high.
            delta = new_max_book - current_max_book
            # Since book sizes are integers, delta >= 1
            if random.random() < np.exp(-delta / T):
                current_max_book = new_max_book
            else:
                # Revert
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)
                edges.add((u_rem, v_rem))
                cur_edges_list.insert(idx_rem, (u_rem, v_rem))
                
                adj[u_add] &= ~(1 << v_add)
                adj[v_add] &= ~(1 << u_add)
                edges.remove((u_add, v_add))
                cur_edges_list.pop()
        
        # Cool down
        T *= cooling_rate
        
        # Early stop if we reach the theoretical lower bound
        if best_max_book == 7:
            break
    
    # Reconstruct edges from best_adj
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                final_edges.append([u, v])
                
    return final_edges
