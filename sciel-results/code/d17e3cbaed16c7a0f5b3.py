import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges_total = len(all_edges)
    
    # Edge index mapping
    edge_idx = {}
    for idx, (i, j) in enumerate(all_edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute for each edge, which triples it belongs to
    # A K3,3 violation involves 3 vertices {i,j,k} and 3 common neighbors
    # For each pair (i,j), let C_ij = N(i) ∩ N(j)
    # If |C_ij| >= 3, then for any k in C_ij, we check if |N(k) ∩ C_ij| >= 3
    
    def count_edges(masks):
        return sum(bin(m).count('1') for m in masks) // 2
    
    def get_violation(masks):
        """Find one K3,3 violation or return None."""
        # For each pair (i,j), compute common neighbors
        for i in range(n):
            for j in range(i+1, n):
                common_ij = masks[i] & masks[j]
                c_ij_count = bin(common_ij).count('1')
                if c_ij_count < 3:
                    continue
                # For each k > j, check if N(k) shares >= 3 with C_ij
                for k in range(j+1, n):
                    common_ijk = common_ij & masks[k]
                    if bin(common_ijk).count('1') >= 3:
                        return (i, j, k, common_ijk)
        return None
    
    def is_valid(masks):
        return get_violation(masks) is None
    
    def try_add_edge(masks, u, v):
        """Try to add edge (u,v), return True if valid."""
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        if is_valid(masks):
            return True
        else:
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            return False
    
    def try_remove_edge(masks, u, v):
        """Remove edge (u,v)."""
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
    
    # Simulated annealing
    best_edges = 0
    best_masks = [0] * n
    
    # Start with a dense random graph
    initial_p = 0.85
    for _ in range(50):  # Multiple restarts
        if time.time() > deadline - 10:
            break
        
        # Initialize
        masks = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < initial_p:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        
        # Repair: remove edges until valid
        max_repair = 200
        for _ in range(max_repair):
            if time.time() > deadline - 5:
                break
            viol = get_violation(masks)
            if viol is None:
                break
            i, j, k, common_mask = viol
            # Get 3 common neighbors
            neighbors = []
            m = common_mask
            while len(neighbors) < 3 and m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                neighbors.append(idx)
                m ^= lsb
            
            # Remove a random edge from the K3,3
            u = random.choice([i, j, k])
            v = random.choice(neighbors)
            try_remove_edge(masks, u, v)
        
        if not is_valid(masks):
            continue
        
        current_edges = count_edges(masks)
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
        
        # Simulated annealing phase
        T = 2.0
        T_min = 0.01
        cooling_rate = 0.9995
        max_sa_steps = 50000
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            # Random move: either add an edge, remove an edge, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random missing edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not missing:
                    break
                u, v = random.choice(missing)
                old_state = (masks[u], masks[v])
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
                if is_valid(masks):
                    new_edges = count_edges(masks)
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        masks[u] = old_state[0]
                        masks[v] = old_state[1]
                else:
                    masks[u] = old_state[0]
                    masks[v] = old_state[1]
                    
            elif move_type < 0.6:
                # Remove a random existing edge
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                if not existing:
                    break
                u, v = random.choice(existing)
                old_state = (masks[u], masks[v])
                try_remove_edge(masks, u, v)
                
                new_edges = count_edges(masks)
                delta = new_edges - current_edges
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_masks = masks[:]
                else:
                    masks[u] = old_state[0]
                    masks[v] = old_state[1]
                    
            else:
                # Swap: remove one edge, add another
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not existing or not missing:
                    break
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                    
                old_state = (masks[u1], masks[v1], masks[u2], masks[v2])
                try_remove_edge(masks, u1, v1)
                masks[u2] |= (1 << v2)
                masks[v2] |= (1 << u2)
                
                if is_valid(masks):
                    new_edges = count_edges(masks)
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        masks[u1] = old_state[0]
                        masks[v1] = old_state[1]
                        masks[u2] = old_state[2]
                        masks[v2] = old_state[3]
                else:
                    masks[u1] = old_state[0]
                    masks[v1] = old_state[1]
                    masks[u2] = old_state[2]
                    masks[v2] = old_state[3]
            
            T *= cooling_rate
        
        # Local search: try to add all possible edges
        for i in range(n):
            for j in range(i+1, n):
                if time.time() > deadline - 3:
                    break
                if not (masks[i] & (1 << j)):
                    if try_add_edge(masks, i, j):
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
    
    # Convert best_masks to edge list
    edges = []
    for i in range(n):
        m = best_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
            
    return edges
