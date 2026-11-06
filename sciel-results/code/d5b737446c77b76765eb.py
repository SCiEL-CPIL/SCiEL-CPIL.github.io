import time
import random

DEADLINE = 115  # Safety margin for 150s limit

def _compute_score(n, adj_masks):
    """
    Computes the maximum degree sum over all triangles in the graph.
    n: number of vertices
    adj_masks: list of int bitmasks representing adjacency
    """
    deg = [bin(mask).count('1') for mask in adj_masks]
    max_sum = 0
    
    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb
            
            common = mi & adj_masks[j]
            
            max_deg_k = -1
            mask_k = common
            while mask_k:
                lsb_k = mask_k & -mask_k
                k = lsb_k.bit_length() - 1
                d = deg[k]
                if d > max_deg_k:
                    max_deg_k = d
                mask_k ^= lsb_k
            
            if max_deg_k >= 0:
                s = deg[i] + deg[j] + max_deg_k
                if s > max_sum:
                    max_sum = s
                    
    return max_sum

def _find_worst_triangle(n, adj_masks):
    """Find the triangle with maximum degree sum."""
    deg = [bin(mask).count('1') for mask in adj_masks]
    worst_sum = 0
    worst_triangle = None
    
    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb
            
            common = mi & adj_masks[j]
            mask_k = common
            while mask_k:
                lsb_k = mask_k & -mask_k
                k = lsb_k.bit_length() - 1
                mask_k ^= lsb_k
                s = deg[i] + deg[j] + deg[k]
                if s > worst_sum:
                    worst_sum = s
                    worst_triangle = (i, j, k)
                    
    return worst_sum, worst_triangle

def _construct_edges_from_masks(adj_masks, n):
    edges = []
    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            edges.append([i, j])
            mask_j_greater ^= lsb
    return edges

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    # Initial state: K_{8,8} + one internal edge (0,1)
    adj_masks = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
    adj_masks[0] |= (1 << 1)
    adj_masks[1] |= (1 << 0)
    
    current_score = _compute_score(n, adj_masks)
    best_score = current_score
    best_adj_masks = list(adj_masks)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Maintain edge lists for O(1) random access
    current_edge_list = []
    non_edge_list = []
    current_edge_set = set()
    non_edge_set = set()
    
    for e in all_edges:
        u, v = e
        if (adj_masks[u] >> v) & 1:
            current_edge_list.append(e)
            current_edge_set.add(e)
        else:
            non_edge_list.append(e)
            non_edge_set.add(e)
            
    random.seed(42)
    
    max_iterations = 100000
    iterations = 0
    stuck_count = 0
    
    while time.time() - start_time < DEADLINE and iterations < max_iterations:
        iterations += 1
        
        # Check time more frequently
        if time.time() - start_time > DEADLINE - 5:
            break
        
        improved = False
        
        # Find worst triangle(s)
        worst_sum, worst_triangle = _find_worst_triangle(n, adj_masks)
        
        if worst_triangle:
            i, j, k = worst_triangle
            # Try removing each edge of the worst triangle
            edges_to_try = [(i, j), (j, k), (i, k)]
            random.shuffle(edges_to_try)
            
            for rem_u, rem_v in edges_to_try:
                rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                if rem_edge not in current_edge_set:
                    continue
                    
                # Remove edge
                adj_masks[rem_u] ^= (1 << rem_v)
                adj_masks[rem_v] ^= (1 << rem_u)
                current_edge_set.discard(rem_edge)
                non_edge_set.add(rem_edge)
                
                # Try adding random non-edges
                attempts = 0
                found_improvement = False
                non_edge_sample = non_edge_list[:]
                random.shuffle(non_edge_sample)
                
                for add_u, add_v in non_edge_sample[:15]:
                    add_edge = (min(add_u, add_v), max(add_u, add_v))
                    if add_edge in current_edge_set:
                        continue
                    
                    # Add edge
                    adj_masks[add_u] ^= (1 << add_v)
                    adj_masks[add_v] ^= (1 << add_u)
                    non_edge_set.discard(add_edge)
                    current_edge_set.add(add_edge)
                    
                    new_score = _compute_score(n, adj_masks)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        found_improvement = True
                        if new_score < best_score:
                            best_score = new_score
                            best_adj_masks = list(adj_masks)
                        break
                    else:
                        # Revert add
                        adj_masks[add_u] ^= (1 << add_v)
                        adj_masks[add_v] ^= (1 << add_u)
                        current_edge_set.discard(add_edge)
                        non_edge_set.add(add_edge)
                
                if found_improvement:
                    break
                else:
                    # Revert removal
                    adj_masks[rem_u] ^= (1 << rem_v)
                    adj_masks[rem_v] ^= (1 << rem_u)
                    non_edge_set.discard(rem_edge)
                    current_edge_set.add(rem_edge)
                    
        if not improved:
            # Try a random swap
            if non_edge_list and current_edge_list:
                rem_u, rem_v = random.choice(current_edge_list)
                add_u, add_v = random.choice(non_edge_list)
                
                rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                add_edge = (min(add_u, add_v), max(add_u, add_v))
                
                # Perform swap
                adj_masks[rem_u] ^= (1 << rem_v)
                adj_masks[rem_v] ^= (1 << rem_u)
                current_edge_set.discard(rem_edge)
                non_edge_set.add(rem_edge)
                
                adj_masks[add_u] ^= (1 << add_v)
                adj_masks[add_v] ^= (1 << add_u)
                non_edge_set.discard(add_edge)
                current_edge_set.add(add_edge)
                
                new_score = _compute_score(n, adj_masks)
                
                # Accept if better, or with small probability if equal (to explore)
                if new_score < current_score or (new_score == current_score and random.random() < 0.15):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj_masks = list(adj_masks)
                    stuck_count = 0
                else:
                    # Revert
                    adj_masks[rem_u] ^= (1 << rem_v)
                    adj_masks[rem_v] ^= (1 << rem_u)
                    non_edge_set.discard(rem_edge)
                    current_edge_set.add(rem_edge)
                    
                    adj_masks[add_u] ^= (1 << add_v)
                    adj_masks[add_v] ^= (1 << add_u)
                    current_edge_set.discard(add_edge)
                    non_edge_set.add(add_edge)
                    stuck_count += 1
                    
                # Perturbation if stuck
                if stuck_count > 40:
                    for _ in range(5):
                        if non_edge_list and current_edge_list:
                            r_u, r_v = random.choice(current_edge_list)
                            a_u, a_v = random.choice(non_edge_list)
                            
                            r_edge = (min(r_u, r_v), max(r_u, r_v))
                            a_edge = (min(a_u, a_v), max(a_u, a_v))
                            
                            adj_masks[r_u] ^= (1 << r_v)
                            adj_masks[r_v] ^= (1 << r_u)
                            current_edge_set.discard(r_edge)
                            non_edge_set.add(r_edge)
                            
                            adj_masks[a_u] ^= (1 << a_v)
                            adj_masks[a_v] ^= (1 << a_u)
                            non_edge_set.discard(a_edge)
                            current_edge_set.add(a_edge)
                            
                    current_score = _compute_score(n, adj_masks)
                    stuck_count = 0

    # Final polish: try to improve from best_adj_masks with more targeted moves
    polish_start = time.time()
    polish_deadline = start_time + DEADLINE - 2
    
    polish_adj = list(best_adj_masks)
    polish_score = _compute_score(n, polish_adj)
    polish_best_score = polish_score
    polish_best_adj = list(polish_adj)
    
    polish_edge_set = set()
    polish_non_edge_set = set()
    polish_edge_list = []
    polish_non_edge_list = []
    
    for e in all_edges:
        u, v = e
        if (polish_adj[u] >> v) & 1:
            polish_edge_set.add(e)
            polish_edge_list.append(e)
        else:
            polish_non_edge_set.add(e)
            polish_non_edge_list.append(e)
    
    polish_iterations = 0
    max_polish_iters = 50000
    
    while time.time() < polish_deadline and polish_iterations < max_polish_iters:
        polish_iterations += 1
        
        if time.time() > polish_deadline - 1:
            break
        
        worst_sum, worst_triangle = _find_worst_triangle(n, polish_adj)
        
        if not worst_triangle:
            break
        
        i, j, k = worst_triangle
        edges_to_try = [(i, j), (j, k), (i, k)]
        random.shuffle(edges_to_try)
        
        found_polish_improvement = False
        
        for rem_u, rem_v in edges_to_try:
            rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
            if rem_edge not in polish_edge_set:
                continue
                
            # Remove edge
            polish_adj[rem_u] ^= (1 << rem_v)
            polish_adj[rem_v] ^= (1 << rem_u)
            polish_edge_set.discard(rem_edge)
            polish_non_edge_set.add(rem_edge)
            
            # Try adding non-edges, prioritizing those that reduce degree sums
            non_edge_sample = polish_non_edge_list[:]
            random.shuffle(non_edge_sample)
            
            for add_u, add_v in non_edge_sample[:20]:
                add_edge = (min(add_u, add_v), max(add_u, add_v))
                if add_edge in polish_edge_set:
                    continue
                
                # Add edge
                polish_adj[add_u] ^= (1 << add_v)
                polish_adj[add_v] ^= (1 << add_u)
                polish_non_edge_set.discard(add_edge)
                polish_edge_set.add(add_edge)
                
                new_score = _compute_score(n, polish_adj)
                
                if new_score < polish_score:
                    polish_score = new_score
                    found_polish_improvement = True
                    if new_score < polish_best_score:
                        polish_best_score = new_score
                        polish_best_adj = list(polish_adj)
                    break
                else:
                    # Revert add
                    polish_adj[add_u] ^= (1 << add_v)
                    polish_adj[add_v] ^= (1 << add_u)
                    polish_edge_set.discard(add_edge)
                    polish_non_edge_set.add(add_edge)
            
            if found_polish_improvement:
                break
            else:
                # Revert removal
                polish_adj[rem_u] ^= (1 << rem_v)
                polish_adj[rem_v] ^= (1 << rem_u)
                polish_non_edge_set.discard(rem_edge)
                polish_edge_set.add(rem_edge)
        
        if not found_polish_improvement:
            # Try random swaps in polish phase
            if polish_non_edge_list and polish_edge_list:
                rem_u, rem_v = random.choice(polish_edge_list)
                add_u, add_v = random.choice(polish_non_edge_list)
                
                rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                add_edge = (min(add_u, add_v), max(add_u, add_v))
                
                polish_adj[rem_u] ^= (1 << rem_v)
                polish_adj[rem_v] ^= (1 << rem_u)
                polish_edge_set.discard(rem_edge)
                polish_non_edge_set.add(rem_edge)
                
                polish_adj[add_u] ^= (1 << add_v)
                polish_adj[add_v] ^= (1 << add_u)
                polish_non_edge_set.discard(add_edge)
                polish_edge_set.add(add_edge)
                
                new_score = _compute_score(n, polish_adj)
                
                if new_score < polish_score or (new_score == polish_score and random.random() < 0.1):
                    polish_score = new_score
                    if new_score < polish_best_score:
                        polish_best_score = new_score
                        polish_best_adj = list(polish_adj)
                else:
                    # Revert
                    polish_adj[rem_u] ^= (1 << rem_v)
                    polish_adj[rem_v] ^= (1 << rem_u)
                    polish_non_edge_set.discard(rem_edge)
                    polish_edge_set.add(rem_edge)
                    
                    polish_adj[add_u] ^= (1 << add_v)
                    polish_adj[add_v] ^= (1 << add_u)
                    polish_edge_set.discard(add_edge)
                    polish_non_edge_set.add(add_edge)
    
    # Use the best from either main search or polish
    if polish_best_score < best_score:
        best_score = polish_best_score
        best_adj_masks = polish_best_adj
    
    best_edges = _construct_edges_from_masks(best_adj_masks, n)
    return {'n': n, 'edges': best_edges}
