import time
import random
import math

DEADLINE = 140  # Safety margin for 150s limit

def _compute_score(n, adj_masks):
    """
    Computes the maximum degree sum over all triangles in the graph.
    n: number of vertices
    adj_masks: list of int bitmasks representing adjacency
    """
    deg = [bin(mask).count('1') for mask in adj_masks]
    max_sum = 0

    # Iterate over all triplets i < j < k
    for i in range(n):
        mi = adj_masks[i]
        for j in range(i + 1, n):
            if (mi >> j) & 1:
                # Find common neighbors k > j
                common = mi & adj_masks[j] & ~((1 << (j + 1)) - 1)
                mask_k = common
                while mask_k:
                    lsb_k = mask_k & -mask_k
                    k = lsb_k.bit_length() - 1
                    mask_k ^= lsb_k
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _find_worst_triangle(n, adj_masks, deg):
    """Find the triangle with maximum degree sum. Returns (score, (i,j,k)) or (0, None)."""
    worst_sum = 0
    worst_tri = None
    for i in range(n):
        mi = adj_masks[i]
        for j in range(i + 1, n):
            if (mi >> j) & 1:
                common = mi & adj_masks[j] & ~((1 << (j + 1)) - 1)
                mask_k = common
                while mask_k:
                    lsb_k = mask_k & -mask_k
                    k = lsb_k.bit_length() - 1
                    mask_k ^= lsb_k
                    s = deg[i] + deg[j] + deg[k]
                    if s > worst_sum:
                        worst_sum = s
                        worst_tri = (i, j, k)
    return worst_sum, worst_tri

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
    
    best_score = float('inf')
    best_adj_masks = None
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Multi-start strategy
    while time.time() - start_time < DEADLINE:
        # Initialize a random graph with target_edges
        random.shuffle(all_edges)
        
        adj_masks = [0] * n
        current_edge_set = set()
        
        for u, v in all_edges[:target_edges]:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            current_edge_set.add((u, v))
            
        non_edge_set = set(all_edges[target_edges:])
        
        deg = [bin(mask).count('1') for mask in adj_masks]
        current_score = _compute_score(n, adj_masks)
        
        if current_score < best_score:
            best_score = current_score
            best_adj_masks = list(adj_masks)
            
        # Local Search with Simulated Annealing
        temp = 5.0
        cool_rate = 0.995
        min_temp = 0.1
        
        stuck_count = 0
        max_stuck = 300
        
        while time.time() - start_time < DEADLINE:
            # Check for worst triangle
            worst_sum, worst_tri = _find_worst_triangle(n, adj_masks, deg)
            
            if worst_tri is None:
                break
                
            i, j, k = worst_tri
            tri_edges = [(i, j), (j, k), (i, k)]
            random.shuffle(tri_edges)
            
            move_found = False
            
            # Try a few random non-edges for the first few removal candidates
            for rem_u, rem_v in tri_edges[:2]: # Only try 2 removals per iteration to save time
                
                if not non_edge_set:
                    continue
                    
                # Pick a random non-edge
                add_u, add_v = random.choice(list(non_edge_set))
                
                # Check validity (should be fine by construction)
                
                # Apply move
                adj_masks[rem_u] ^= (1 << rem_v)
                adj_masks[rem_v] ^= (1 << rem_u)
                adj_masks[add_u] ^= (1 << add_v)
                adj_masks[add_v] ^= (1 << add_u)
                
                deg[rem_u] -= 1
                deg[rem_v] -= 1
                deg[add_u] += 1
                deg[add_v] += 1
                
                rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                add_edge = (min(add_u, add_v), max(add_u, add_v))
                
                current_edge_set.discard(rem_edge)
                non_edge_set.add(rem_edge)
                non_edge_set.discard(add_edge)
                current_edge_set.add(add_edge)
                
                new_score = _compute_score(n, adj_masks)
                
                delta = new_score - current_score
                
                if delta < 0 or (temp > min_temp and random.random() < math.exp(-delta / temp)):
                    current_score = new_score
                    stuck_count = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_adj_masks = list(adj_masks)
                    move_found = True
                else:
                    # Revert
                    adj_masks[rem_u] ^= (1 << rem_v)
                    adj_masks[rem_v] ^= (1 << rem_u)
                    adj_masks[add_u] ^= (1 << add_v)
                    adj_masks[add_v] ^= (1 << add_u)
                    
                    deg[rem_u] += 1
                    deg[rem_v] += 1
                    deg[add_u] -= 1
                    deg[add_v] -= 1
                    
                    current_edge_set.add(rem_edge)
                    non_edge_set.discard(rem_edge)
                    non_edge_set.add(add_edge)
                    current_edge_set.discard(add_edge)
                    
                    stuck_count += 1
                
                if move_found:
                    break
                
            if not move_found:
                stuck_count += 5
                
            temp *= cool_rate
            
            if stuck_count > max_stuck:
                # Restart this local search
                break
                
    if best_adj_masks is None:
        # Fallback
        random.shuffle(all_edges)
        edges = all_edges[:target_edges]
        return {'n': n, 'edges': edges}
        
    best_edges = _construct_edges_from_masks(best_adj_masks, n)
    return {'n': n, 'edges': best_edges}
