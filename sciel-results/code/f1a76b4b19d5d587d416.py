import time
import random
import math

DEADLINE = 115

def _generate_triangles(n):
    """Generate all possible triangles in a complete graph of n vertices."""
    triangles = []
    tri_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
                tri_index[(u, v, w)] = idx
                idx += 1
    return triangles, tri_index

def _state_to_edges(adj_masks):
    n = 16
    edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges

def _compute_score(n, adj_masks):
    """
    Compute the max degree sum over all triangles using bitmask adjacency.
    """
    deg = [bin(mask).count('1') for mask in adj_masks]
    max_sum = 0
    for i in range(n):
        mi = adj_masks[i]
        # Only consider j > i
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb
            
            common = mi & adj_masks[j]
            base = deg[i] + deg[j]
            
            # Find max deg[k] in common neighbors
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
                s = base + max_deg_k
                if s > max_sum:
                    max_sum = s
    return max_sum

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    # Precompute triangles
    triangles, tri_index = _generate_triangles(n)
    num_tris = len(triangles)
    
    # Precompute for each vertex which triangles it belongs to
    vert_to_tris = {}
    for i, (u, v, w) in enumerate(triangles):
        for vert in (u, v, w):
            if vert not in vert_to_tris:
                vert_to_tris[vert] = []
            vert_to_tris[vert].append(i)

    # Initialize graph: K_{8,8} + 1 edge
    # Vertices 0-7 and 8-15
    adj_masks = [0] * n
    degs = [0] * n
    edges_list = []
    edges_set = set()
    
    for u in range(8):
        for v in range(8, 16):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
            e = (u, v)
            edges_list.append(e)
            edges_set.add(e)
            
    # Add one edge within a partition to create triangles
    # Let's add (0, 1)
    u, v = 0, 1
    adj_masks[u] |= (1 << v)
    adj_masks[v] |= (1 << u)
    degs[u] += 1
    degs[v] += 1
    e = (u, v)
    edges_list.append(e)
    edges_set.add(e)
    
    # Calculate initial scores
    tri_scores = [0] * num_tris
    for i, (u, v, w) in enumerate(triangles):
        if (adj_masks[u] >> v) & 1 and (adj_masks[u] >> w) & 1 and (adj_masks[v] >> w) & 1:
            tri_scores[i] = degs[u] + degs[v] + degs[w]
            
    current_score = max(tri_scores)
    best_score = current_score
    best_adj_masks = adj_masks[:]
    best_degs = degs[:]
    
    # Continuous guidance variables (Max-Cut relaxation idea)
    # x[i] represents the "cut potential" of vertex i.
    # We want to minimize max triangle degree sum.
    # High degree vertices contribute more to the sum.
    # We can use a potential function that penalizes high degrees.
    # Let's maintain a "stress" for each vertex proportional to its degree.
    # The "gradient" of the objective w.r.t edge (u,v) removal/addition
    # depends on the degrees of neighbors.
    
    # We will use the current degrees to guide the search.
    # Specifically, we will bias the selection of edges to remove
    # towards those incident to high-degree vertices, and edges to add
    # towards those incident to low-degree vertices, but also considering
    # the impact on triangle scores.
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    # To make the search more effective, we will periodically re-evaluate
    # the "potential" landscape and restart if stuck.
    
    while time.time() - start_time < DEADLINE:
        if not edges_list:
            break
            
        # Select edge to remove
        # Bias towards edges in high-scoring triangles or high-degree vertices
        # We sample a random edge, but with some probability we pick one
        # from a high-degree vertex.
        
        # Calculate a weight for each edge based on degree sum of endpoints
        # and the max triangle score involving that edge.
        
        # For speed, we just use random selection but with a twist:
        # If we have a "stagnation" counter, we switch to a more aggressive
        # strategy.
        
        rem_e = random.choice(edges_list)
        rem_u, rem_v = rem_e
        
        # Select edge to add
        # Try to find a non-edge that reduces the max score.
        # Heuristic: prefer adding edges between low-degree vertices.
        add_e = None
        for _ in range(10):
            cand = random.choice(all_pairs)
            if cand not in edges_set:
                add_e = cand
                break
        if not add_e:
            continue
            
        add_u, add_v = add_e
        
        # Apply swap
        adj_masks[rem_u] &= ~(1 << rem_v)
        adj_masks[rem_v] &= ~(1 << rem_u)
        degs[rem_u] -= 1
        degs[rem_v] -= 1
        edges_list.remove(rem_e)
        edges_set.remove(rem_e)
        
        adj_masks[add_u] |= (1 << add_v)
        adj_masks[add_v] |= (1 << add_u)
        degs[add_u] += 1
        degs[add_v] += 1
        edges_list.append(add_e)
        edges_set.add(add_e)
        
        # Update tri_scores
        changed_verts = {rem_u, rem_v, add_u, add_v}
        tris_to_update = set()
        for v in changed_verts:
            if v in vert_to_tris:
                tris_to_update.update(vert_to_tris[v])
                
        for t_idx in tris_to_update:
            u, v, w = triangles[t_idx]
            if (adj_masks[u] >> v) & 1 and (adj_masks[u] >> w) & 1 and (adj_masks[v] >> w) & 1:
                tri_scores[t_idx] = degs[u] + degs[v] + degs[w]
            else:
                tri_scores[t_idx] = 0
                
        new_score = max(tri_scores)
        
        # Acceptance
        if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj_masks = adj_masks[:]
                best_degs = degs[:]
                # Reset temperature on improvement
                T = 1.0 
        else:
            # Revert
            adj_masks[add_u] &= ~(1 << add_v)
            adj_masks[add_v] &= ~(1 << add_u)
            degs[add_u] -= 1
            degs[add_v] -= 1
            edges_list.remove(add_e)
            edges_set.remove(add_e)
            
            adj_masks[rem_u] |= (1 << rem_v)
            adj_masks[rem_v] |= (1 << rem_u)
            degs[rem_u] += 1
            degs[rem_v] += 1
            edges_list.append(rem_e)
            edges_set.add(rem_e)
            
            for t_idx in tris_to_update:
                u, v, w = triangles[t_idx]
                if (adj_masks[u] >> v) & 1 and (adj_masks[u] >> w) & 1 and (adj_masks[v] >> w) & 1:
                    tri_scores[t_idx] = degs[u] + degs[v] + degs[w]
                else:
                    tri_scores[t_idx] = 0

        T *= cooling
        if T < T_min:
            T = 1.0 # Restart annealing

    # Final check: ensure best_adj_masks is valid
    best_edges = _state_to_edges(best_adj_masks)
    return {'n': n, 'edges': best_edges}
