import numpy as np
import time
import random
import sys

def _has_c6_bitmask(adj_bits, n):
    """
    Check if the graph represented by adj_bits (list of int) contains a C6.
    adj_bits[i] is a bitmask of neighbors of i.
    """
    # A C6 is v0-v1-v2-v3-v4-v5-v0.
    # To avoid duplicates, we can fix v0 and v1 such that v1 > v0.
    # Then we need a path v1-v2-v3-v4-v5-v0 of length 5, with all vertices distinct.
    
    for v0 in range(n):
        # Iterate neighbors v1 of v0 with v1 > v0
        mask_v1 = adj_bits[v0] & ((1 << (n - v0 - 1)) - 1) << (v0 + 1)
        while mask_v1:
            lsb_v1 = mask_v1 & -mask_v1
            v1 = lsb_v1.bit_length() - 1
            mask_v1 -= lsb_v1
            
            # We need a path v1 -> v2 -> v3 -> v4 -> v5 -> v0
            # v2 is neighbor of v1, not v0, not v1
            mask_v2 = adj_bits[v1] & ~(1 << v0) & ~(1 << v1)
            while mask_v2:
                lsb_v2 = mask_v2 & -mask_v2
                v2 = lsb_v2.bit_length() - 1
                mask_v2 -= lsb_v2
                
                # v3 is neighbor of v2, not v0, v1, v2
                mask_v3 = adj_bits[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                while mask_v3:
                    lsb_v3 = mask_v3 & -mask_v3
                    v3 = lsb_v3.bit_length() - 1
                    mask_v3 -= lsb_v3
                    
                    # v4 is neighbor of v3, not v0, v1, v2, v3
                    mask_v4 = adj_bits[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                    while mask_v4:
                        lsb_v4 = mask_v4 & -mask_v4
                        v4 = lsb_v4.bit_length() - 1
                        mask_v4 -= lsb_v4
                        
                        # v5 is neighbor of v4, not v0, v1, v2, v3, v4
                        # And v5 must be neighbor of v0
                        mask_v5 = adj_bits[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3) & ~(1 << v4)
                        mask_v5 &= adj_bits[v0]
                        
                        if mask_v5:
                            return True
    return False

def _build_adj_bits_from_colors(colors, n):
    """
    Build adjacency bitmasks for each color class.
    colors is a list of length n*(n-1)//2.
    """
    adj_bits = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj_bits[c][i] |= (1 << j)
            adj_bits[c][j] |= (1 << i)
            idx += 1
    return adj_bits

def _is_valid_coloring(colors, n):
    """Check if the edge coloring has no C6 in any color class."""
    adj_bits = _build_adj_bits_from_colors(colors, n)
    for c in range(4):
        if _has_c6_bitmask(adj_bits[c], n):
            return False
    return True

def _spectral_coloring(n, seed):
    """
    Generate edge colors using spectral embedding.
    """
    rng = np.random.RandomState(seed)
    
    # Generate a random regular graph
    d = min(10, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    # Use a simple method to generate a d-regular graph
    # Walecki construction or random matching
    adj = np.zeros((n, n), dtype=int)
    for _ in range(d // 2):
        perm = rng.permutation(n)
        for i in range(n // 2):
            u = perm[2*i]
            v = perm[2*i+1]
            if u != v and adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
    
    np.fill_diagonal(adj, 0)
    
    # Compute Laplacian
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    
    if n <= 4:
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
        
    # Take the 4 smallest non-zero eigenvalues
    vecs = eigvecs[:, 1:5]
    points = vecs.T # Shape (4, n)
    
    # Random rotation to break symmetry
    R = rng.randn(4, 4)
    R, _ = np.linalg.qr(R)
    
    rotated_points = R @ points
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = rotated_points[:, j] - rotated_points[:, i]
            signs = np.sign(diff)
            signs[signs == 0] = 1
            
            b0 = 1 if signs[0] > 0 else 0
            b1 = 1 if signs[1] > 0 else 0
            b2 = 1 if signs[2] > 0 else 0
            b3 = 1 if signs[3] > 0 else 0
            
            idx = b0*8 + b1*4 + b2*2 + b3
            color = idx % 4
            edge_colors.append(color)
            
    return edge_colors

def _repair_coloring(colors, n, max_steps=5000):
    """
    Local search to remove C6s using bitmask acceleration.
    """
    colors = list(colors)
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    # Precompute adjacency bitmasks
    adj_bits = _build_adj_bits_from_colors(colors, n)
    
    def find_c6(c):
        """Returns a C6 as a tuple of 6 vertices if found in color c, else None."""
        # Reuse logic from _has_c6_bitmask but return the cycle
        for v0 in range(n):
            mask_v1 = adj_bits[c][v0] & ((1 << (n - v0 - 1)) - 1) << (v0 + 1)
            while mask_v1:
                lsb_v1 = mask_v1 & -mask_v1
                v1 = lsb_v1.bit_length() - 1
                mask_v1 -= lsb_v1
                
                mask_v2 = adj_bits[c][v1] & ~(1 << v0) & ~(1 << v1)
                while mask_v2:
                    lsb_v2 = mask_v2 & -mask_v2
                    v2 = lsb_v2.bit_length() - 1
                    mask_v2 -= lsb_v2
                    
                    mask_v3 = adj_bits[c][v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v3:
                        lsb_v3 = mask_v3 & -mask_v3
                        v3 = lsb_v3.bit_length() - 1
                        mask_v3 -= lsb_v3
                        
                        mask_v4 = adj_bits[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                        while mask_v4:
                            lsb_v4 = mask_v4 & -mask_v4
                            v4 = lsb_v4.bit_length() - 1
                            mask_v4 -= lsb_v4
                            
                            mask_v5 = adj_bits[c][v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3) & ~(1 << v4)
                            mask_v5 &= adj_bits[c][v0]
                            
                            if mask_v5:
                                lsb_v5 = mask_v5 & -mask_v5
                                v5 = lsb_v5.bit_length() - 1
                                return (v0, v1, v2, v3, v4, v5)
        return None

    for step in range(max_steps):
        c6_found = None
        c_found = -1
        for c in range(4):
            c6 = find_c6(c)
            if c6:
                c6_found = c6
                c_found = c
                break
        
        if not c6_found:
            return colors # Valid!
            
        # We have a C6 in color c_found: v0-v1-v2-v3-v4-v5-v0
        v0, v1, v2, v3, v4, v5 = c6_found
        
        edges_in_cycle = [
            (v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)
        ]
        
        # Shuffle edges to randomize
        random.shuffle(edges_in_cycle)
        
        move_made = False
        for u, v in edges_in_cycle:
            idx = get_edge_idx(u, v)
            old_color = c_found
            # Try other colors
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                # Tentatively change
                colors[idx] = new_color
                adj_bits[old_color][u] &= ~(1 << v)
                adj_bits[old_color][v] &= ~(1 << u)
                adj_bits[new_color][u] |= (1 << v)
                adj_bits[new_color][v] |= (1 << u)
                
                # Check if this creates a C6 in new_color
                # We only need to check if the new edge (u,v) is part of a C6 in new_color
                # This is a local check: path of length 5 from v to u in new_color \ {(u,v)}
                # But since we just added (u,v), we check if there is a path v-...-u of length 5 in the graph BEFORE adding (u,v)?
                # No, the C6 would be u-v-...-u.
                # The path is v-w1-w2-w3-w4-u.
                # w1..w4 must be distinct from u,v.
                
                # Efficient check:
                # We need to know if there is a path of length 5 between u and v in adj_bits[new_color] WITHOUT using the edge (u,v).
                # But we just added (u,v) to adj_bits[new_color].
                # So we should check the graph *before* adding (u,v) for a path of length 5?
                # If there is a path of length 5 between u and v in G_new_color \ {(u,v)}, then adding (u,v) creates a C6.
                
                # Let's revert for a moment to check? No, we can just check the current state.
                # If we check for a C6 in the *entire* new_color graph, it's slow.
                # Let's check if (u,v) is in a C6.
                # A C6 containing (u,v) is u-v-w1-w2-w3-w4-u.
                # So we need a path v-w1-w2-w3-w4-u of length 5 in the graph *without* the edge (u,v).
                
                # We can temporarily remove (u,v) from adj_bits[new_color] for the check?
                # Or just check if there is a path of length 5 between u and v in adj_bits[new_color] that doesn't use the direct edge.
                # Since n is small, we can just run a BFS/DFS for path length 5.
                
                def has_path_len_5(adj, start, end):
                    # adj is list of bitmasks
                    # We want path start -> w1 -> w2 -> w3 -> w4 -> end
                    # w1, w2, w3, w4 distinct and not start/end
                    # This is equivalent to checking if end is in the 5-neighborhood of start, avoiding immediate backtracking?
                    # Actually, simple BFS with depth 5.
                    
                    # To be fast, we can use the bitmask trick.
                    # But depth 5 is small.
                    
                    visited = {start}
                    frontier = [start]
                    
                    for _ in range(5):
                        next_frontier = []
                        for node in frontier:
                            mask = adj[node]
                            # Exclude visited? No, we need to ensure simple path.
                            # For small n, we can just track visited sets in recursion.
                            # But iterative BFS with state (node, visited_mask) is expensive.
                            # Given n<=20, we can just do a recursive DFS with pruning.
                        pass
                    return False # Placeholder
                
                # Simpler: Just check if the *entire* new_color graph has a C6.
                # Since we only changed one edge, the only new C6s must involve this edge.
                # So we check if (u,v) is in a C6 in new_color.
                
                def is_edge_in_c6(adj, u, v):
                    # Check if there is a path of length 5 from v to u in adj, not using edge (u,v) directly as part of the path?
                    # The path is v -> ... -> u.
                    # The edge (u,v) is the closing edge.
                    # So we need a path of length 5 from v to u in the graph *excluding* the edge (u,v).
                    
                    # Temporarily remove (u,v)
                    adj[v] &= ~(1 << u)
                    adj[u] &= ~(1 << v)
                    
                    # Check for path v-w1-w2-w3-w4-u
                    # w1 in adj[v]
                    mask_w1 = adj[v]
                    while mask_w1:
                        lsb_w1 = mask_w1 & -mask_w1
                        w1 = lsb_w1.bit_length() - 1
                        mask_w1 -= lsb_w1
                        
                        if w1 == u: continue # Path length 1, not 5
                        
                        mask_w2 = adj[w1] & ~(1 << v)
                        while mask_w2:
                            lsb_w2 = mask_w2 & -mask_w2
                            w2 = lsb_w2.bit_length() - 1
                            mask_w2 -= lsb_w2
                            
                            if w2 == v or w2 == u: continue
                            
                            mask_w3 = adj[w2] & ~(1 << v) & ~(1 << w1)
                            while mask_w3:
                                lsb_w3 = mask_w3 & -mask_w3
                                w3 = lsb_w3.bit_length() - 1
                                mask_w3 -= lsb_w3
                                
                                if w3 == v or w3 == u or w3 == w1: continue
                                
                                mask_w4 = adj[w3] & ~(1 << v) & ~(1 << w1) & ~(1 << w2)
                                while mask_w4:
                                    lsb_w4 = mask_w4 & -mask_w4
                                    w4 = lsb_w4.bit_length() - 1
                                    mask_w4 -= lsb_w4
                                    
                                    if w4 == v or w4 == u or w4 == w1 or w4 == w2: continue
                                    
                                    if u in adj[w4]:
                                        return True
                    # Restore (u,v)
                    adj[v] |= (1 << u)
                    adj[u] |= (1 << v)
                    return False

                if is_edge_in_c6(adj_bits[new_color], u, v):
                    # Revert
                    colors[idx] = old_color
                    adj_bits[old_color][u] |= (1 << v)
                    adj_bits[old_color][v] |= (1 << u)
                    adj_bits[new_color][u] &= ~(1 << v)
                    adj_bits[new_color][v] &= ~(1 << u)
                else:
                    # Accept
                    move_made = True
                    break
                    
            if move_made:
                break
                
        if not move_made:
            # Stuck, break to try a new seed
            break

    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    # Try n=18, 17, ...
    # 120s limit.
    # Let's allocate time wisely.
    
    for n in range(18, 9, -1):
        found = False
        start_time = time.time()
        # Try multiple seeds
        for seed in range(100):
            if time.time() - start_time > 10: # Limit per n to 10s? No, total 120s.
                break
            # Generate spectral coloring
            colors = _spectral_coloring(n, seed)
            
            # Repair
            colors = _repair_coloring(colors, n, max_steps=1000)
            
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback
    for n in range(9, 1, -1):
        for seed in range(20):
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=500)
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
