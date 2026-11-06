import numpy as np
import time
import random
import sys

def _has_c6_bitmask(adj_masks, n):
    """
    Check if any color class contains a C6 using bitmasks.
    adj_masks[c] is a list of length n, where adj_masks[c][v] is an integer bitmask of neighbors.
    Returns True if C6 found.
    """
    for c in range(4):
        adj = adj_masks[c]
        # Iterate over edges (u, v) with u < v
        for u in range(n):
            mask_u = adj[u]
            # Iterate over neighbors v > u
            v_mask = mask_u & ~((1 << (u + 1)) - 1) # neighbors > u
            while v_mask:
                v = (v_mask & -v_mask).bit_length() - 1
                v_mask &= v_mask - 1
                
                # We have edge (u, v).
                # Look for path u-v-w1-w2-w3-w4-u
                # w1 in adj[v] \ {u}
                # w2 in adj[w1] \ {u, v}
                # ...
                
                # This is still O(n^6) in worst case, but bitmasks make inner loops fast.
                # Let's try to find a path of length 4 from v to u.
                # Path: v - a - b - c - d - u
                # a in adj[v] \ {u}
                # b in adj[a] \ {u, v}
                # c in adj[b] \ {u, v, a}
                # d in adj[c] \ {u, v, a, b}
                # check if u in adj[d]
                
                # Optimization: Precompute common neighbors?
                # For n=20, simple iteration is fine if we prune.
                
                mask_a = adj[v] & ~(1 << u)
                while mask_a:
                    a = (mask_a & -mask_a).bit_length() - 1
                    mask_a &= mask_a - 1
                    
                    mask_b = adj[a] & ~(1 << u) & ~(1 << v)
                    while mask_b:
                        b = (mask_b & -mask_b).bit_length() - 1
                        mask_b &= mask_b - 1
                        
                        mask_c = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                        while mask_c:
                            c = (mask_c & -mask_c).bit_length() - 1
                            mask_c &= mask_c - 1
                            
                            mask_d = adj[c] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                            while mask_d:
                                d = (mask_d & -mask_d).bit_length() - 1
                                mask_d &= mask_d - 1
                                
                                if adj[d] & (1 << u):
                                    return True
    return False

def _build_bitmasks(colors, n):
    """Build bitmask adjacency lists for each color."""
    adj_masks = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj_masks[c][i] |= (1 << j)
            adj_masks[c][j] |= (1 << i)
            idx += 1
    return adj_masks

def _spectral_coloring_v2(n, seed):
    """
    Improved spectral coloring.
    1. Generate random regular graph.
    2. Get top 4 non-trivial Laplacian eigenvectors.
    3. Assign vertices to 16 types based on signs.
    4. Use a fixed, optimized mapping from type-pairs to colors.
    """
    rng = np.random.RandomState(seed)
    
    # Generate random regular graph
    d = min(10, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    # Simple random regular graph generation
    adj_mat = np.zeros((n, n), dtype=int)
    for _ in range(d // 2):
        perm = rng.permutation(n)
        for i in range(n // 2):
            u, v = perm[2*i], perm[2*i+1]
            if u != v and adj_mat[u, v] == 0:
                adj_mat[u, v] = 1
                adj_mat[v, u] = 1
    
    # Laplacian
    degree = adj_mat.sum(axis=1)
    L = np.diag(degree) - adj_mat
    
    # Eigenvectors
    eigvals, eigvecs = np.linalg.eigh(L)
    # Take indices 1 to 4
    vecs = eigvecs[:, 1:5]
    
    # Signs
    signs = np.sign(vecs)
    signs[signs == 0] = 1
    
    # Type for each vertex: 4 bits
    types = np.zeros(n, dtype=int)
    for i in range(n):
        t = 0
        for k in range(4):
            if signs[k, i] > 0:
                t |= (1 << k)
        types[i] = t
        
    # Map types to colors.
    # We have 16 types. Pairs (t1, t2) with t1 <= t2.
    # We need to assign colors to these pairs.
    # A good heuristic is to use the XOR of types or distance in Hamming space.
    # Let's try: color = (t1 ^ t2) % 4.
    # This ensures that vertices with similar types (small Hamming distance) get low color values?
    # Or maybe color = (t1 + t2) % 4?
    
    # Let's try a few mappings and pick the one with fewest C6s?
    # For now, use a random rotation of the type space to break symmetry.
    # Or just use (t1 ^ t2) % 4.
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            t1 = types[i]
            t2 = types[j]
            # Use XOR
            color = (t1 ^ t2) % 4
            colors.append(color)
            
    return colors

def _repair_coloring_v2(colors, n, time_limit=10.0):
    """
    Local search to remove C6s.
    """
    start_time = time.time()
    colors = list(colors)
    adj_masks = _build_bitmasks(colors, n)
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    def find_c6_cycle(c):
        """Find a C6 in color c, return list of vertices or None."""
        adj = adj_masks[c]
        for u in range(n):
            mask_u = adj[u]
            v_mask = mask_u & ~((1 << (u + 1)) - 1)
            while v_mask:
                v = (v_mask & -v_mask).bit_length() - 1
                v_mask &= v_mask - 1
                mask_a = adj[v] & ~(1 << u)
                while mask_a:
                    a = (mask_a & -mask_a).bit_length() - 1
                    mask_a &= mask_a - 1
                    mask_b = adj[a] & ~(1 << u) & ~(1 << v)
                    while mask_b:
                        b = (mask_b & -mask_b).bit_length() - 1
                        mask_b &= mask_b - 1
                        mask_c = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                        while mask_c:
                            c_v = (mask_c & -mask_c).bit_length() - 1
                            mask_c &= mask_c - 1
                            mask_d = adj[c_v] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                            while mask_d:
                                d = (mask_d & -mask_d).bit_length() - 1
                                mask_d &= mask_d - 1
                                if adj[d] & (1 << u):
                                    return [u, v, a, b, c_v, d]
        return None

    def update_adj_masks(idx, old_c, new_c, u, v):
        adj_masks[old_c][u] &= ~(1 << v)
        adj_masks[old_c][v] &= ~(1 << u)
        adj_masks[new_c][u] |= (1 << v)
        adj_masks[new_c][v] |= (1 << u)

    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        # Find a C6
        c6_found = None
        c_found = -1
        for c in range(4):
            cyc = find_c6_cycle(c)
            if cyc:
                c6_found = cyc
                c_found = c
                break
                
        if not c6_found:
            return colors
            
        # Pick a random edge from the cycle to flip
        edges_in_cycle = [
            (c6_found[0], c6_found[1]),
            (c6_found[1], c6_found[2]),
            (c6_found[2], c6_found[3]),
            (c6_found[3], c6_found[4]),
            (c6_found[4], c6_found[5]),
            (c6_found[5], c6_found[0])
        ]
        u, v = random.choice(edges_in_cycle)
        idx = get_edge_idx(u, v)
        old_color = colors[idx]
        
        # Try a new color
        new_color = random.randint(0, 3)
        if new_color == old_color:
            new_color = (new_color + 1) % 4
            
        # Check if new color creates a C6 involving (u, v)
        # We need to check if there is a path of length 4 from v to u in new_color
        # that doesn't use the edge (u, v) (which we are adding).
        # Actually, we are adding (u, v) to new_color.
        # So we check if there is a path v - ... - u of length 4 in new_color.
        
        adj_new = adj_masks[new_color]
        # Temporarily add edge
        adj_new[u] |= (1 << v)
        adj_new[v] |= (1 << u)
        
        # Check for C6 through (u, v)
        # Path u - v - w1 - w2 - w3 - w4 - u
        has_conflict = False
        mask_w1 = adj_new[v] & ~(1 << u)
        while mask_w1 and not has_conflict:
            w1 = (mask_w1 & -mask_w1).bit_length() - 1
            mask_w1 &= mask_w1 - 1
            mask_w2 = adj_new[w1] & ~(1 << u) & ~(1 << v)
            while mask_w2 and not has_conflict:
                w2 = (mask_w2 & -mask_w2).bit_length() - 1
                mask_w2 &= mask_w2 - 1
                mask_w3 = adj_new[w2] & ~(1 << u) & ~(1 << v) & ~(1 << w1)
                while mask_w3 and not has_conflict:
                    w3 = (mask_w3 & -mask_w3).bit_length() - 1
                    mask_w3 &= mask_w3 - 1
                    mask_w4 = adj_new[w3] & ~(1 << u) & ~(1 << v) & ~(1 << w1) & ~(1 << w2)
                    while mask_w4 and not has_conflict:
                        w4 = (mask_w4 & -mask_w4).bit_length() - 1
                        mask_w4 &= mask_w4 - 1
                        if adj_new[w4] & (1 << u):
                            has_conflict = True
                            
        # Revert temporary addition
        adj_new[u] &= ~(1 << v)
        adj_new[v] &= ~(1 << u)
        
        if not has_conflict:
            # Accept change
            update_adj_masks(idx, old_color, new_color, u, v)
            colors[idx] = new_color
        else:
            # Try other colors? Or just skip?
            # For now, skip.
            pass

    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        found = False
        # Try multiple seeds
        for seed in range(20):
            colors = _spectral_coloring_v2(n, seed)
            colors = _repair_coloring_v2(colors, n, time_limit=5.0)
            
            # Final verification
            adj_masks = _build_bitmasks(colors, n)
            if not _has_c6_bitmask(adj_masks, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback
    for n in range(9, 1, -1):
        for seed in range(20):
            colors = _spectral_coloring_v2(n, seed)
            colors = _repair_coloring_v2(colors, n, time_limit=5.0)
            adj_masks = _build_bitmasks(colors, n)
            if not _has_c6_bitmask(adj_masks, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
