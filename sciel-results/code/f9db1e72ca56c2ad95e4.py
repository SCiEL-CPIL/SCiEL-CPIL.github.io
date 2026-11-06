import numpy as np
import time
import random

def _has_c6_bitset(adj, n):
    """
    Check if adjacency (list of int bitsets) contains a C6.
    adj[i] is a bitmask of neighbors of i.
    """
    # A C6 is a cycle v0-v1-v2-v3-v4-v5-v0
    # We iterate over edges (v0, v1) to avoid duplicate checks
    for v0 in range(n):
        # Neighbors of v0
        n0 = adj[v0]
        # Iterate over v1 in n0
        # To ensure unique edge, we can just iterate all and rely on the cycle structure?
        # No, better to iterate v1 > v0? No, bitsets don't have order easily.
        # Just iterate all neighbors.
        while n0:
            lsb = n0 & -n0
            v1 = lsb.bit_length() - 1
            n0 ^= lsb
            
            # We have edge (v0, v1).
            # Need path v1 -> v2 -> v3 -> v4 -> v5 -> v0
            # v2 in N(v1) \ {v0}
            # v5 in N(v0) \ {v1}
            
            n1 = adj[v1] & ~lsb & ~(1 << v0) # N(v1) without v1, v0
            n0_rest = adj[v0] & ~lsb & ~(1 << v1) # N(v0) without v0, v1
            
            # We need to find v2, v3, v4, v5
            # v2 in n1
            # v5 in n0_rest
            # v3 in N(v2) \ {v0, v1, v5}
            # v4 in N(v3) \ {v0, v1, v2, v5}
            # v5 in N(v4)
            
            # To speed up, we can iterate v2 and v5 simultaneously?
            # Or just nested loops.
            
            while n1:
                lsb2 = n1 & -n1
                v2 = lsb2.bit_length() - 1
                n1 ^= lsb2
                
                # v2 is chosen.
                # v5 must be in n0_rest.
                # v3 in N(v2) \ {v0, v1, v5}
                # Let's iterate v5 first?
                # Actually, let's just do the deep search.
                
                # Optimization: Precompute N(v) for all v? adj is already that.
                
                # Check if there exists v5 in n0_rest such that there is a path v2->v3->v4->v5 in G \ {v0, v1}
                # This is a path of length 3 from v2 to v5.
                
                # Let's iterate v5
                temp_n0 = n0_rest
                while temp_n0:
                    lsb5 = temp_n0 & -temp_n0
                    v5 = lsb5.bit_length() - 1
                    temp_n0 ^= lsb5
                    
                    # Now we need path v2 -> v3 -> v4 -> v5
                    # v3 in N(v2) \ {v0, v1, v5}
                    # v4 in N(v3) \ {v0, v1, v2, v5}
                    # v5 in N(v4)
                    
                    n2 = adj[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v5)
                    
                    while n2:
                        lsb3 = n2 & -n2
                        v3 = lsb3.bit_length() - 1
                        n2 ^= lsb3
                        
                        n3 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v5)
                        
                        # Check if any neighbor of v3 (v4) is connected to v5
                        # v4 in n3
                        # Need v5 in N(v4)
                        
                        # Intersection of n3 and N(v5)?
                        # If n3 & adj[v5] is non-zero, we found a C6.
                        # But we must ensure v4 != v5 (already ensured by mask)
                        # and v4 != v2 (ensured)
                        
                        if n3 & adj[v5]:
                            return True
                            
    return False

def _build_adj_bitsets(edge_colors, n):
    adj = [0] * 4
    for c in range(4):
        adj[c] = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = edge_colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _is_valid_coloring_bitset(edge_colors, n):
    adj = _build_adj_bitsets(edge_colors, n)
    for c in range(4):
        if _has_c6_bitset(adj[c], n):
            return False
    return True

def _spectral_partition(n, seed):
    """
    Partition n vertices into 4 clusters using spectral embedding.
    Returns list of 4 lists of vertex indices.
    """
    rng = np.random.RandomState(seed)
    
    # Generate random regular graph
    d = min(10, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    adj_mat = np.zeros((n, n), dtype=int)
    for _ in range(d // 2):
        perm = rng.permutation(n)
        for i in range(n // 2):
            u = perm[2*i]
            v = perm[2*i+1]
            if u != v and adj_mat[u, v] == 0:
                adj_mat[u, v] = 1
                adj_mat[v, u] = 1
    
    # Laplacian
    degree = adj_mat.sum(axis=1)
    D = np.diag(degree)
    L = D - adj_mat
    
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Top 4 non-trivial eigenvectors (indices 1,2,3,4)
    # If n < 5, handle separately
    if n <= 4:
        return [[i] for i in range(n)]
        
    vecs = eigvecs[:, 1:5]
    
    # K-Means with 4 clusters
    # Initialize centroids randomly from points
    points = vecs.T # (n, 4)
    centroids = points[rng.choice(n, 4, replace=False)]
    
    clusters = [[] for _ in range(4)]
    
    for _ in range(10): # 10 iterations
        # Assign points
        labels = np.argmin(np.linalg.norm(points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2), axis=1)
        
        # Update centroids
        new_centroids = []
        for k in range(4):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids.append(points[mask].mean(axis=0))
            else:
                new_centroids.append(centroids[k]) # Keep old if empty
        centroids = np.array(new_centroids)
        
        # Final assignment
        labels = np.argmin(np.linalg.norm(points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2), axis=1)
        
        for i in range(n):
            clusters[labels[i]].append(i)
            
    return clusters

def _color_from_clusters(clusters, n):
    """
    Generate edge colors based on cluster indices.
    """
    # Map vertex to cluster index
    v_to_c = [0] * n
    for c_idx, cluster in enumerate(clusters):
        for v in cluster:
            v_to_c[v] = c_idx
            
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            ci = v_to_c[i]
            cj = v_to_c[j]
            
            if ci == cj:
                color = 3
            else:
                diff = (cj - ci) % 4
                if diff == 1:
                    color = 0
                elif diff == 3:
                    color = 1
                else: # diff == 2
                    # Randomly assign 2 or 3? 
                    # To avoid bias, use a hash or random
                    # Let's use 2 for now, repair will fix
                    color = 2
            edge_colors.append(color)
            
    return edge_colors

def _repair_coloring_bitset(edge_colors, n, max_steps=2000):
    colors = list(edge_colors)
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    adjs = _build_adj_bitsets(colors, n)
    
    for step in range(max_steps):
        c6_found = False
        c_found = -1
        cycle = None
        
        for c in range(4):
            if _has_c6_bitset(adjs[c], n):
                c_found = c
                c6_found = True
                # We need the actual cycle to recolor. 
                # _has_c6_bitset only returns bool. 
                # We need a version that returns the cycle.
                # Let's modify _has_c6_bitset to return cycle or None.
                break
        
        if not c6_found:
            return colors
            
        # Find the cycle
        # Re-implement search to return cycle
        # This is inefficient to search twice. 
        # Let's create a helper that returns the cycle.
        
        cycle = _find_c6_cycle(adjs[c_found], n)
        if not cycle:
            return colors # Should not happen
            
        # Recolor a random edge in the cycle
        u, v = random.choice([(cycle[i], cycle[(i+1)%6]) for i in range(6)])
        idx = get_edge_idx(u, v)
        old_color = colors[idx]
        
        # Try new colors
        for new_color in range(4):
            if new_color == old_color: continue
            
            # Check if changing (u,v) to new_color creates a C6
            # We need to check if there is a C6 in new_color containing (u,v)
            # This is equivalent to checking if there is a path of length 5 from u to v in G_new_color \ {(u,v)}
            # Since (u,v) is being added, we check path in G_new_color \ {(u,v)}?
            # Actually, we remove (u,v) from old_color, add to new_color.
            # We only need to check if new_color has a C6 involving (u,v).
            
            # Temporarily update adjs
            adjs[old_color][u] &= ~(1 << v)
            adjs[old_color][v] &= ~(1 << u)
            adjs[new_color][u] |= (1 << v)
            adjs[new_color][v] |= (1 << u)
            
            # Check if new_color has a C6 through (u,v)
            # Path u -> ... -> v of length 5 in new_color \ {(u,v)}
            # This is a path of length 5 from u to v in the graph where (u,v) is removed.
            # Since we just added it, we can check if there is a path of length 5 in the *current* adjs[new_color] 
            # that does NOT use the direct edge?
            # No, a C6 containing (u,v) is (u,v) + path of length 5.
            # So we check if there is a path of length 5 from u to v in adjs[new_color] \ {(u,v)}.
            
            def has_path_len_5(adj, s, t, n):
                # adj is list of bitsets
                # s, t are vertices
                # We want path s-v1-v2-v3-v4-t
                # v1 in N(s) \ {t}
                # v4 in N(t) \ {s}
                # Path v1->v2->v3->v4 of length 3
                
                n_s = adj[s] & ~(1 << t)
                n_t = adj[t] & ~(1 << s)
                
                while n_s:
                    lsb1 = n_s & -n_s
                    v1 = lsb1.bit_length() - 1
                    n_s ^= lsb1
                    
                    n_t_temp = n_t
                    while n_t_temp:
                        lsb4 = n_t_temp & -n_t_temp
                        v4 = lsb4.bit_length() - 1
                        n_t_temp ^= lsb4
                        
                        # Need path v1 -> v2 -> v3 -> v4
                        # v2 in N(v1) \ {s, t, v4}
                        # v3 in N(v2) \ {s, t, v1, v4}
                        # v4 in N(v3)
                        
                        n1 = adj[v1] & ~(1 << s) & ~(1 << t) & ~(1 << v4)
                        
                        while n1:
                            lsb2 = n1 & -n1
                            v2 = lsb2.bit_length() - 1
                            n1 ^= lsb2
                            
                            n2 = adj[v2] & ~(1 << s) & ~(1 << t) & ~(1 << v1) & ~(1 << v4)
                            
                            # Check if N(v2) intersects N(v4)
                            # v3 in N(v2) and v3 in N(v4)
                            # But v3 must be distinct from s,t,v1,v2,v4
                            # n2 already excludes s,t,v1,v4
                            # N(v4) might contain v2? We excluded v2 from n2? No, n2 is N(v2).
                            # We need v3 in n2 AND v3 in adj[v4].
                            # And v3 != v4 (ensured if we mask v4 from adj[v4]? No, adj[v4] doesn't have v4).
                            # But we must ensure v3 != v2 (ensured by n2 definition? No, n2 is N(v2), so v2 is not in n2).
                            
                            if n2 & adj[v4]:
                                return True
                                
                return False

            if has_path_len_5(adjs[new_color], u, v, n):
                # Revert
                adjs[new_color][u] &= ~(1 << v)
                adjs[new_color][v] &= ~(1 << u)
                adjs[old_color][u] |= (1 << v)
                adjs[old_color][v] |= (1 << u)
            else:
                # Accept
                colors[idx] = new_color
                break
                
        # If no new color worked, we are stuck. 
        # For now, just break or try random restart?
        # If we break, we return invalid coloring.
        # We should probably try to recolor another edge in the cycle.
        # But for simplicity, if we can't fix it, we fail.
        
    return colors

def _find_c6_cycle(adj, n):
    # Same as _has_c6_bitset but returns cycle
    for v0 in range(n):
        n0 = adj[v0]
        while n0:
            lsb = n0 & -n0
            v1 = lsb.bit_length() - 1
            n0 ^= lsb
            n1 = adj[v1] & ~lsb & ~(1 << v0)
            n0_rest = adj[v0] & ~lsb & ~(1 << v1)
            while n1:
                lsb2 = n1 & -n1
                v2 = lsb2.bit_length() - 1
                n1 ^= lsb2
                temp_n0 = n0_rest
                while temp_n0:
                    lsb5 = temp_n0 & -temp_n0
                    v5 = lsb5.bit_length() - 1
                    temp_n0 ^= lsb5
                    n2 = adj[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v5)
                    while n2:
                        lsb3 = n2 & -n2
                        v3 = lsb3.bit_length() - 1
                        n2 ^= lsb3
                        n3 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v5)
                        if n3 & adj[v5]:
                            # Find v4
                            lsb4 = (n3 & adj[v5]) & -((n3 & adj[v5]))
                            v4 = lsb4.bit_length() - 1
                            return [v0, v1, v2, v3, v4, v5]
    return None

def construct():
    best_n = 1
    best_colors = [0]
    
    for n in range(17, 9, -1):
        found = False
        for seed in range(20):
            clusters = _spectral_partition(n, seed)
            colors = _color_from_clusters(clusters, n)
            colors = _repair_coloring_bitset(colors, n, max_steps=500)
            if _is_valid_coloring_bitset(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    for n in range(9, 1, -1):
        for seed in range(20):
            clusters = _spectral_partition(n, seed)
            colors = _color_from_clusters(clusters, n)
            colors = _repair_coloring_bitset(colors, n, max_steps=500)
            if _is_valid_coloring_bitset(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
