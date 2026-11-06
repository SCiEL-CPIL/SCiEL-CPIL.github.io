import numpy as np
import time
import random

def _generate_regular_graph(n, d, rng):
    """
    Generate a random d-regular graph on n vertices.
    Uses a simple retry method with random permutations.
    """
    if d >= n:
        d = n - 1
    if d % 2 != 0:
        d -= 1
    if d < 2:
        d = 2 if n > 2 else 0

    adj = np.zeros((n, n), dtype=int)
    
    # We need to add d/2 perfect matchings
    for _ in range(d // 2):
        # Try to find a perfect matching that doesn't create multi-edges
        # Simple approach: shuffle vertices, pair them up. Check for conflicts.
        # If conflict, retry the whole matching.
        success = False
        for _attempt in range(100):
            perm = rng.permutation(n)
            valid = True
            edges_to_add = []
            for i in range(n // 2):
                u = perm[2*i]
                v = perm[2*i+1]
                if adj[u, v] == 1:
                    valid = False
                    break
                edges_to_add.append((u, v))
            
            if valid:
                for u, v in edges_to_add:
                    adj[u, v] = 1
                    adj[v, u] = 1
                success = True
                break
        
        if not success:
            # Fallback: just accept irregularity or force it? 
            # For spectral purposes, approximate regularity is okay, 
            # but let's try a different permutation strategy if it fails often.
            # For n<=20, this should succeed quickly.
            pass

    np.fill_diagonal(adj, 0)
    return adj

def _spectral_coloring(n, seed):
    """
    Generate edge colors using spectral embedding.
    1. Generate random regular graph.
    2. Compute Laplacian eigenvectors.
    3. Use first 2 eigenvectors to assign vertices to 4 quadrants.
    4. Color edge (i,j) as (group[i] + group[j]) % 4.
    """
    rng = np.random.RandomState(seed)
    
    d = min(8, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    adj = _generate_regular_graph(n, d, rng)
    
    # Compute Laplacian
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # We want the first 2 non-trivial eigenvectors (indices 1 and 2)
    # to define the 2D embedding for quadrant assignment.
    # The prompt mentions "top 4 eigenvectors", but for 4 colors via quadrants,
    # 2 dimensions are sufficient to define 4 quadrants.
    # If we used 4 dimensions, we'd have 16 orthants.
    # Let's stick to 2 dimensions for clear "quadrant" logic, 
    # or use 4 dimensions and map 16 orthants to 4 groups.
    # The prompt says "derived from the top 4 eigenvectors... assigning colors based on quadrant signs".
    # This implies using all 4.
    # Let's use the first 4 eigenvectors.
    # Vertex i has vector p_i = [v1[i], v2[i], v3[i], v4[i]].
    # We can map the 16 orthants to 4 groups.
    # Or, simpler: use the first 2 eigenvectors to get 4 quadrants.
    # The "top 4" might refer to the fact that we have 4 colors.
    # Let's use the first 2 eigenvectors for the groups, as it's more standard for 4-way partitioning.
    # However, to strictly follow "top 4 eigenvectors", let's compute 4.
    
    if n <= 4:
        # Fallback
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
        
    # Take eigenvectors 1, 2, 3, 4
    vecs = eigvecs[:, 1:5]
    
    # Assign groups based on signs of the first 2 eigenvectors (quadrants)
    # This uses the spectral information to partition the graph.
    groups = np.zeros(n, dtype=int)
    for i in range(n):
        x = vecs[0, i]
        y = vecs[1, i]
        if x >= 0 and y >= 0:
            groups[i] = 0
        elif x >= 0 and y < 0:
            groups[i] = 1
        elif x < 0 and y >= 0:
            groups[i] = 2
        else:
            groups[i] = 3
            
    # To break symmetry and potentially improve, we can add a random rotation
    # to the 2D plane before assigning quadrants.
    # Let's generate the coloring multiple times with different rotations?
    # No, let's just do one per seed. The seed changes the graph, so the eigenvectors change.
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            c = (groups[i] + groups[j]) % 4
            edge_colors.append(c)
            
    return edge_colors

def _has_c6_bitset(adj_bits, n):
    """
    Check if adjacency (list of bitmasks) contains a C6.
    """
    # A C6 is v0-v1-v2-v3-v4-v5-v0
    # We can iterate over edges (v0, v1) and check for paths of length 4 from v1 to v0
    # that avoid v0, v1 internally.
    
    for v0 in range(n):
        # Iterate over neighbors v1 of v0
        mask_v0 = adj_bits[v0]
        # Only consider v1 > v0 to avoid double counting edges? 
        # Actually, for C6 detection, we just need to find ANY C6.
        # Symmetry: v0 < v1 < ... is not required for the cycle, 
        # but we can fix v0 and v1 to be an edge.
        
        # To speed up, we can iterate over edges (v0, v1) with v0 < v1.
        # Then we need a path v1 -> v2 -> v3 -> v4 -> v5 -> v0
        # such that v2, v3, v4, v5 are distinct and not v0, v1.
        
        # Let's iterate v0
        for v1 in range(v0 + 1, n):
            if not (mask_v0 & (1 << v1)):
                continue
            
            # Path from v1 to v0 of length 4: v1-v2-v3-v4-v5-v0
            # v2 is neighbor of v1, not v0
            # v3 is neighbor of v2, not v0, v1
            # v4 is neighbor of v3, not v0, v1, v2
            # v5 is neighbor of v4, not v0, v1, v2, v3
            # AND v0 is neighbor of v5
            
            # This is 4 nested loops.
            # Optimization: Precompute neighbors?
            # For n=17, 2^17 is small.
            
            # v2 candidates: neighbors of v1, excluding v0
            cand_v2 = adj_bits[v1] & ~(1 << v0)
            
            # Iterate v2
            while cand_v2:
                lsb_v2 = cand_v2 & -cand_v2
                v2 = lsb_v2.bit_length() - 1
                cand_v2 ^= lsb_v2
                
                # v3 candidates: neighbors of v2, excluding v0, v1
                cand_v3 = adj_bits[v2] & ~(1 << v0) & ~(1 << v1)
                
                while cand_v3:
                    lsb_v3 = cand_v3 & -cand_v3
                    v3 = lsb_v3.bit_length() - 1
                    cand_v3 ^= lsb_v3
                    
                    # v4 candidates: neighbors of v3, excluding v0, v1, v2
                    cand_v4 = adj_bits[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    
                    while cand_v4:
                        lsb_v4 = cand_v4 & -cand_v4
                        v4 = lsb_v4.bit_length() - 1
                        cand_v4 ^= lsb_v4
                        
                        # v5 candidates: neighbors of v4, excluding v0, v1, v2, v3
                        cand_v5 = adj_bits[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                        
                        # v5 must also be a neighbor of v0
                        cand_v5 &= adj_bits[v0]
                        
                        if cand_v5:
                            # Found a C6
                            return True
    return False

def _build_adj_bits(edge_colors, n):
    """Build adjacency bitmasks for each color class."""
    adj_bits = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = edge_colors[idx]
            adj_bits[c][i] |= (1 << j)
            adj_bits[c][j] |= (1 << i)
            idx += 1
    return adj_bits

def _is_valid_coloring(edge_colors, n):
    """Check if the edge coloring has no C6 in any color class."""
    adj_bits = _build_adj_bits(edge_colors, n)
    for c in range(4):
        if _has_c6_bitset(adj_bits[c], n):
            return False
    return True

def _repair_coloring(edge_colors, n, max_steps=10000):
    """
    Local search to remove C6s using bitmasks for speed.
    """
    colors = list(edge_colors)
    adj_bits = _build_adj_bits(colors, n)
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    def find_c6_vertices(adj_b, n):
        # Returns a list of vertices in the C6 if found, else None
        for v0 in range(n):
            mask_v0 = adj_b[v0]
            for v1 in range(v0 + 1, n):
                if not (mask_v0 & (1 << v1)):
                    continue
                cand_v2 = adj_b[v1] & ~(1 << v0)
                while cand_v2:
                    lsb_v2 = cand_v2 & -cand_v2
                    v2 = lsb_v2.bit_length() - 1
                    cand_v2 ^= lsb_v2
                    cand_v3 = adj_b[v2] & ~(1 << v0) & ~(1 << v1)
                    while cand_v3:
                        lsb_v3 = cand_v3 & -cand_v3
                        v3 = lsb_v3.bit_length() - 1
                        cand_v3 ^= lsb_v3
                        cand_v4 = adj_b[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while cand_v4:
                            lsb_v4 = cand_v4 & -cand_v4
                            v4 = lsb_v4.bit_length() - 1
                            cand_v4 ^= lsb_v4
                            cand_v5 = adj_b[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                            cand_v5 &= adj_b[v0]
                            if cand_v5:
                                lsb_v5 = cand_v5 & -cand_v5
                                v5 = lsb_v5.bit_length() - 1
                                return [v0, v1, v2, v3, v4, v5]
        return None

    def has_c6_through_edge(adj_b, u, v):
        # Check if there is a path of length 4 from v to u in adj_b,
        # such that the internal vertices are distinct from u, v.
        # v - w1 - w2 - w3 - u
        # w1 in adj[v] \ {u}
        # w2 in adj[w1] \ {u, v}
        # w3 in adj[w2] \ {u, v, w1}
        # u in adj[w3]
        
        cand_w1 = adj_b[v] & ~(1 << u)
        while cand_w1:
            lsb_w1 = cand_w1 & -cand_w1
            w1 = lsb_w1.bit_length() - 1
            cand_w1 ^= lsb_w1
            
            cand_w2 = adj_b[w1] & ~(1 << u) & ~(1 << v)
            while cand_w2:
                lsb_w2 = cand_w2 & -cand_w2
                w2 = lsb_w2.bit_length() - 1
                cand_w2 ^= lsb_w2
                
                cand_w3 = adj_b[w2] & ~(1 << u) & ~(1 << v) & ~(1 << w1)
                cand_w3 &= adj_b[u] # w3 must connect to u
                if cand_w3:
                    return True
        return False

    for step in range(max_steps):
        # Find a C6 in any color class
        c6_found = None
        c_found = -1
        for c in range(4):
            c6 = find_c6_vertices(adj_bits[c], n)
            if c6:
                c6_found = c6
                c_found = c
                break
        
        if not c6_found:
            return colors # Valid!
            
        # We have a C6 in color c_found: v0-v1-v2-v3-v4-v5-v0
        edges_in_cycle = [
            (c6_found[0], c6_found[1]),
            (c6_found[1], c6_found[2]),
            (c6_found[2], c6_found[3]),
            (c6_found[3], c6_found[4]),
            (c6_found[4], c6_found[5]),
            (c6_found[5], c6_found[0])
        ]
        
        # Try to recolor one of the edges to break the cycle
        # Pick a random edge
        u, v = random.choice(edges_in_cycle)
        idx = get_edge_idx(u, v)
        old_color = colors[idx]
        
        # Try a new color different from old_color
        # Prefer colors that don't create new C6s immediately
        # Just pick a random different color
        new_color = random.randint(0, 3)
        if new_color == old_color:
            new_color = (new_color + 1) % 4
            
        # Tentatively change color
        colors[idx] = new_color
        
        # Update bitmasks
        adj_bits[old_color][u] &= ~(1 << v)
        adj_bits[old_color][v] &= ~(1 << u)
        adj_bits[new_color][u] |= (1 << v)
        adj_bits[new_color][v] |= (1 << u)
        
        # Check if this creates a new C6 in new_color involving u or v
        # Specifically, check if the edge (u,v) is in a C6 in new_color
        if has_c6_through_edge(adj_bits[new_color], u, v):
            # Revert
            colors[idx] = old_color
            adj_bits[new_color][u] &= ~(1 << v)
            adj_bits[new_color][v] &= ~(1 << u)
            adj_bits[old_color][u] |= (1 << v)
            adj_bits[old_color][v] |= (1 << u)
        else:
            # Accept change
            pass

    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    # Try n=17 first, as it's the known best.
    # If we can find n=18, that would be a breakthrough.
    # Start from 18 down to 10.
    
    for n in range(18, 9, -1):
        found = False
        # Try multiple seeds
        # Limit time per n to avoid timeout
        start_time = time.time()
        for seed in range(100):
            if time.time() - start_time > 90: # Leave time for verification and return
                break
                
            # Generate spectral coloring
            colors = _spectral_coloring(n, seed)
            
            # Repair
            colors = _repair_coloring(colors, n, max_steps=5000)
            
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback for smaller n if 18-10 failed
    for n in range(9, 1, -1):
        for seed in range(20):
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=2000)
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
