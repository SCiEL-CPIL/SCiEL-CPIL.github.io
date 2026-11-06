"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _has_k222_optimized(adj_mask, n):
    """
    Check if graph with adjacency bitmasks adj_mask contains K2,2,2.
    adj_mask[i] is a bitmask of neighbors of i.
    """
    # Iterate over all pairs (a, b)
    for a in range(n):
        for b in range(a + 1, n):
            # Common neighbors of a and b
            s_ab = adj_mask[a] & adj_mask[b]
            # Remove a and b from consideration (though they shouldn't be in each other's mask if simple graph, 
            # but self-loops are 0. Neighbors are distinct from a,b)
            # We need at least 2 common neighbors to form the other parts
            if s_ab.bit_count() < 2:
                continue
            
            # Iterate over pairs (c, d) in s_ab
            # We need to pick two distinct vertices c, d from s_ab
            # To avoid duplicates, we can just iterate over all pairs in the set
            verts_in_sab = []
            mask = s_ab
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                idx = (lsb.bit_length() - 1)
                verts_in_sab.append(idx)
                mask ^= lsb
            
            # If we have at least 2 vertices, iterate over pairs
            # Optimization: If the set is large, this is still C(k, 2)
            # But k is usually small in sparse graphs. In dense graphs, it's large.
            # For Paley(30), degree is ~14. So k ~ 14. C(14,2) = 91.
            # Total pairs (a,b) is C(30,2) = 435.
            # 435 * 91 = ~40k iterations. Much better than 500k.
            
            for i in range(len(verts_in_sab)):
                c = verts_in_sab[i]
                for j in range(i + 1, len(verts_in_sab)):
                    d = verts_in_sab[j]
                    
                    # Common neighbors of c and d
                    s_cd = adj_mask[c] & adj_mask[d]
                    
                    # We need e, f such that:
                    # 1. e, f are in s_ab (neighbors of a, b)
                    # 2. e, f are in s_cd (neighbors of c, d)
                    # 3. e, f are distinct from a, b, c, d
                    # 4. e != f
                    
                    # Intersection of s_ab and s_cd
                    common = s_ab & s_cd
                    
                    # Remove a, b, c, d
                    common &= ~(1 << a)
                    common &= ~(1 << b)
                    common &= ~(1 << c)
                    common &= ~(1 << d)
                    
                    if common.bit_count() >= 2:
                        return True
                        
    return False

def _build_adj_mask(colors, n):
    """Build adjacency bitmasks from color list."""
    adj_mask = np.zeros(n, dtype=np.uint64)
    idx = 0
    for i in range(n):
        mask = 0
        for j in range(i + 1, n):
            if colors[idx] == 1:
                mask |= (1 << j)
                adj_mask[j] |= (1 << i)
            idx += 1
        adj_mask[i] = mask
    return adj_mask

def _is_valid_fast(colors, n):
    """Check if coloring is valid (no K2,2,2 in either color class)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    
    adj1_mask = _build_adj_mask(colors, n)
    if _has_k222_optimized(adj1_mask, n):
        return False
    
    # Build mask for color 0
    # adj0_mask[i] is neighbors in color 0
    # This is equivalent to (full_mask ^ adj1_mask[i]) & ~(1<<i)
    full_mask = (1 << n) - 1
    adj0_mask = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        adj0_mask[i] = (full_mask ^ adj1_mask[i]) & ~(1 << i)
        
    if _has_k222_optimized(adj0_mask, n):
        return False
    return True

def _paley_colors(n, p):
    """
    Generate colors for the first n vertices of the Paley graph on p vertices.
    Color 1 if (j-i) mod p is a quadratic residue.
    Color 0 otherwise.
    """
    if p < n:
        return None
    squares = set((x * x) % p for x in range(1, p))
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            colors.append(1 if diff in squares else 0)
    return colors

def _local_search_extend(colors, n, target_n, deadline):
    """
    Try to extend a valid coloring of n vertices to target_n vertices
    using local search.
    """
    if n >= target_n:
        return True, colors
    
    cur_colors = list(colors)
    cur_n = n
    
    while cur_n < target_n:
        if time.time() > deadline:
            return False, None
        
        new_v = cur_n
        best_assignment = None
        
        # Try multiple random assignments for the new vertex's edges
        for attempt in range(500):
            if time.time() > deadline:
                return False, None
            
            # Generate random assignment for edges from new vertex to existing ones
            assignment = [np.random.randint(2) for _ in range(cur_n)]
            
            temp_colors = cur_colors + assignment
            temp_n = cur_n + 1
            temp_adj1 = _build_adj_mask(temp_colors, temp_n)
            
            if not _has_k222_optimized(temp_adj1, temp_n):
                full_mask = (1 << temp_n) - 1
                temp_adj0 = np.zeros(temp_n, dtype=np.uint64)
                for i in range(temp_n):
                    temp_adj0[i] = (full_mask ^ temp_adj1[i]) & ~(1 << i)
                
                if not _has_k222_optimized(temp_adj0, temp_n):
                    best_assignment = assignment
                    break
        
        if best_assignment is not None:
            cur_colors.extend(best_assignment)
            cur_n += 1
        else:
            # Try flipping edges in current coloring to make room
            success = False
            for flip_attempt in range(500):
                if time.time() > deadline:
                    return False, None
                
                edge_idx = np.random.randint(len(cur_colors))
                old_val = cur_colors[edge_idx]
                cur_colors[edge_idx] = 1 - old_val
                
                assignment = [np.random.randint(2) for _ in range(cur_n)]
                temp_colors = cur_colors + assignment
                temp_n = cur_n + 1
                temp_adj1 = _build_adj_mask(temp_colors, temp_n)
                
                if not _has_k222_optimized(temp_adj1, temp_n):
                    full_mask = (1 << temp_n) - 1
                    temp_adj0 = np.zeros(temp_n, dtype=np.uint64)
                    for i in range(temp_n):
                        temp_adj0[i] = (full_mask ^ temp_adj1[i]) & ~(1 << i)
                    
                    if not _has_k222_optimized(temp_adj0, temp_n):
                        cur_colors.extend(assignment)
                        cur_n += 1
                        success = True
                        break
                else:
                    cur_colors[edge_idx] = old_val
            
            if not success:
                if 'edge_idx' in locals():
                    cur_colors[edge_idx] = old_val
                return False, None
    
    return True, cur_colors

def construct():
    deadline = time.time() + 130  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Try known good constructions
    candidates = []
    
    # Generate Paley 29
    c29 = _paley_colors(29, 29)
    if c29:
        candidates.append((29, c29))
        
    # Generate Paley 37 and try to find a valid subset of size 30
    c37_30 = _paley_colors(30, 37)
    if c37_30:
        candidates.append((30, c37_30))
    
    # Also try Paley with other primes
    for p in [31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p >= 30:
            c_p = _paley_colors(30, p)
            if c_p:
                candidates.append((30, c_p))
    
    # Sort by n descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    for n, colors in candidates:
        if time.time() > deadline:
            break
        try:
            if _is_valid_fast(colors, n):
                best_n = n
                best_colors = colors
                if n == 30:
                    return {'n': n, 'colors': colors}
        except Exception:
            continue
            
    # If 30 failed, try local search to extend 29 to 30
    if best_n == 29 and c29:
        if time.time() < deadline - 10:
            try:
                success, ext_colors = _local_search_extend(c29, 29, 30, deadline)
                if success and ext_colors:
                    if _is_valid_fast(ext_colors, 30):
                        return {'n': 30, 'colors': ext_colors}
            except Exception:
                pass
    
    if best_colors is None:
        return {'n': 1, 'colors': []}
        
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
