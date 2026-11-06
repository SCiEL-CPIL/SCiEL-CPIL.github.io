"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

# Pre-compute pairings for K2,2,2 detection
_PAIRINGS = [
    ((0,1),(2,3),(4,5)), ((0,1),(2,4),(3,5)), ((0,1),(2,5),(3,4)),
    ((0,2),(1,3),(4,5)), ((0,2),(1,4),(3,5)), ((0,2),(1,5),(3,4)),
    ((0,3),(1,2),(4,5)), ((0,3),(1,4),(2,5)), ((0,3),(1,5),(2,4)),
    ((0,4),(1,2),(3,5)), ((0,4),(1,3),(2,5)), ((0,4),(1,5),(2,3)),
    ((0,5),(1,2),(3,4)), ((0,5),(1,3),(2,4)), ((0,5),(1,4),(2,3)),
]

def _has_k222_fast(adj, n):
    """
    Check if graph with adjacency matrix adj (numpy 2D array of 0/1) on n vertices
    contains K2,2,2. Uses bitmask representation for speed.
    """
    # Convert adjacency to bitmasks for faster intersection checks
    # adj_mask[i] is a bitmask of vertices connected to i
    adj_mask = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i, j]:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # For each 6-subset, check all 15 pairings
    for verts in itertools.combinations(range(n), 6):
        v = list(verts)
        for p1, p2, p3 in _PAIRINGS:
            a, b = v[p1[0]], v[p1[1]]
            c, d = v[p2[0]], v[p2[1]]
            e, f = v[p3[0]], v[p3[1]]
            
            # Check if A-B is a complete bipartite: all edges between {a,b} and {c,d}
            # adj_mask[a] & adj_mask[b] should contain c and d
            ab_common = adj_mask[a] & adj_mask[b]
            if not (ab_common & (1 << c)) or not (ab_common & (1 << d)):
                continue
            
            # Check if A-C is a complete bipartite: all edges between {a,b} and {e,f}
            ac_common = adj_mask[a] & adj_mask[b]
            if not (ac_common & (1 << e)) or not (ac_common & (1 << f)):
                continue
            
            # Check if B-C is a complete bipartite: all edges between {c,d} and {e,f}
            bc_common = adj_mask[c] & adj_mask[d]
            if not (bc_common & (1 << e)) or not (bc_common & (1 << f)):
                continue
            
            return True
    return False

def _build_adj_fast(colors, n):
    """Build adjacency matrix as numpy 2D array of 0/1 from color list."""
    adj = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if colors[idx] == 1:
                adj[i, j] = 1
                adj[j, i] = 1
            idx += 1
    return adj

def _is_valid_fast(colors, n):
    """Check if coloring is valid (no K2,2,2 in either color class)."""
    if len(colors) != n*(n-1)//2:
        return False
    
    adj1 = _build_adj_fast(colors, n)
    if _has_k222_fast(adj1, n):
        return False
    
    # Complement graph
    adj0 = (1 - adj1).astype(np.int8)
    np.fill_diagonal(adj0, 0)
    
    if _has_k222_fast(adj0, n):
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
    squares = set((x*x) % p for x in range(1, p))
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % p
            colors.append(1 if diff in squares else 0)
    return colors

def _local_search_extend(colors, n, target_n, deadline):
    """
    Try to extend a valid coloring of n vertices to target_n vertices
    using local search with edge flipping. Returns (success, extended_colors) or (False, None).
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
        for attempt in range(200):
            if time.time() > deadline:
                return False, None
            
            # Generate random assignment for edges from new vertex to existing ones
            assignment = [np.random.randint(2) for _ in range(cur_n)]
            
            temp_colors = cur_colors + assignment
            temp_n = cur_n + 1
            temp_adj = _build_adj_fast(temp_colors, temp_n)
            
            if not _has_k222_fast(temp_adj, temp_n):
                temp_adj0 = (1 - temp_adj).astype(np.int8)
                np.fill_diagonal(temp_adj0, 0)
                if not _has_k222_fast(temp_adj0, temp_n):
                    best_assignment = assignment
                    break
        
        if best_assignment is not None:
            cur_colors.extend(best_assignment)
            cur_n += 1
        else:
            # Try flipping edges in current coloring to make room
            success = False
            for flip_attempt in range(200):
                if time.time() > deadline:
                    return False, None
                
                edge_idx = np.random.randint(len(cur_colors))
                old_val = cur_colors[edge_idx]
                cur_colors[edge_idx] = 1 - old_val
                
                assignment = [np.random.randint(2) for _ in range(cur_n)]
                temp_colors = cur_colors + assignment
                temp_n = cur_n + 1
                temp_adj = _build_adj_fast(temp_colors, temp_n)
                
                if not _has_k222_fast(temp_adj, temp_n):
                    temp_adj0 = (1 - temp_adj).astype(np.int8)
                    np.fill_diagonal(temp_adj0, 0)
                    if not _has_k222_fast(temp_adj0, temp_n):
                        cur_colors.extend(assignment)
                        cur_n += 1
                        success = True
                        break
                else:
                    cur_colors[edge_idx] = old_val
            
            if not success:
                # Restore edge if we flipped it
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
        if time.time() < deadline - 15:
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
