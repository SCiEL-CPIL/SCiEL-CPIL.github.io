"""Search for max n with 4-colouring of K_n having no monochromatic C6."""
import time
import random
import itertools

_DEADLINE = 115.0  # safety margin under 120s
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _edge_index(n, i, j):
    if i > j: i, j = j, i
    return i * n - i * (i + 1) // 2 + (j - i - 1)

# Precompute C6 patterns in K6 as 15-bit masks and edge lists
# Vertices 0..5. Edges (0,1)..(4,5) mapped to bits 0..14.
# Edge (i,j) index in K6: i*6 - i*(i+1)//2 + (j-i-1)
def _get_c6_data():
    patterns = []
    pattern_edges = []
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        cyc = (0, perm[0], perm[1], perm[2], perm[3], perm[4])
        if perm[0] > perm[4]:
            continue
        
        mask = 0
        edges = []
        for k in range(6):
            u, v = cyc[k], cyc[(k+1)%6]
            if u > v: u, v = v, u
            idx = u * 6 - u * (u + 1) // 2 + (v - u - 1)
            mask |= (1 << idx)
            edges.append(idx)
        patterns.append(mask)
        pattern_edges.append(tuple(edges))
    return patterns, pattern_edges

_C6_PATTERNS, _C6_PATTERN_EDGES = _get_c6_data()

def _is_valid_fast(colors, n):
    """Check if coloring is valid (no monochromatic C6) using bitmask check on 6-subsets."""
    if n < 6:
        return True
    
    for subset in itertools.combinations(range(n), 6):
        masks = [0, 0, 0, 0]
        
        for i in range(6):
            for j in range(i+1, 6):
                u, v = subset[i], subset[j]
                idx_global = _edge_index(n, u, v)
                c = colors[idx_global]
                idx_local = i * 6 - i * (i + 1) // 2 + (j - i - 1)
                masks[c] |= (1 << idx_local)
        
        for c in range(4):
            m = masks[c]
            for p in _C6_PATTERNS:
                if (m & p) == p:
                    return False
    return True

def _min_conflicts_fast(colors, n, max_iter=50000):
    """Local search using incremental conflict counting."""
    m = n * (n - 1) // 2
    cur = colors[:]
    
    # Precompute structure for incremental updates
    # subsets: list of 6-tuples of vertices
    # subset_edge_map: for each subset index, list of 15 global edge indices
    # pattern_edges_local: for each pattern, list of 6 local edge indices (0-14)
    
    subsets = list(itertools.combinations(range(n), 6))
    num_subsets = len(subsets)
    
    # Map local edge (i,j) in K6 to global edge index for each subset
    # local_edge_idx: 0..14
    # We need to know which subsets and patterns are affected by a global edge change.
    
    # Affected subsets for each global edge
    # edge_to_subsets[e] = list of subset indices
    edge_to_subsets = [[] for _ in range(m)]
    
    # For each subset, compute the global edge indices for its 15 local edges
    # subset_global_edges[s] = list of 15 global edge indices
    subset_global_edges = [None] * num_subsets
    
    # Precompute which patterns contain which local edges
    # pattern_to_local_edges[p] = tuple of 6 local edge indices
    # local_edge_to_patterns[le] = list of pattern indices
    local_edge_to_patterns = [[] for _ in range(15)]
    for p_idx, p_edges in enumerate(_C6_PATTERN_EDGES):
        for le in p_edges:
            local_edge_to_patterns[le].append(p_idx)
            
    for s_idx, subset in enumerate(subsets):
        edges = []
        for i in range(6):
            for j in range(i+1, 6):
                u, v = subset[i], subset[j]
                g_idx = _edge_index(n, u, v)
                edges.append(g_idx)
        subset_global_edges[s_idx] = edges
        
        # Update edge_to_subsets
        for g_idx in edges:
            edge_to_subsets[g_idx].append(s_idx)
            
    # Conflict counts: conflicts[s_idx][p_idx][c]
    # Dimensions: num_subsets x 60 x 4
    # Using a list of lists of lists or a flat array?
    # num_subsets can be large (e.g. 12376 for n=17).
    # 12376 * 60 * 4 = ~3M integers. This is manageable.
    # But updating it efficiently is key.
    
    # Let's use a list of arrays for speed? Or just nested lists.
    # conflicts[s_idx] is a list of 60 patterns, each pattern is a list of 4 counts.
    conflicts = [[ [0]*4 for _ in range(len(_C6_PATTERNS)) ] for _ in range(num_subsets)]
    
    total_conflicts = 0
    
    # Initialize conflicts
    for s_idx, subset in enumerate(subsets):
        # Build masks for this subset
        masks = [0, 0, 0, 0]
        g_edges = subset_global_edges[s_idx]
        for le in range(15):
            g_idx = g_edges[le]
            c = cur[g_idx]
            masks[c] |= (1 << le)
            
        for p_idx, p_mask in enumerate(_C6_PATTERNS):
            for c in range(4):
                if (masks[c] & p_mask) == p_mask:
                    conflicts[s_idx][p_idx][c] = 1
                    total_conflicts += 1
                    
    if total_conflicts == 0:
        return cur, 0
        
    best = cur[:]
    best_conf = total_conflicts
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
            
        # Pick a random edge
        e = random.randrange(m)
        old_c = cur[e]
        new_c = (old_c + 1 + random.randrange(3)) % 4
        
        # Update conflicts for all subsets containing edge e
        delta = 0
        for s_idx in edge_to_subsets[e]:
            g_edges = subset_global_edges[s_idx]
            # Find local index of edge e in this subset
            # We can precompute a map from global edge to local index for each subset?
            # Or just find it. Since we have the list, finding is O(15).
            # Let's precompute local_index_map for each subset?
            # Actually, we can just iterate le in 0..14 and check if g_edges[le] == e.
            le = -1
            for k in range(15):
                if g_edges[k] == e:
                    le = k
                    break
            if le == -1: continue # Should not happen
            
            # Patterns affected are those containing local edge le
            for p_idx in local_edge_to_patterns[le]:
                p_mask = _C6_PATTERNS[p_idx]
                
                # Check if the new color creates a conflict
                # We need to know the colors of the other 5 edges in the pattern
                # If the other 5 edges in the pattern are all new_c, then the pattern becomes monochromatic
                # If the other 5 edges in the pattern are all old_c, then the pattern ceases to be monochromatic (if it was)
                
                # Let's check the colors of the 5 other edges in the pattern
                p_edges = _C6_PATTERN_EDGES[p_idx]
                
                # Count how many of the other 5 edges have color new_c and old_c
                # Note: The edge e itself is excluded from this count
                count_new = 0
                count_old = 0
                for k in p_edges:
                    if k == le:
                        continue
                    g_idx = g_edges[k]
                    c = cur[g_idx]
                    if c == new_c:
                        count_new += 1
                    elif c == old_c:
                        count_old += 1
                
                # If count_new == 5, then the pattern is monochromatic in new_c
                # If count_old == 5, then the pattern was monochromatic in old_c (and no longer is)
                
                # Update conflicts[s_idx][p_idx][new_c]
                if count_new == 5:
                    # Becomes conflict
                    if conflicts[s_idx][p_idx][new_c] == 0:
                        conflicts[s_idx][p_idx][new_c] = 1
                        delta += 1
                else:
                    # No conflict in new_c (even if it was, it's not, because one edge changed away from it? 
                    # Wait. If the edge changed FROM old_c TO new_c.
                    # If the other 5 are new_c, then now all 6 are new_c. Conflict created.
                    # If the other 5 are NOT all new_c, then no conflict in new_c.
                    # What if it was a conflict in new_c before? That's impossible because the edge was old_c.
                    # So conflicts[s_idx][p_idx][new_c] can only go 0 -> 1.
                    pass

                # Update conflicts[s_idx][p_idx][old_c]
                if count_old == 5:
                    # Was conflict in old_c, now broken
                    if conflicts[s_idx][p_idx][old_c] == 1:
                        conflicts[s_idx][p_idx][old_c] = 0
                        delta -= 1
                else:
                    # Was not conflict in old_c (or still is? No, edge is no longer old_c)
                    # So conflicts[s_idx][p_idx][old_c] can only go 1 -> 0.
                    pass

        total_conflicts += delta
        cur[e] = new_c
        
        if total_conflicts < best_conf:
            best = cur[:]
            best_conf = total_conflicts
            if best_conf == 0:
                return best, 0
        else:
            # Revert
            cur[e] = old_c
            total_conflicts -= delta
            
    return best, best_conf

def construct():
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    random.seed(12345)
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if _remaining() < 10.0:
            break
            
        m = n * (n - 1) // 2
        found = False
        
        # Try multiple restarts of Min-Conflicts
        # With incremental updates, we can do more iterations
        # Estimate: 1 move takes ~1000-2000 ops. 100k moves ~ 10-20 seconds?
        # Let's try 50000 iterations per restart.
        for restart in range(10):
            if _remaining() < 8.0:
                break
            rand_colors = [random.randrange(4) for _ in range(m)]
            res, conf = _min_conflicts_fast(rand_colors, n, max_iter=50000)
            if conf == 0:
                # Verify with fast validator
                if _is_valid_fast(res, n):
                    best_n = n
                    best_colors = res
                    found = True
                    break
        
        if found:
            break
            
    # Ensure we at least have the seed n=9
    if best_n < 9:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
        
    return {'n': best_n, 'colors': best_colors}
