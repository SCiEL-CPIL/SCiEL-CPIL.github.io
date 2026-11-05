"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.

Strategy:
- Fast C6 existence test using bitmasks.
- Local search that minimises the number of monochromatic C6s.
- Incremental C6 counting: when an edge colour changes, only recompute the
  contribution of that edge to each colour class's C6 count.
- Start from n=17 and try to improve; fall back to smaller n if needed.
"""
import time
import random

DEADLINE_MARGIN = 5.0


def _has_c6_bitset(adj, n):
    """
    Check if the graph defined by adjacency masks 'adj' contains a 6-cycle.
    adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists.
    """
    for v0 in range(n):
        mask_v1 = adj[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1
            mask_v2 = adj[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1
                mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1
                    mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1
                        mask_v5 = adj[v4] & adj[v0]
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
                        if mask_v5:
                            return True
    return False


def _build_adj(colors, n):
    """Build adjacency bitmasks for each colour class."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj


def _count_c6s_for_color(adj_c, n):
    """Count C6s in a single colour class (adj_c is list of bitmasks)."""
    total = 0
    for v0 in range(n):
        mask_v1 = adj_c[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1
            mask_v2 = adj_c[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1
                mask_v3 = adj_c[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1
                    mask_v4 = adj_c[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1
                        mask_v5 = adj_c[v4] & adj_c[v0]
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
                        if mask_v5:
                            total += 1
    return total


def _edge_index(i, j, n):
    """Map vertex pair (i,j) with i<j to edge index in lexicographic order."""
    # Number of edges before vertex i: i*(n-1) - i*(i-1)//2
    # Edges from i to vertices < i are not counted since we only consider i<j.
    # Actually, for lexicographic order of pairs (a,b) with a<b:
    # For vertex i, the edges are (i, i+1), (i, i+2), ..., (i, n-1).
    # The index of (i, j) is: sum_{k=0}^{i-1} (n-1-k) + (j - i - 1)
    # = i*(n-1) - i*(i-1)//2 + j - i - 1
    # Let's verify: n=4, edges: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
    # i=0, j=1: 0*(3) - 0 + 1-0-1 = 0. Correct.
    # i=0, j=2: 0 + 2-0-1 = 1. Correct.
    # i=1, j=2: 1*(3) - 1*0//2 + 2-1-1 = 3 + 0 = 3. Correct.
    # i=1, j=3: 1*(3) - 0 + 3-1-1 = 3+1 = 4. Correct.
    # i=2, j=3: 2*(3) - 2*1//2 + 3-2-1 = 6 - 1 + 0 = 5. Correct.
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)


def _vertex_pair(edge_idx, n):
    """Map edge index back to vertex pair (i, j) with i < j."""
    # We can precompute or do binary search. For n<=20, linear is fine.
    # But let's do a direct inverse.
    # The number of edges before vertex i is: i*(n-1) - i*(i-1)//2
    # We need to find i such that:
    # i*(n-1) - i*(i-1)//2 <= edge_idx < (i+1)*(n-1) - (i+1)*i//2
    # Let's just iterate since n is small.
    idx = 0
    for i in range(n):
        count = n - 1 - i
        if edge_idx < idx + count:
            j = i + 1 + (edge_idx - idx)
            return i, j
        idx += count
    return -1, -1


def _find_valid_coloring(n, time_limit=10.0, seed=42):
    """
    Find a valid 4-coloring of K_n using local search with incremental C6 counting.
    """
    random.seed(seed)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2

    # Precompute vertex pairs for each edge index
    edge_to_vertices = []
    for e in range(num_edges):
        edge_to_vertices.append(_vertex_pair(e, n))

    # Start with random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    adj = _build_adj(colors, n)
    
    # Compute initial C6 counts per color
    c6_counts = [_count_c6s_for_color(adj[c], n) for c in range(4)]
    total_c6 = sum(c6_counts)
    
    if total_c6 == 0:
        return colors
    
    best_colors = colors[:]
    best_total = total_c6
    
    # Local search: flip random edges, keep if it reduces total C6 count
    # Use incremental update: when edge e changes from color c_old to c_new,
    # we need to recompute C6 counts for c_old and c_new.
    # For efficiency, we can recompute the full count for those two colors.
    # Since n<=20, this is fast enough.
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        if best_total == 0:
            return best_colors
        
        # Pick a random edge to flip
        edge_idx = random.randint(0, num_edges - 1)
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        
        if old_color == new_color:
            continue
        
        colors[edge_idx] = new_color
        adj = _build_adj(colors, n)
        c6_counts = [_count_c6s_for_color(adj[c], n) for c in range(4)]
        new_total = sum(c6_counts)
        
        if new_total < best_total:
            best_total = new_total
            best_colors = colors[:]
        else:
            # Revert
            colors[edge_idx] = old_color
    
    if best_total == 0:
        return best_colors
    return None


def _construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    best_result = None
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 2.0:
            break
        time_left = deadline - time.time()
        if time_left < 3.0:
            break
        
        # Try a few seeds
        for seed in [42, 123, 7, 999, 555]:
            if time.time() > deadline - 2.0:
                break
            time_left = deadline - time.time()
            if time_left < 2.0:
                break
            
            result = _find_valid_coloring(n, time_limit=min(time_left - 1.0, 15.0), seed=seed)
            if result is not None:
                best_result = {'n': n, 'colors': result}
                break
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: try smaller n with more time
        for n in range(9, 3, -1):
            if time.time() > deadline - 1.0:
                break
            time_left = deadline - time.time()
            if time_left < 1.0:
                break
            result = _find_valid_coloring(n, time_limit=max(1.0, time_left - 0.5), seed=42)
            if result is not None:
                best_result = {'n': n, 'colors': result}
                break
    
    if best_result is None:
        # Absolute fallback: n=4
        colors = [0, 1, 2, 3, 0, 1]
        best_result = {'n': 4, 'colors': colors}
    
    return best_result


def construct():
    return _construct()
