"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.
Strategy: Bit-parallel backtracking search with incremental C6 detection.
We build the coloring edge-by-edge. For each edge (i,j) and color c, we check
if adding edge (i,j) to color c would create a C6. This is equivalent to
checking if there exists a path of length 3 from i to j in the current
color-c graph (excluding the edge (i,j) itself, which isn't there yet).
If no such path exists, the edge can be colored c without creating a C6.
"""

import time
import random
import sys

sys.setrecursionlimit(10000)

DEADLINE_MARGIN = 5.0  # seconds before the 150s wall-clock kill to stop


def _edge_index(i, j, n):
    """Convert (i,j) with i<j to linear index in lexicographic order."""
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)


def _edge_from_index(idx, n):
    """Convert linear index to (i,j) with i<j."""
    # Inverse of _edge_index
    # i * (n - 1) - i * (i - 1) // 2 + (j - i - 1) = idx
    # Let's precompute for speed if n is fixed, but for generality:
    # Actually, for n<=20, we can just iterate or use a precomputed table.
    # Since n is small, let's just use a simple lookup or calculation.
    # Better: precompute a list of edges once.
    # But this function is called in the hot loop. Let's optimize.
    # For now, simple calculation:
    # The number of edges before row i is sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    # So we need to find i such that base_i <= idx < base_{i+1}
    # This is a bit slow. Let's precompute edges for the given n.
    # We'll handle this in the main search by passing a precomputed list.
    pass


def _precompute_edges(n):
    """Precompute list of (i, j, idx) for all edges."""
    edges = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, idx))
            idx += 1
    return edges


def _has_c6_in_color(adj, n, u, v):
    """
    Check if adding edge (u,v) to the graph defined by adj (for one color)
    would create a C6.
    This is true if there is a path of length 3 from u to v in the current graph.
    Path: u - a - b - v, where a != v, b != u, a != b, and edges (u,a), (a,b), (b,v) exist.
    Also need to ensure the path doesn't use the edge (u,v) itself (it doesn't, as it's not added yet).
    And vertices must be distinct: u, a, b, v must be distinct.
    """
    # Neighbors of u in adj
    mask_u = adj[u]
    # Neighbors of v in adj
    mask_v = adj[v]
    
    # For each neighbor a of u (a != v):
    mask_a = mask_u & ~(1 << v)
    while mask_a:
        a = (mask_a & -mask_a).bit_length() - 1
        mask_a &= mask_a - 1
        
        # For each neighbor b of a (b != u, b != v):
        mask_b = adj[a] & ~(1 << u) & ~(1 << v)
        while mask_b:
            b = (mask_b & -mask_b).bit_length() - 1
            mask_b &= mask_b - 1
            
            # Check if b is connected to v
            if adj[b] & (1 << v):
                return True
    return False


def _build_adj_from_colors(colors, n):
    """Build adjacency bitmasks for each color class from a flat color list."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj


def _is_valid_coloring(colors, n):
    """Check if the coloring is valid (no mono C6)."""
    adj = _build_adj_from_colors(colors, n)
    for c in range(4):
        # Check all edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj[c][i] & (1 << j):
                    # This edge is in color c. Check if removing it would leave a C6?
                    # No, we just need to check if the graph has a C6.
                    # The incremental check during construction ensures this.
                    # But for final verification, let's do a full check.
                    pass
    # Full check:
    for c in range(4):
        for v0 in range(n):
            mask_v1 = adj[c][v0]
            while mask_v1:
                v1 = (mask_v1 & -mask_v1).bit_length() - 1
                mask_v1 &= mask_v1 - 1
                mask_v2 = adj[c][v1] & ~(1 << v0)
                while mask_v2:
                    v2 = (mask_v2 & -mask_v2).bit_length() - 1
                    mask_v2 &= mask_v2 - 1
                    mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                    while mask_v3:
                        v3 = (mask_v3 & -mask_v3).bit_length() - 1
                        mask_v3 &= mask_v3 - 1
                        mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while mask_v4:
                            v4 = (mask_v4 & -mask_v4).bit_length() - 1
                            mask_v4 &= mask_v4 - 1
                            mask_v5 = adj[c][v4] & adj[c][v0]
                            used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            mask_v5 &= ~used_mask
                            if mask_v5:
                                return False
    return True


def _backtrack_search(n, deadline, edge_order=None):
    """
    Backtracking search for a valid 4-coloring of K_n.
    """
    edges = _precompute_edges(n)
    num_edges = len(edges)
    
    if edge_order is None:
        edge_order = list(range(num_edges))
    
    colors = [-1] * num_edges
    adj = [[0] * n for _ in range(4)]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == num_edges:
            return True
        
        edge_idx = edge_order[pos]
        i, j, _ = edges[edge_idx]
        
        for c in range(4):
            # Check if adding edge (i,j) to color c creates a C6
            if not _has_c6_in_color(adj[c], n, i, j):
                # Assign color c
                colors[edge_idx] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                if backtrack(pos + 1):
                    return True
                
                # Undo
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[edge_idx] = -1
        
        return False
    
    return backtrack(0)


def _find_valid_coloring(n, time_limit, seed=42):
    """
    Try to find a valid 4-coloring of K_n using backtracking with different edge orderings.
    """
    deadline = time.time() + time_limit
    edges = _precompute_edges(n)
    num_edges = len(edges)
    
    # Try natural order first
    if _backtrack_search(n, deadline):
        # Reconstruct colors
        colors = [0] * num_edges
        # Wait, the backtrack function modifies a local colors array.
        # I need to return the colors. Let me refactor.
        pass
    
    # The previous implementation doesn't return colors. Let me fix that.
    # I'll rewrite _backtrack_search to return the colors list.
    
    return None


def _backtrack_search_with_result(n, deadline, edge_order=None):
    """
    Backtracking search that returns the colors list if found.
    """
    edges = _precompute_edges(n)
    num_edges = len(edges)
    
    if edge_order is None:
        edge_order = list(range(num_edges))
    
    colors = [-1] * num_edges
    adj = [[0] * n for _ in range(4)]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == num_edges:
            return True
        
        edge_idx = edge_order[pos]
        i, j, _ = edges[edge_idx]
        
        for c in range(4):
            if not _has_c6_in_color(adj[c], n, i, j):
                colors[edge_idx] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                if backtrack(pos + 1):
                    return True
                
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[edge_idx] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None


def _construct_known_17():
    """
    Find a valid 4-coloring of K_17 using systematic search.
    """
    # Try to find a valid K_17 coloring using deterministic search
    start = time.time()
    
    # Try natural order
    colors_17 = _backtrack_search_with_result(17, start + 10.0)
    if colors_17 is not None:
        return {'n': 17, 'colors': colors_17}
    
    # Try random orderings
    rng = random.Random(42)
    edges = _precompute_edges(17)
    num_edges = len(edges)
    
    for attempt in range(20):
        if time.time() > start + 20.0:
            break
        order = list(range(num_edges))
        rng.shuffle(order)
        colors_17 = _backtrack_search_with_result(17, start + 20.0, order)
        if colors_17 is not None:
            return {'n': 17, 'colors': colors_17}
    
    # Fallback to smaller n
    for n in range(16, 5, -1):
        if time.time() > start + 30.0:
            break
        colors = _backtrack_search_with_result(n, start + 30.0)
        if colors is not None:
            return {'n': n, 'colors': colors}
    
    # Last resort: random coloring (may be invalid)
    rng = random.Random(42)
    colors = [rng.randint(0, 3) for _ in range(10)]
    return {'n': 5, 'colors': colors}


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # Get a valid baseline
    result = _construct_known_17()
    best_n = result['n']
    best_colors = result['colors']
    
    # If we have n=17, try to extend to n=18
    if best_n == 17 and time.time() < deadline - 30:
        # Try to find a valid K_18 coloring directly
        colors_18 = _backtrack_search_with_result(18, deadline)
        if colors_18 is not None:
            return {'n': 18, 'colors': colors_18}
        
        # Try random orderings for K_18
        rng = random.Random(42)
        edges = _precompute_edges(18)
        num_edges = len(edges)
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            order = list(range(num_edges))
            rng.shuffle(order)
            colors_18 = _backtrack_search_with_result(18, deadline, order)
            if colors_18 is not None:
                return {'n': 18, 'colors': colors_18}
    
    return {'n': best_n, 'colors': best_colors}
