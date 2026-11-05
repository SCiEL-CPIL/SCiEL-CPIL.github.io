"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.
Combined approach:
  1. Direct Z_17 cyclic construction (fast, known-valid if partition is right).
  2. Bit-parallel backtracking with constraint propagation for n=18, 17, 16.
  3. Local search with C6-counting objective as fallback.
"""
import time
import random
import sys
import numpy as np

DEADLINE_MARGIN = 5.0

def _build_adjacency(colors, n):
    """Build adjacency masks for each color class."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _has_c6_in_adj(adj, n):
    """Check if graph with adjacency masks 'adj' contains a 6-cycle."""
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

def _is_valid(colors, n):
    """Check if the coloring is valid (no mono C6 in any color class)."""
    adj = _build_adjacency(colors, n)
    for c in range(4):
        if _has_c6_in_adj(adj[c], n):
            return False
    return True

def _count_c6s(adj, n):
    """Count total number of C6s across all 4 color classes."""
    total = 0
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
                                total += mask_v5.bit_count()
    return total

def _construct_z17():
    """
    Construct a 4-coloring of K_17 using the cyclic group Z_17.
    """
    partitions = [
        [{1, 2, 4, 8}, {3, 5, 6, 7}, {9, 10, 12, 14}, {11, 13, 15, 16}],
        [{1, 3, 4, 9}, {2, 5, 7, 10}, {6, 8, 11, 13}, {12, 14, 15, 16}],
        [{1, 2, 4, 8}, {9, 13, 15, 16}, {3, 5, 6, 7}, {10, 11, 12, 14}],
        [{1, 2, 4, 16}, {3, 5, 7, 9}, {6, 8, 10, 12}, {11, 13, 14, 15}],
        [{1, 3, 6, 12}, {2, 4, 8, 9}, {5, 7, 11, 14}, {10, 13, 15, 16}],
        [{1, 4, 10, 14}, {2, 3, 7, 16}, {5, 6, 11, 13}, {8, 9, 12, 15}],
        [{1, 5, 6, 10}, {2, 7, 11, 16}, {3, 8, 12, 15}, {4, 9, 13, 14}],
    ]
    
    n = 17
    num_edges = n * (n - 1) // 2
    
    for part in partitions:
        color_map = [0] * 17
        for c, s in enumerate(part):
            for d in s:
                color_map[d] = c
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 0:
                    d = n
                colors.append(color_map[d])
        
        if _is_valid(colors, n):
            return {'n': n, 'colors': colors}
    
    return None

def _backtracking_search(n, deadline, max_depth=300000):
    """
    Bit-parallel backtracking search for a valid 4-coloring of K_n.
    Edges are assigned in lexicographic order.
    Uses constraint propagation: when assigning color c to edge (i,j),
    check if this creates a C6 in color c.
    """
    num_edges = n * (n - 1) // 2
    colors = [-1] * num_edges
    adj = [[0] * n for _ in range(4)]
    
    # Precompute edge indices
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
    
    # State for backtracking
    state = {
        'colors': colors,
        'adj': adj,
        'edge_pos': 0,
        'node_count': 0,
        'found': False,
        'solution': None
    }
    
    def _check_c6_after_add(color, i, j):
        """
        Check if adding edge (i,j) with given color creates a C6.
        A C6 through (i,j) would be i-j-v2-v3-v4-v5-i or similar.
        We need to check if there exists a path of length 4 from i to j
        in the color graph, avoiding i and j in the middle.
        Actually, a C6 containing edge (i,j) is: i - j - a - b - c - d - i
        So we need a path j-a-b-c-d-i of length 4 in color, with all distinct.
        Equivalently, a path from j to i of length 4 in color (after adding i-j).
        But we can check: is there a path of length 4 from i to j in the current
        color graph (before adding i-j)? No, that would be a C5.
        
        Let's think differently. After adding (i,j) to color c, a C6 exists
        if there is a path of length 5 between i and j in color c, OR
        a path of length 4 between i and j in color c (forming a C5 with the edge... no).
        
        Actually, the simplest check: after adding (i,j), check if there's a C6
        containing (i,j). This means there exist distinct vertices a,b,c,d such that
        j-a, a-b, b-c, c-d, d-i are all in color c, and a,b,c,d are distinct from i,j.
        
        So we need: a in N_c(j), b in N_c(a) \\ {j}, c in N_c(b) \\ {j,a},
        d in N_c(c) \\ {j,a,b}, and d in N_c(i) \\ {j,a,b,c}.
        
        This is a 4-hop path from j to i avoiding i,j in the middle.
        """
        # Get neighbors of j in color c
        neighbors_j = adj[color][j] & ~(1 << i)  # exclude i (the edge we're adding)
        if not neighbors_j:
            return False
        
        # For each a in neighbors_j
        mask_a = neighbors_j
        while mask_a:
            a = (mask_a & -mask_a).bit_length() - 1
            mask_a &= mask_a - 1
            
            # b in N_c(a) excluding j and i
            mask_b = adj[color][a] & ~(1 << j) & ~(1 << i)
            while mask_b:
                b = (mask_b & -mask_b).bit_length() - 1
                mask_b &= mask_b - 1
                if b == a:
                    continue
                
                # c in N_c(b) excluding j, i, a
                mask_c = adj[color][b] & ~(1 << j) & ~(1 << i) & ~(1 << a)
                while mask_c:
                    c = (mask_c & -mask_c).bit_length() - 1
                    mask_c &= mask_c - 1
                    if c == a or c == b:
                        continue
                    
                    # d in N_c(c) excluding j, i, a, b, and d must be in N_c(i)
                    mask_d = adj[color][c] & ~(1 << j) & ~(1 << i) & ~(1 << a) & ~(1 << b)
                    mask_d &= adj[color][i] & ~(1 << j) & ~(1 << a) & ~(1 << b)
                    if mask_d:
                        return True
        
        return False
    
    def _backtrack(pos):
        """Recursive backtracking. Returns True if solution found."""
        if state['found']:
            return True
        
        if time.time() > deadline:
            return False
        
        state['node_count'] += 1
        if state['node_count'] % 10000 == 0 and state['node_count'] > max_depth:
            return False
        
        if pos == num_edges:
            state['found'] = True
            state['solution'] = colors[:]
            return True
        
        i = pos // (n - 1)  # This is not quite right for lexicographic ordering
        # Let's use a different approach: iterate through edges in order
        # Edge pos corresponds to (i,j) where:
        # pos = i*(2*n-i-1)//2 + (j-i-1) for i<j
        # Let's just precompute the edge list
        
        # Actually, let's restructure: use a flat edge list
        pass
    
    # Let's use a cleaner iterative approach with a stack
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
    
    # Use recursive backtracking with sys.setrecursionlimit
    sys.setrecursionlimit(100000)
    
    def _bt(pos):
        if state['found']:
            return True
        if time.time() > deadline:
            return False
        if state['node_count'] > max_depth:
            return False
        
        if pos == num_edges:
            state['found'] = True
            state['solution'] = colors[:]
            return True
        
        state['node_count'] += 1
        i, j = edge_list[pos]
        
        for c in range(4):
            # Try color c for edge (i,j)
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            colors[pos] = c
            
            # Check if this creates a C6
            if not _check_c6_after_add(c, i, j):
                if _bt(pos + 1):
                    return True
            
            # Undo
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            colors[pos] = -1
        
        return False
    
    _bt(0)
    
    if state['found']:
        return state['solution']
    return None

def _local_search(n, time_limit=10.0):
    """Local search to find valid coloring."""
    random.seed(42)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    adj = _build_adjacency(colors, n)
    current_c6s = _count_c6s(adj, n)
    
    if current_c6s == 0:
        return colors
    
    best_colors = colors[:]
    best_c6s = current_c6s
    
    temp = 1.0
    step_size = 0.999
    
    for _ in range(100000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors
            
        edge_idx = random.randint(0, num_edges - 1)
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        
        colors[edge_idx] = new_color
        adj = _build_adjacency(colors, n)
        new_c6s = _count_c6s(adj, n)
        
        delta = new_c6s - current_c6s
        if delta < 0 or (random.random() < np.exp(-delta / temp)):
            current_c6s = new_c6s
            if new_c6s < best_c6s:
                best_c6s = new_c6s
                best_colors = colors[:]
        else:
            colors[edge_idx] = old_color
        
        temp *= step_size
        if temp < 0.01:
            break
    
    if best_c6s == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # First, try the direct Z_17 construction
    result = _construct_z17()
    if result is not None:
        return result
    
    # Try backtracking for n=18 (limited time)
    if time.time() < deadline - 30.0:
        bt_deadline = min(deadline - 25.0, start_time + 30.0)
        colors = _backtracking_search(18, bt_deadline, max_depth=500000)
        if colors is not None:
            if _is_valid(colors, 18):
                return {'n': 18, 'colors': colors}
    
    # Try backtracking for n=17
    if time.time() < deadline - 20.0:
        bt_deadline = min(deadline - 15.0, start_time + 40.0)
        colors = _backtracking_search(17, bt_deadline, max_depth=1000000)
        if colors is not None:
            if _is_valid(colors, 17):
                return {'n': 17, 'colors': colors}
    
    # Try local search for n=17
    if time.time() < deadline - 15.0:
        colors = _local_search(17, time_limit=min(20.0, deadline - time.time() - 10.0))
        if colors is not None:
            if _is_valid(colors, 17):
                return {'n': 17, 'colors': colors}
    
    # Try backtracking for n=16
    if time.time() < deadline - 15.0:
        bt_deadline = min(deadline - 10.0, start_time + 50.0)
        colors = _backtracking_search(16, bt_deadline, max_depth=1000000)
        if colors is not None:
            if _is_valid(colors, 16):
                return {'n': 16, 'colors': colors}
    
    # Try local search for n=16
    if time.time() < deadline - 10.0:
        colors = _local_search(16, time_limit=min(15.0, deadline - time.time() - 5.0))
        if colors is not None:
            if _is_valid(colors, 16):
                return {'n': 16, 'colors': colors}
    
    # Try local search for n=15
    if time.time() < deadline - 10.0:
        colors = _local_search(15, time_limit=min(10.0, deadline - time.time() - 5.0))
        if colors is not None:
            if _is_valid(colors, 15):
                return {'n': 15, 'colors': colors}
    
    # Try smaller n as fallback
    for n in range(14, 3, -1):
        if time.time() > deadline - 5.0:
            break
        colors = _local_search(n, time_limit=5.0)
        if colors is not None:
            if _is_valid(colors, n):
                return {'n': n, 'colors': colors}
    
    # Absolute fallback: n=4
    random.seed(42)
    colors = [random.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}
