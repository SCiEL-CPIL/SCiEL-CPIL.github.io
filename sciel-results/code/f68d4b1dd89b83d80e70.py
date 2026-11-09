"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_c4(coloring, n, i, j, color):
    """Check if adding edge (i,j) with given color creates a C4 in that color class."""
    # A C4 is formed if i and j already have >=2 common neighbors in this color
    common = 0
    for k in range(n):
        if k == i or k == j:
            continue
        idx_ik = _get_idx(i, k, n)
        idx_jk = _get_idx(j, k, n)
        
        if coloring[idx_ik] == color and coloring[idx_jk] == color:
            common += 1
            if common >= 2:
                return False  # C4 would be formed
    return True

def _get_idx(i, j, n):
    """Get index in the flat colors array for edge (i,j) where i<j."""
    if i > j:
        i, j = j, i
    # Number of edges before vertex i: sum_{v=0}^{i-1} (n-1-v) = i*(n-1) - i*(i-1)//2
    # = i*n - i - i*(i-1)//2 = i*n - i - (i^2-i)//2 = i*n - i - i^2/2 + i/2
    # = i*n - i/2 - i^2/2 = i*(2n - i - 1)//2
    # Then within vertex i's edges, edge (i,j) is at position (j-i-1)
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _construct_cyclic(n):
    """Construct a cyclic coloring of K_n using distances."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            d = min(d, n - d)
            # Map distance to color: use d mod 5, but ensure we use all 5 colors
            colors.append(d % 5)
    return colors

def _validate_coloring(n, colors):
    """Validate that no color class contains a C4."""
    # For each pair of vertices, check if they have >=2 common neighbors in any color
    for i in range(n):
        for j in range(i + 1, n):
            # Count common neighbors in each color
            common_counts = [0] * 5
            for k in range(n):
                if k == i or k == j:
                    continue
                idx_ik = _get_idx(i, k, n)
                idx_jk = _get_idx(j, k, n)
                c_ik = colors[idx_ik]
                c_jk = colors[idx_jk]
                if c_ik == c_jk:
                    common_counts[c_ik] += 1
            
            if any(c >= 2 for c in common_counts):
                return False
    return True

def _try_backtracking(n, deadline):
    """Try to find a valid 5-coloring of K_n using backtracking."""
    n_edges = n * (n - 1) // 2
    colors = [-1] * n_edges
    
    # Precompute adjacency: for each vertex, list of other vertices
    # Use symmetry: process edges in order
    
    def backtrack(edge_idx):
        if time.time() > deadline:
            return None
        if edge_idx == n_edges:
            return colors[:]
        
        i = edge_idx
        # Decode edge_idx to (i,j)
        # This is tricky; let's precompute
        pass
    
    # Simpler: iterate through edges in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n_edges:
            return colors[:]
        
        i, j = edges[idx]
        for color in range(5):
            if _check_c4(colors, n, i, j, color):
                colors[idx] = color
                result = backtrack(idx + 1)
                if result is not None:
                    return result
        colors[idx] = -1
        return None
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with the known good cyclic construction for n=11
    best_n = 11
    best_colors = _construct_cyclic(11)
    
    # Try to find larger valid colorings
    for n in range(12, 18):  # Try up to n=17 to stay within time
        if time.time() > deadline - 30:
            break
        
        # Try multiple random greedy attempts
        found = None
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            random.seed(42 + attempt * 1000)
            colors = [-1] * (n * (n - 1) // 2)
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            
            success = True
            for idx, (i, j) in enumerate(edges):
                valid_colors = []
                for c in range(5):
                    if _check_c4(colors, n, i, j, c):
                        valid_colors.append(c)
                
                if not valid_colors:
                    success = False
                    break
                
                colors[idx] = random.choice(valid_colors)
            
            if success and _validate_coloring(n, colors):
                found = colors
                break
        
        if found is not None:
            best_n = n
            best_colors = found
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
