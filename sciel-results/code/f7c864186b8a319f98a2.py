"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _get_idx(i, j, n):
    """Get index in the flat colors array for edge (i,j) where i<j."""
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _validate_coloring(n, colors):
    """Validate that no color class contains a C4."""
    for i in range(n):
        for j in range(i + 1, n):
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

def _greedy_cost_minimize(n, deadline, seed=42):
    """
    Greedy coloring with cost minimization:
    - Track for each pair (u,v) and color c, the number of common neighbors
    - When assigning color c to edge (i,j), reject if it would create a C4 (>=2 common neighbors)
    - Among valid colors, prefer the one that minimizes the number of new 'at-risk' pairs
      (pairs that go from 0 to 1 common neighbor in that color)
    """
    rng = random.Random(seed)
    n_edges = n * (n - 1) // 2
    colors = [-1] * n_edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # common_neighbors[color][u][v] = number of common neighbors of u and v in color
    # Use a 2D list for each color: common[color][u][v]
    common = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    for idx, (i, j) in enumerate(edges):
        if time.time() > deadline:
            return None
        
        best_color = -1
        best_cost = float('inf')
        valid_colors = []
        
        for c in range(5):
            # Check if this color would create a C4
            if common[c][i][j] >= 2:
                continue
            
            # Count how many new at-risk pairs this would create
            cost = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                # If edge (i,k) has color c and edge (j,k) has color c, 
                # then (i,j) gets one more common neighbor
                # But we need to check what happens to pairs (i,k) and (j,k)
                # Actually, adding edge (i,j) with color c affects:
                # - common[c][i][k] for all k where edge (j,k) is color c
                # - common[c][j][k] for all k where edge (i,k) is color c
                
                # For pair (i, k): if edge (j,k) is color c, then common[c][i][k] increases by 1
                # If it was 0, it becomes 1 (at-risk) -> cost +1
                # If it was 1, it becomes 2 (C4) -> invalid, but we already checked common[c][i][j]
                
                idx_jk = _get_idx(j, k, n)
                if colors[idx_jk] == c:
                    if common[c][i][k] == 0:
                        cost += 1
                    elif common[c][i][k] >= 1:
                        # This would make it 2, creating a C4
                        cost = float('inf')
                        break
                
                idx_ik = _get_idx(i, k, n)
                if colors[idx_ik] == c:
                    if common[c][j][k] == 0:
                        cost += 1
                    elif common[c][j][k] >= 1:
                        # This would make it 2, creating a C4
                        cost = float('inf')
                        break
            
            if cost < float('inf'):
                valid_colors.append((cost, c))
        
        if not valid_colors:
            return None
        
        # Sort by cost, break ties randomly
        valid_colors.sort()
        # Pick from the best-cost colors with some randomness
        min_cost = valid_colors[0][0]
        candidates = [c for (cost, c) in valid_colors if cost == min_cost]
        best_color = rng.choice(candidates)
        
        colors[idx] = best_color
        c = best_color
        
        # Update common neighbor counts
        for k in range(n):
            if k == i or k == j:
                continue
            idx_ik = _get_idx(i, k, n)
            idx_jk = _get_idx(j, k, n)
            
            if colors[idx_ik] == c:
                common[c][j][k] += 1
                common[c][k][j] += 1
            if colors[idx_jk] == c:
                common[c][i][k] += 1
                common[c][k][i] += 1
    
    # Final validation
    if _validate_coloring(n, colors):
        return colors
    return None

def _construct_cyclic(n):
    """Construct a cyclic coloring of K_n using distances."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            d = min(d, n - d)
            colors.append(d % 5)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Start with a small known-good coloring
    best_n = 1
    best_colors = [0]
    
    # Try increasing n with greedy cost minimization
    for n in range(2, 29):
        if time.time() > deadline - 10:
            break
        
        found = None
        # Try multiple seeds for diversity
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline - 5:
                break
            
            result = _greedy_cost_minimize(n, deadline - 2, seed)
            if result is not None:
                found = result
                break
        
        if found is not None:
            best_n = n
            best_colors = found
        else:
            # If we can't find a valid coloring for this n, try one less
            # and break since larger n will likely also fail
            if n > 2:
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
