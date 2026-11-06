import time
import random
import numpy as np

DEADLINE_MARGIN = 5.0

def _has_c6_bitset(adj, n):
    """
    Check if the graph defined by adjacency masks 'adj' contains a 6-cycle.
    adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists.
    n is the number of vertices.
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
                        
                        # Check if there exists v5 such that:
                        # v5 in N(v4) AND v5 in N(v0)
                        # v5 not in {v0, v1, v2, v3, v4}
                        mask_v5 = adj[v4] & adj[v0]
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
                        
                        if mask_v5:
                            return True
    return False

def _build_adj(colors, n):
    """Build adjacency bitmasks for each color class."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_c6s(colors, n):
    """
    Count the number of monochromatic C6s.
    Returns a list of 4 counts, one per color.
    """
    adj = _build_adj(colors, n)
    counts = [0] * 4
    for c in range(4):
        # Reuse the logic but count instead of return
        # To save time, we can just check existence for the first few, 
        # but for local search we need the count to guide the search.
        # However, counting ALL C6s is more expensive than checking existence.
        # For n=18, the number of C6s might be small.
        # Let's just check existence for now and use a penalty based on 
        # "distance" if we can't count easily.
        # Actually, let's just return 1 if any C6 exists, 0 otherwise.
        # This makes it a 0-1 loss function, which is hard for local search.
        # Better: Count the number of "almost C6s" or just use the existence.
        # Let's stick to existence for speed, but use random restarts.
        if _has_c6_bitset(adj[c], n):
            counts[c] = 1
        else:
            counts[c] = 0
    return counts

def _is_valid(colors, n):
    adj = _build_adj(colors, n)
    for c in range(4):
        if _has_c6_bitset(adj[c], n):
            return False
    return True

def _find_valid_17():
    """
    Find a valid 4-coloring of K_17.
    Tries a few deterministic patterns first, then random search.
    """
    n = 17
    # Pattern 1: The one from the current program
    colors = [((j - i) * 3 + i) % 4 for i in range(n) for j in range(i + 1, n)]
    if _is_valid(colors, n):
        return colors

    # Pattern 2: Another simple pattern
    colors = [((j - i) * 7 + i * 2) % 4 for i in range(n) for j in range(i + 1, n)]
    if _is_valid(colors, n):
        return colors

    # Random search
    random.seed(42)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        if _is_valid(colors, n):
            return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + (120.0 - DEADLINE_MARGIN)
    
    # Step 1: Get a valid baseline for n=17
    base_colors_17 = _find_valid_17()
    if base_colors_17 is None:
        # Fallback: Return n=1 if we can't even find 17 (should not happen)
        return {'n': 1, 'colors': []}
    
    best_n = 17
    best_colors = base_colors_17
    
    # Step 2: Try to extend to n=18
    # We will perform a local search on the n=18 graph.
    # Initialize with the n=17 coloring + random edges for vertex 17.
    
    n_target = 18
    edges_17 = n_target * (n_target - 1) // 2
    edges_16 = 17 * 16 // 2
    
    # Pre-calculate indices for the new edges
    # Edges are stored in lexicographic order (i, j) with i < j.
    # For n=18, the new edges are (0,17), (1,17), ..., (16,17).
    # These are the last 17 entries in the color list.
    
    # We will run multiple restarts of local search.
    random.seed(123)
    
    while time.time() < deadline:
        # Initialize n=18 coloring
        colors_18 = base_colors_17.copy()
        # Add random colors for the new edges
        new_edges_colors = [random.randint(0, 3) for _ in range(17)]
        colors_18.extend(new_edges_colors)
        
        # Local search
        current_penalty = sum(_count_c6s(colors_18, n_target))
        
        if current_penalty == 0:
            return {'n': 18, 'colors': colors_18}
        
        # Simulated Annealing / Hill Climbing
        # We will try to flip a random edge color.
        # To make it faster, we can focus on edges incident to vertex 17,
        # as they are the most likely to cause issues.
        
        improved = False
        for _ in range(1000): # Max steps per restart
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            # Bias towards edges incident to 17
            if random.random() < 0.5:
                # Edge (i, 17)
                i = random.randint(0, 16)
                # Find index of (i, 17) in the list
                # Index for (i, j) is i*n - i*(i+1)/2 + (j-i-1) ?
                # Let's use a helper or precomputed map.
                # For n=18, index of (i, 17) is:
                # Sum of edges for k < i: k*(18-1) - k*(k+1)/2? No.
                # Total edges before row i is sum_{k=0}^{i-1} (18 - 1 - k) = sum_{k=0}^{i-1} (17 - k)
                # = 17*i - i*(i-1)/2
                # Then offset by (17 - i - 1) = 16 - i
                # So index = 17*i - i*(i-1)//2 + (16 - i)
                idx = 17 * i - i * (i - 1) // 2 + (16 - i)
            else:
                idx = random.randint(0, edges_17 - 1)
            
            old_color = colors_18[idx]
            new_color = (old_color + random.randint(1, 3)) % 4
            
            if old_color == new_color:
                continue
                
            colors_18[idx] = new_color
            new_penalty = sum(_count_c6s(colors_18, n_target))
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                improved = True
                if current_penalty == 0:
                    return {'n': 18, 'colors': colors_18}
            else:
                # Revert
                colors_18[idx] = old_color
        
        # If no improvement in this restart, try a new random initialization
        # (The loop continues, creating a new colors_18)

    # If n=18 not found, return n=17
    return {'n': 17, 'colors': base_colors_17}
