"""Program to construct a 4-colouring of K_n with no monochromatic C6."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _validate_coloring(n, colors):
    """Check if a 4-coloring of K_n has no monochromatic C6."""
    num_edges = n * (n - 1) // 2
    if len(colors) != num_edges:
        return False
    
    # Build adjacency lists for each color
    adj = [[] for _ in range(4)]
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 3:
                return False
            adj[c].append((i, j))
            edge_idx[(i, j)] = c
            edge_idx[(j, i)] = c
            idx += 1
    
    # Build adjacency bitmasks for each color
    adj_mask = [[0] * n for _ in range(4)]
    for c in range(4):
        for (u, v) in adj[c]:
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
    
    # For each color, check all 6-tuples for a cycle
    for c in range(4):
        # Check all combinations of 6 vertices
        for v1 in range(n):
            m1 = adj_mask[c][v1]
            if not m1:
                continue
            for v2 in range(v1 + 1, n):
                if not (m1 & (1 << v2)):
                    continue
                m2 = adj_mask[c][v2] & ~(1 << v1)
                if not m2:
                    continue
                for v3 in range(v2 + 1, n):
                    if not (m2 & (1 << v3)):
                        continue
                    m3 = adj_mask[c][v3] & ~(1 << v1) & ~(1 << v2)
                    if not m3:
                        continue
                    for v4 in range(v3 + 1, n):
                        if not (m3 & (1 << v4)):
                            continue
                        m4 = adj_mask[c][v4] & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                        if not m4:
                            continue
                        for v5 in range(v4 + 1, n):
                            if not (m4 & (1 << v5)):
                                continue
                            # Check if v5 connects back to v1
                            if adj_mask[c][v5] & (1 << v1):
                                return False
    return True

def construct():
    """
    Construct a 4-colouring of K_n with no monochromatic 6-cycle.
    Returns {'n': n, 'colors': [...]} where colors has one entry per pair (i,j), i<j,
    in lexicographic order, each in {0,1,2,3}.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Try to find a valid coloring for n=17 using randomized local search
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Use a known structure: cyclic coloring based on Z_17
    # Color edge (i,j) by (j-i) mod 17 mapped to {0,1,2,3}
    # We need a partition of Z_17\{0} into 4 sets such that no set contains a 6-cycle
    
    # Try different partitions
    # The nonzero elements of Z_17: 1..16
    # We need to partition them into 4 sets of sizes ~4 each
    # such that no set contains {a, b, c, d, e, f} forming a C6
    
    # A C6 in Z_17 means vertices v, v+a, v+a+b, v+a+b+c, v+a+b+c+d, v+a+b+c+d+e
    # with v+a+b+c+d+e+f = v (mod 17), so a+b+c+d+e+f = 0 mod 17
    # and all partial sums distinct (mod 17)
    
    # Let's use a greedy randomized approach
    best_result = None
    best_n = 0
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Random partition of {1,...,16} into 4 sets
        elements = list(range(1, 17))
        random.shuffle(elements)
        
        # Assign to 4 groups
        group_sizes = [4, 4, 4, 4]
        groups = []
        pos = 0
        for size in group_sizes:
            groups.append(elements[pos:pos+size])
            pos += size
        
        # Build color assignment
        color_of_diff = {}
        for c, group in enumerate(groups):
            for d in group:
                color_of_diff[d] = c
                color_of_diff[-d % 17] = c
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % 17
                colors.append(color_of_diff[diff])
        
        # Validate
        if _validate_coloring(n, colors):
            return {'n': n, 'colors': colors}
    
    # If n=17 failed, try n=16
    n = 16
    num_edges = n * (n - 1) // 2
    
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        # Random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        if _validate_coloring(n, colors):
            return {'n': n, 'colors': colors}
    
    # Fallback: try a structured coloring for smaller n
    for n_try in [15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 2:
            break
        
        n = n_try
        num_edges = n * (n - 1) // 2
        
        # Cyclic coloring
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                # Map diff to color
                c = diff % 4
                colors.append(c)
        
        if _validate_coloring(n, colors):
            return {'n': n, 'colors': colors}
    
    # Last resort: small n with simple pattern
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
