import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin

    def has_mono_k222_fast(n, adj_bitmasks):
        """Fast check for monochromatic K2,2,2.
        adj_bitmasks: list of 2 lists, adj_bitmasks[c][i] is an integer bitmask
        where bit j is set if edge (i,j) has color c.
        """
        for c in range(2):
            bm = adj_bitmasks[c]
            for a in range(n):
                for b in range(a + 1, n):
                    S_mask = bm[a] & bm[b]
                    S_mask &= ~(1 << a)
                    S_mask &= ~(1 << b)
                    if S_mask.bit_count() < 4:
                        continue
                    
                    # Get list of vertices in S
                    verts = []
                    m = S_mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        verts.append(idx)
                        m ^= lsb
                    
                    # Check for K2,2 in induced subgraph on verts
                    lv = len(verts)
                    for i in range(lv):
                        x = verts[i]
                        for j in range(i + 1, lv):
                            y = verts[j]
                            common = bm[x] & bm[y] & S_mask
                            if common.bit_count() >= 2:
                                return True
        return False

    def make_bitmasks(n, colors):
        """Convert flat color list to bitmask arrays."""
        bm = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                bm[c][i] |= (1 << j)
                bm[c][j] |= (1 << i)
                idx += 1
        return bm

    def check_valid(n, colors):
        bm = make_bitmasks(n, colors)
        return not has_mono_k222_fast(n, bm)

    # Try Paley 29 first
    n29 = 29
    squares29 = {x * x % n29 for x in range(1, n29)}
    colors29 = [0 if (j - i) % n29 in squares29 else 1 for i in range(n29) for j in range(i + 1, n29)]
    
    if not check_valid(n29, colors29):
        # Should not happen, but fallback
        n17 = 17
        squares17 = {x * x % n17 for x in range(1, n17)}
        colors17 = [0 if (j - i) % n17 in squares17 else 1 for i in range(n17) for j in range(i + 1, n17)]
        return {'n': n17, 'colors': colors17}
    
    # Try to extend to n=30
    if time.time() < deadline - 30:
        n30 = 30
        best_n = n29
        best_colors = colors29[:]
        
        random.seed(42)
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Randomly assign colors for edges from 29 to 0..28
            new_edges = [random.randint(0, 1) for _ in range(29)]
            full_colors = colors29 + new_edges
            
            if check_valid(n30, full_colors):
                best_n = n30
                best_colors = full_colors
                break
            
            # Local search: iteratively flip edges to reduce violations
            for step in range(5000):
                if time.time() > deadline - 10:
                    break
                
                bm = make_bitmasks(n30, full_colors)
                if not has_mono_k222_fast(n30, bm):
                    best_n = n30
                    best_colors = full_colors
                    break
                
                # Find a violating K2,2,2 and flip one of its edges
                # For efficiency, just flip a random edge
                edge_idx = random.randint(0, len(full_colors) - 1)
                full_colors[edge_idx] = 1 - full_colors[edge_idx]
        
        if best_n == n30:
            return {'n': best_n, 'colors': best_colors}
    
    return {'n': n29, 'colors': colors29}

# EVOLVE-BLOCK-END
