"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def _edge_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i*(n-1) - i*(i-1)//2 + (j - i - 1)
    
    def _verify_full(colors, n):
        """Full verification that no color class contains a C6."""
        if n < 6:
            return True
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
        
        for c in range(4):
            for subset in combinations(range(n), 6):
                # Check all 6-cycles on this subset
                for perm in [(0,1,2,3,4,5),(0,1,2,3,5,4),(0,1,2,4,3,5),(0,1,2,4,5,3),
                            (0,1,3,2,4,5),(0,1,3,2,5,4),(0,1,3,4,2,5),(0,1,3,4,5,2),
                            (0,1,3,5,2,4),(0,1,3,5,4,2),(0,1,4,2,3,5),(0,1,4,2,5,3),
                            (0,1,4,3,2,5),(0,1,4,3,5,2),(0,1,4,5,2,3),(0,1,4,5,3,2),
                            (0,1,5,2,3,4),(0,1,5,2,4,3),(0,1,5,3,2,4),(0,1,5,3,4,2),
                            (0,1,5,4,2,3),(0,1,5,4,3,2),(0,2,1,3,4,5),(0,2,1,3,5,4),
                            (0,2,1,4,3,5),(0,2,1,4,5,3),(0,2,1,5,3,4),(0,2,1,5,4,3),
                            (0,2,3,1,4,5),(0,2,3,1,5,4),(0,2,3,4,1,5),(0,2,3,4,5,1),
                            (0,2,3,5,1,4),(0,2,3,5,4,1),(0,2,4,1,3,5),(0,2,4,1,5,3),
                            (0,2,4,3,1,5),(0,2,4,3,5,1),(0,2,4,5,1,3),(0,2,4,5,3,1),
                            (0,2,5,1,3,4),(0,2,5,1,4,3),(0,2,5,3,1,4),(0,2,5,3,4,1),
                            (0,2,5,4,1,3),(0,2,5,4,3,1),(0,3,1,2,4,5),(0,3,1,2,5,4),
                            (0,3,1,4,2,5),(0,3,1,4,5,2),(0,3,1,5,2,4),(0,3,1,5,4,2),
                            (0,3,2,1,4,5),(0,3,2,1,5,4),(0,3,2,4,1,5),(0,3,2,4,5,1),
                            (0,3,2,5,1,4),(0,3,2,5,4,1),(0,3,4,1,2,5),(0,3,4,1,5,2),
                            (0,3,4,2,1,5),(0,3,4,2,5,1),(0,3,4,5,1,2),(0,3,4,5,2,1),
                            (0,3,5,1,2,4),(0,3,5,1,4,2),(0,3,5,2,1,4),(0,3,5,2,4,1),
                            (0,3,5,4,1,2),(0,3,5,4,2,1),(0,4,1,2,3,5),(0,4,1,2,5,3),
                            (0,4,1,3,2,5),(0,4,1,3,5,2),(0,4,1,5,2,3),(0,4,1,5,3,2),
                            (0,4,2,1,3,5),(0,4,2,1,5,3),(0,4,2,3,1,5),(0,4,2,3,5,1),
                            (0,4,2,5,1,3),(0,4,2,5,3,1),(0,4,3,1,2,5),(0,4,3,1,5,2),
                            (0,4,3,2,1,5),(0,4,3,2,5,1),(0,4,3,5,1,2),(0,4,3,5,2,1),
                            (0,4,5,1,2,3),(0,4,5,1,3,2),(0,4,5,2,1,3),(0,4,5,2,3,1),
                            (0,4,5,3,1,2),(0,4,5,3,2,1),(0,5,1,2,3,4),(0,5,1,2,4,3),
                            (0,5,1,3,2,4),(0,5,1,3,4,2),(0,5,1,4,2,3),(0,5,1,4,3,2),
                            (0,5,2,1,3,4),(0,5,2,1,4,3),(0,5,2,3,1,4),(0,5,2,3,4,1),
                            (0,5,2,4,1,3),(0,5,2,4,3,1),(0,5,3,1,2,4),(0,5,3,1,4,2),
                            (0,5,3,2,1,4),(0,5,3,2,4,1),(0,5,3,4,1,2),(0,5,3,4,2,1),
                            (0,5,4,1,2,3),(0,5,4,1,3,2),(0,5,4,2,1,3),(0,5,4,2,3,1),
                            (0,5,4,3,1,2),(0,5,4,3,2,1)]:
                    ok = True
                    for k in range(6):
                        u = subset[perm[k]]
                        w = subset[perm[(k+1)%6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        return False
        return True
    
    def _check_c6_incremental_bitmask(adj_masks, n, v, i, c):
        """Check if adding edge (i,v) with color c creates a C6 in color c.
        adj_masks[c][x] is a bitmask of neighbors of x in color c."""
        mask_v = adj_masks[c][v]
        # For each neighbor d of v in color c (d != i), check if there's a path of length 4 from i to d
        for d in range(n):
            if d == i or d == v:
                continue
            if not (mask_v >> d) & 1:
                continue
            # Check if there's a path i-a-b-e-d of length 4 in G_c without using v
            # a is neighbor of i in c, a != v, a != d
            # b is neighbor of a in c, b != v, b != i, b != d, b != a
            # e is neighbor of b in c, e != v, e != i, e != d, e != a, e != b
            # and e is connected to d in c
            mask_i = adj_masks[c][i]
            mask_d = adj_masks[c][d]
            # Exclude v, i, d from consideration for intermediate nodes
            exclude = (1 << v) | (1 << i) | (1 << d)
            mask_i_filtered = mask_i & ~exclude
            for a in range(n):
                if not (mask_i_filtered >> a) & 1:
                    continue
                mask_a = adj_masks[c][a]
                mask_a_filtered = mask_a & ~exclude & ~(1 << a)
                for b in range(n):
                    if not (mask_a_filtered >> b) & 1:
                        continue
                    mask_b = adj_masks[c][b]
                    mask_b_filtered = mask_b & ~exclude & ~(1 << b)
                    for e in range(n):
                        if not (mask_b_filtered >> e) & 1:
                            continue
                        if (mask_d >> e) & 1:
                            return False
            return True
        return True
    
    def _insert_vertex(adj_masks, n, new_v, deadline):
        """Try to insert a new vertex new_v with colors for edges to 0..new_v-1.
        Returns list of colors or None."""
        # We need to assign colors for edges (i, new_v) for i = 0..new_v-1
        # such that no C6 is created in any color class.
        
        def _search(i, assigned):
            if time.time() > deadline:
                return None
            
            if i == new_v:
                return assigned
            
            # Try each color for edge (i, new_v)
            for c in range(4):
                if _check_c6_incremental_bitmask(adj_masks, new_v, new_v, i, c):
                    # Temporarily add the edge
                    adj_masks[c][i] |= (1 << new_v)
                    adj_masks[c][new_v] |= (1 << i)
                    assigned.append(c)
                    
                    result = _search(i + 1, assigned)
                    if result is not None:
                        return result
                    
                    # Backtrack
                    adj_masks[c][i] &= ~(1 << new_v)
                    adj_masks[c][new_v] &= ~(1 << i)
                    assigned.pop()
            
            return None
        
        return _search(0, [])
    
    # Start with n=1 (no edges)
    n = 1
    adj_masks = [[0] * 20 for _ in range(4)]  # Pre-allocate for up to 20 vertices
    colors = []
    
    # Try to grow from n=1 to n=20
    for target_n in range(2, 21):
        if time.time() > deadline - 10:
            break
        
        # Initialize adj_masks for the new vertex
        for c in range(4):
            adj_masks[c][target_n - 1] = 0
        
        # Try to insert vertex target_n-1
        result = _insert_vertex(adj_masks, target_n - 1, target_n - 1, deadline - 5)
        if result is not None:
            # Add the colors to our list
            for i, c in enumerate(result):
                colors.append(c)
            n = target_n
        else:
            break
    
    # Convert to the required format
    # We need colors in lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
    final_colors = [0] * (n*(n-1)//2)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            final_colors[idx] = colors[idx]
            idx += 1
    
    # Verify
    if n >= 6 and not _verify_full(final_colors, n):
        # Fallback
        n = 5
        final_colors = [0] * (n*(n-1)//2)
    
    return {'n': n, 'colors': final_colors}
# EVOLVE-BLOCK-END
