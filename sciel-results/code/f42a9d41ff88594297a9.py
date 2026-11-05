"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.
Combines Z_17 cyclic construction with bit-parallel C6 detection and local search.
"""
import time
import random
import sys

DEADLINE = 140.0  # 10s safety margin before 150s kill

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

def _count_c6s_fast(adj, n):
    """Count total number of C6s across all 4 color classes (approximate, for local search)."""
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
                            total += bin(mask_v5).count('1')
    return total

def _edge_index(n, i, j):
    """Get the index in the colors array for edge (i,j) where i<j."""
    # Number of edges before row i: i*(i-1)//2
    # Position in row i: j - i - 1
    return i * (i - 1) // 2 + (j - i - 1)

def _local_search(colors, n, time_limit):
    """Local search to minimize C6 count. Returns improved coloring or None."""
    start = time.time()
    best_colors = colors[:]
    best_count = _count_c6s_fast(_build_adjacency(best_colors, n), n)
    
    if best_count == 0:
        return best_colors
    
    adj = _build_adjacency(best_colors, n)
    num_edges = n * (n - 1) // 2
    
    while time.time() - start < time_limit:
        # Pick a random edge to flip
        ei = random.randint(0, num_edges - 1)
        # Convert edge index to (i,j)
        i = 0
        while i * (i - 1) // 2 <= ei:
            if i * (i - 1) // 2 + (i - 1) > ei:
                break
            i += 1
        # Actually, let me compute properly
        # Find i such that i*(i-1)//2 <= ei < (i+1)*i//2
        i = 1
        while (i + 1) * i // 2 <= ei:
            i += 1
        # Now i is such that i*(i-1)//2 <= ei < (i+1)*i//2
        # Wait, let me redo. For vertex i, edges are (i, i+1), ..., (i, n-1)
        # Count before vertex i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # This is getting complicated. Let me just iterate.
        
        # Simpler: precompute mapping
        # For now, just pick random i<j
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        idx = _edge_index(n, i, j)
        
        old_color = best_colors[idx]
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        
        # Update adjacency
        adj[old_color][i] &= ~(1 << j)
        adj[old_color][j] &= ~(1 << i)
        adj[new_color][i] |= (1 << j)
        adj[new_color][j] |= (1 << i)
        best_colors[idx] = new_color
        
        new_count = _count_c6s_fast(adj, n)
        
        if new_count <= best_count:
            best_count = new_count
            if best_count == 0:
                return best_colors
        else:
            # Revert
            adj[new_color][i] &= ~(1 << j)
            adj[new_color][j] &= ~(1 << i)
            adj[old_color][i] |= (1 << j)
            adj[old_color][j] |= (1 << i)
            best_colors[idx] = old_color
    
    return best_colors

def _try_cyclic_17():
    """Try to find a valid Z_17 cyclic coloring."""
    n = 17
    # In Z_17, edge (i,j) has color f((j-i) mod 17)
    # We need to partition {1,...,16} into 4 sets of 4
    # such that no color class contains a C6.
    
    # A C6 in the cyclic group: differences d1,...,d6 with sum=0 mod 17,
    # all in same color set, and partial sums distinct.
    
    # Let's try a known good partition. One approach: use the fact that
    # the multiplicative group Z_17* has order 16. We can use cosets of
    # subgroups.
    
    # Z_17* is cyclic of order 16. Let g=3 be a primitive root mod 17.
    # Powers of 3: 3^0=1, 3^1=3, 3^2=9, 3^3=27=10, 3^4=30=13, 3^5=39=5,
    # 3^6=15, 3^7=45=11, 3^8=33=16, 3^9=48=14, 3^10=42=8, 3^11=24=7,
    # 3^12=21=4, 3^13=12, 3^14=36=2, 3^15=6, 3^16=18=1
    # So the powers are: 1,3,9,10,13,5,15,11,16,14,8,7,4,12,2,6
    
    # Partition into 4 sets of 4 consecutive powers (cosets of subgroup of order 4):
    # H = {1, 3^4=13, 3^8=16, 3^12=4} = {1, 4, 13, 16}
    # 3H = {3, 12, 2, 6} (3*1=3, 3*4=12, 3*13=39=5? wait)
    
    # Let me just compute: subgroup of order 4 generated by 3^4=13.
    # 13^0=1, 13^1=13, 13^2=169=169 mod 17 = 169-10*17=169-170=-1=16, 13^3=13*16=208=208-12*17=208-204=4, 13^4=4*13=52=52-3*17=52-51=1
    # So H = {1, 13, 16, 4}
    
    # Cosets: 1*H={1,13,16,4}, 3*H={3,39=5,48=14,12}, 9*H={9,117=15,144=8,36=2}, 10*H={10,130=11,160=7,40=6}
    # Check: {1,13,16,4} ∪ {3,5,14,12} ∪ {9,15,8,2} ∪ {10,11,7,6} = all of 1..16? 
    # 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 ✓
    
    sets = [
        [1, 4, 13, 16],   # color 0
        [3, 5, 12, 14],   # color 1
        [2, 8, 9, 15],    # color 2
        [6, 7, 10, 11]    # color 3
    ]
    
    # Build coloring
    color_map = {}
    for c, s in enumerate(sets):
        for d in s:
            color_map[d] = c
            color_map[-d % 17] = c  # symmetric
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % 17
            colors.append(color_map[d])
    
    if _is_valid(colors, n):
        return colors
    
    # Try other partitions via local search on the partition
    # This is complex, so let's just try a few random partitions
    for _ in range(100):
        # Random partition of {1,...,16} into 4 sets of 4
        elems = list(range(1, 17))
        random.shuffle(elems)
        sets = [elems[i*4:(i+1)*4] for i in range(4)]
        
        color_map = {}
        for c, s in enumerate(sets):
            for d in s:
                color_map[d] = c
                color_map[-d % 17] = c
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % 17
                colors.append(color_map[d])
        
        if _is_valid(colors, n):
            return colors
    
    return None

def construct():
    start_time = time.time()
    
    # Try to get a valid coloring for n=17
    colors_17 = _try_cyclic_17()
    
    if colors_17 is not None:
        best_n = 17
        best_colors = colors_17
    else:
        # Fallback: use the n=9 coloring
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to extend to n=18, 19 using local search
    for target_n in [18, 19]:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        num_edges = target_n * (target_n - 1) // 2
        
        # Initialize by extending the best coloring
        # For new vertex target_n-1, assign colors randomly, then refine
        init_colors = best_colors[:]
        for i in range(target_n - 1):
            init_colors.append(random.randint(0, 3))
        
        # Local search with time budget
        time_budget = min(20, DEADLINE - 5 - (time.time() - start_time))
        if time_budget > 1:
            improved = _local_search(init_colors, target_n, time_budget)
            if improved is not None and _is_valid(improved, target_n):
                best_n = target_n
                best_colors = improved
                continue
            elif improved is not None:
                # Use improved as base for further search
                init_colors = improved
        
        # If we have time, try more restarts
        if time.time() - start_time < DEADLINE - 15:
            for _ in range(5):
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                init_colors = best_colors[:]
                for i in range(target_n - 1):
                    init_colors.append(random.randint(0, 3))
                
                time_budget = min(15, DEADLINE - 5 - (time.time() - start_time))
                if time_budget > 1:
                    improved = _local_search(init_colors, target_n, time_budget)
                    if improved is not None and _is_valid(improved, target_n):
                        best_n = target_n
                        best_colors = improved
                        break
    
    return {'n': best_n, 'colors': best_colors}
