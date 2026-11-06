import time
import random
from itertools import combinations

def _has_c6(adj_masks, n):
    """Check if any color class contains a C6 using bitmask adjacency.
    adj_masks[col] is a list of n integers, each a bitmask of neighbors in that color."""
    for col in range(4):
        am = adj_masks[col]
        # For C6: v0-v1-v2-v3-v4-v5-v0
        # v0 and v3 are non-adjacent. Paths of length 3 from v0 to v3.
        # For each non-edge (a,b), find length-3 paths a-x-y-b.
        # If two such paths share no internal vertices, we have C6.
        for a in range(n):
            for b in range(a + 1, n):
                if am[a] & (1 << b):
                    continue  # adjacent, skip
                # Find all length-3 paths from a to b: a-x-y-b
                # x must be neighbor of a, y must be neighbor of b, x-y must be edge
                paths = []
                # x: neighbors of a
                x_mask = am[a]
                while x_mask:
                    x_bit = x_mask & (-x_mask)
                    x = x_bit.bit_length() - 1
                    x_mask ^= x_bit
                    if x == b:
                        continue
                    # y: neighbors of b that are also neighbors of x
                    y_mask = am[b] & am[x]
                    # y must not be a or b
                    y_mask &= ~(1 << a)
                    y_mask &= ~(1 << b)
                    while y_mask:
                        y_bit = y_mask & (-y_mask)
                        y = y_bit.bit_length() - 1
                        y_mask ^= y_bit
                        paths.append((x, y))
                # Check if any two paths are internally disjoint
                if len(paths) >= 2:
                    for i in range(len(paths)):
                        for j in range(i + 1, len(paths)):
                            x1, y1 = paths[i]
                            x2, y2 = paths[j]
                            if x1 != x2 and x1 != y2 and y1 != x2 and y1 != y2:
                                return True
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    def time_check():
        return time.time() > deadline

    # Precompute a known valid n=17 coloring using Z_17 construction
    # Vertices 0..16, color edge (i,j) by (i-j) mod 17
    # Need partition of {1,...,16} into 4 sets of 4
    
    # Let's use a known valid partition for Z_17:
    # After searching, a valid partition is:
    # Color 0: {1, 2, 4, 8}  (powers of 2 mod 17)
    # Color 1: {3, 6, 12, 9} (powers of 3 mod 17: 3,9,10,13,5,15,11,16,14,8,7,4,12,2,6,1)
    # Actually let me just try random partitions and verify
    
    def build_adj_masks(colors, n):
        """Build adjacency masks from color list in lex order."""
        adj_masks = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                col = colors[idx]
                adj_masks[col][i] |= (1 << j)
                adj_masks[col][j] |= (1 << i)
                idx += 1
        return adj_masks

    def verify(colors, n):
        adj_masks = build_adj_masks(colors, n)
        return not _has_c6(adj_masks, n)

    # Try to find a valid n=17 coloring via Z_17 difference set approach
    # Partition {1,...,16} into 4 sets of 4
    best_n = 1
    best_colors = [0]
    
    # Start with n=17, try Z_17 construction
    p = 17
    # Try all partitions of {1,...,16} into 4 sets of 4
    # That's 16!/(4!^4 * 4!) which is huge. Use random search.
    
    found_17 = False
    for attempt in range(200):
        if time_check():
            break
        random.seed(attempt * 7919 + 42)
        elems = list(range(1, 17))
        random.shuffle(elems)
        partition = [elems[i*4:(i+1)*4] for i in range(4)]
        
        # Build coloring: color of edge (i,j) is the set containing (i-j) mod 17 or (j-i) mod 17
        colors = []
        valid_partition = True
        for i in range(p):
            for j in range(i + 1, p):
                diff = (i - j) % p
                if diff == 0:
                    diff = p  # shouldn't happen
                # Find which set contains diff
                col_found = -1
                for c in range(4):
                    if diff in partition[c]:
                        col_found = c
                        break
                if col_found == -1:
                    valid_partition = False
                    break
                colors.append(col_found)
            if not valid_partition:
                break
        
        if not valid_partition:
            continue
        
        if verify(colors, p):
            best_n = p
            best_colors = colors
            found_17 = True
            break
    
    if found_17:
        # Try to extend to n=18
        for target_n in [18, 19, 20]:
            if time_check():
                break
            n = target_n
            num_edges = n * (n - 1) // 2
            
            # Start from the n=17 coloring and add vertex n-1
            # Try random colorings for edges involving the new vertex
            found = False
            for attempt in range(200):
                if time_check():
                    break
                random.seed(attempt * 104729 + target_n * 31)
                
                # Copy existing edges
                colors = best_colors[:]
                # Add edges from new vertex (n-1) to vertices 0..n-2
                for i in range(n - 1):
                    colors.append(random.randint(0, 3))
                
                if verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
            
            if found:
                continue
            else:
                break
    
    # If we didn't find n=17, try smaller n with random search
    if best_n < 17:
        for target_n in range(16, max(best_n, 8), -1):
            if time_check():
                break
            n = target_n
            num_edges = n * (n - 1) // 2
            for attempt in range(100):
                if time_check():
                    break
                random.seed(attempt * 7919 + n)
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                if verify(colors, n):
                    best_n = n
                    best_colors = colors
                    break
            if best_n >= target_n:
                break
    
    # Ensure we have at least a valid small coloring
    if best_n < 2:
        best_n = 2
        best_colors = [0]
    
    return {'n': best_n, 'colors': best_colors}
