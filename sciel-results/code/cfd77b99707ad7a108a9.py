import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Known solution for n=17 based on F_17
    n = 17
    # Primitive root 3 for F_17
    # Map difference d to color
    # d -> log_3(d) mod 16 -> color = log_3(d) % 4
    
    # Precompute log table for F_17
    log_table = {}
    g = 3
    val = 1
    for k in range(16):
        log_table[val] = k
        val = (val * g) % 17
        
    def get_color(i, j):
        d = (j - i) % 17
        if d == 0:
            return 0 # Should not happen for i != j
        return log_table[d] % 4

    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(get_color(i, j))
            
    # Verify the coloring
    def has_c6_in_color_class(color_idx, n, colors):
        # Reconstruct the set of differences for this color
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * n - i * (i + 1) // 2 + (j - i - 1)
                # This indexing is tricky, let's just recompute or store matrix
                pass
        
        # Better: Build adjacency matrix or list of edges for the color
        # Since it's translation invariant, we can just use the difference set
        diff_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * n - i * (i + 1) // 2 + (j - i - 1)
                if colors[idx] == color_idx:
                    d = (j - i) % n
                    diff_set.add(d)
                    diff_set.add(-d % n)
                    
        # Check for C6 in translation invariant graph
        # d1..d6 in diff_set, sum=0, partial sums distinct
        diffs = list(diff_set)
        L = len(diffs)
        if L < 6:
            return False
            
        # Iterate over all combinations of 6 differences
        # 4^6 is small if L is small. Here L is likely 8 (since 4 residues per color class in F_17*? No, 4 elements in each color class, but +/- makes 8)
        # Wait, the set of differences for one color in the F_17 construction has 4 elements in F_17*.
        # So diff_set will have 4 elements and their negatives. Size 8.
        # 8^6 = 262144. Very fast.
        
        for d1 in diffs:
            s1 = d1
            if s1 == 0: continue
            for d2 in diffs:
                s2 = (s1 + d2) % n
                if s2 == 0 or s2 == s1: continue
                for d3 in diffs:
                    s3 = (s2 + d3) % n
                    if s3 == 0 or s3 in (s1, s2): continue
                    for d4 in diffs:
                        s4 = (s3 + d4) % n
                        if s4 == 0 or s4 in (s1, s2, s3): continue
                        for d5 in diffs:
                            s5 = (s4 + d5) % n
                            if s5 == 0 or s5 in (s1, s2, s3, s4): continue
                            for d6 in diffs:
                                s6 = (s5 + d6) % n
                                if s6 == 0:
                                    # Check distinctness of s1..s5
                                    # We already ensured s1..s5 are distinct and non-zero
                                    return True
        return False

    valid = True
    for c in range(4):
        if has_c6_in_color_class(c, n, colors):
            valid = False
            break
            
    if valid:
        return {'n': n, 'colors': colors}
        
    # If n=17 failed (should not), fallback to n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
