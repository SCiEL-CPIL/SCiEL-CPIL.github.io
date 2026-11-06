import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115
    
    # Try n=17 using circulant construction
    result = find_circulant_coloring(17)
    if result is not None:
        return {'n': 17, 'colors': result}
        
    # Fallback to n=16
    result = find_circulant_coloring(16)
    if result is not None:
        return {'n': 16, 'colors': result}
        
    # Fallback to n=15
    result = find_circulant_coloring(15)
    if result is not None:
        return {'n': 15, 'colors': result}
        
    # Fallback to a trivial small graph
    return {'n': 5, 'colors': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2]}

def find_circulant_coloring(n):
    """
    Finds a valid 4-coloring of K_n using a circulant graph construction.
    Returns list of colors for edges (i,j) i<j in lexicographic order, or None.
    """
    if n < 3:
        return []
        
    # We only need to color differences 1 to n//2
    # C(d) = C(n-d)
    half = n // 2
    # If n is even, difference n/2 is self-inverse.
    # Differences to assign: 1, ..., half
    # Total assignments: 4^half
    
    # Precompute bad subsets for this n
    # A subset S of {1, ..., n-1} is "bad" if it contains a set of 6 differences
    # that form a C6 in the circulant graph.
    
    # To save time, we only precompute if n is small enough (n<=17)
    # For n=17, half=8. 4^8 = 65536.
    # Bad subset precomputation cost: O(n^6 * 2^(n/2)?) No, O(n^6) to find minimal bad sets.
    
    # Identify differences 1..n-1
    # We represent subsets as bitmasks of size n-1.
    # Bit i-1 corresponds to difference i.
    
    bad_min = [False] * (1 << (n-1))
    
    # Iterate all 6-tuples of differences
    # This is O(n^6). For n=17, 17^6 = 24M. This might be slow in Python.
    # Optimization: Only consider tuples that sum to 0 mod n.
    # We can use recursion or pruning.
    # Or just iterate and check sum.
    
    # Given the time limit, let's optimize the tuple iteration.
    # For n=17, we can iterate d1..d5 and compute d6 = -sum(d1..d5).
    # Check if d6 is in 1..n-1.
    
    for d1 in range(1, n):
        for d2 in range(1, n):
            for d3 in range(1, n):
                for d4 in range(1, n):
                    for d5 in range(1, n):
                        s5 = (d1 + d2 + d3 + d4 + d5) % n
                        d6 = (-s5) % n
                        if d6 == 0:
                            continue
                        # Check partial sums distinct and non-zero
                        # Vertices: 0, d1, d1+d2, d1+d2+d3, d1+d2+d3+d4, d1+..+d5
                        # Must be distinct and non-zero (except start/end which is 0)
                        # Actually v0=0, v1=d1, v2=d1+d2, v3=.., v4=.., v5=..
                        # v6=0.
                        # v1..v5 must be non-zero and distinct.
                        v1 = d1 % n
                        v2 = (d1 + d2) % n
                        v3 = (d1 + d2 + d3) % n
                        v4 = (d1 + d2 + d3 + d4) % n
                        v5 = (d1 + d2 + d3 + d4 + d5) % n
                        
                        if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0 or v5 == 0:
                            continue
                        if len({v1, v2, v3, v4, v5}) != 5:
                            continue
                            
                        # Valid C6 found with steps d1..d6
                        # Mark the set of differences used
                        mask = 0
                        for d in [d1, d2, d3, d4, d5, d6]:
                            mask |= (1 << (d-1))
                        bad_min[mask] = True
                        
    # SOS DP to propagate badness to supersets
    bad = bad_min[:]
    for i in range(n-1):
        for mask in range(1 << (n-1)):
            if mask & (1 << i):
                if bad[mask ^ (1 << i)]:
                    bad[mask] = True
                    
    # Now search for a valid coloring of 1..half
    # 4 colors
    # We can use recursion or itertools.product
    # For n=17, half=8. 4^8 = 65536.
    
    for colors in itertools.product(range(4), repeat=half):
        # Check if this coloring is valid
        # Construct sets S_c for c in 0..3
        # S_c contains differences d in 1..n-1 such that color(d) == c
        # color(d) = colors[d-1] if d <= half
        # color(d) = colors[n-d-1] if d > half
        
        S = [set() for _ in range(4)]
        for d in range(1, n):
            if d <= half:
                c = colors[d-1]
            else:
                c = colors[n-d-1]
            S[c].add(d)
            
        valid = True
        for c in range(4):
            mask = 0
            for d in S[c]:
                mask |= (1 << (d-1))
            if bad[mask]:
                valid = False
                break
                
        if valid:
            # Construct the full edge coloring
            edge_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    diff = (j - i) % n
                    if diff <= half:
                        c = colors[diff-1]
                    else:
                        c = colors[n-diff-1]
                    edge_colors.append(c)
            return edge_colors
            
    return None
# EVOLVE-BLOCK-END
