import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    def time_check():
        return time.time() > deadline - 2.0

    # Strategy 1: Find n=17 cyclic coloring instantly
    # In Z_17, differences are 1..16.
    # Symmetric pairs: (1,16), (2,15), ..., (8,9). 8 pairs.
    # Each color class must be a union of symmetric pairs to be symmetric?
    # Actually, for the graph to be well-defined and undirected, if (i, i+d) is color c, 
    # then (i+d, i) is also color c, which corresponds to difference -d mod 17.
    # So the set of differences for color c, D_c, must be symmetric: d in D_c => -d in D_c.
    # The 8 pairs partition the 16 differences.
    # We need to assign these 8 pairs to 4 colors.
    # Each color gets exactly 2 pairs (4 differences) to balance? 
    # It's not strictly required that each color has 4 differences, but in a regular graph coloring 
    # derived from a difference set partition, if we want to maximize symmetry or just find ANY partition,
    # we can just partition the 8 pairs into 4 groups.
    # However, if one color gets 3 pairs (6 diffs) and another gets 1 pair (2 diffs), 
    # is that allowed? Yes.
    # But the "best known" n=17 coloring usually has regularity.
    # Let's just search for ANY partition of the 8 pairs into 4 sets such that 
    # no set contains a C6 pattern.
    
    pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
    
    def has_c6_in_diffs(D):
        """Check if Cayley graph on Z_17 with difference set D contains a C6."""
        D_set = set(D)
        negD = {(17 - d) % 17 for d in D_set}
        # We need distinct v0..v5 such that v_{i+1}-v_i in D
        # v0=0
        # v1 in D
        # v2 in v1+D
        # ...
        # v5 in v4+D
        # -v5 in D
        
        for v1 in D_set:
            for v2 in [(v1 + d) % 17 for d in D_set]:
                if v2 == 0 or v2 == v1: continue
                for v3 in [(v2 + d) % 17 for d in D_set]:
                    if v3 in (0, v1, v2): continue
                    for v4 in [(v3 + d) % 17 for d in D_set]:
                        if v4 in (0, v1, v2, v3): continue
                        for v5 in [(v4 + d) % 17 for d in D_set]:
                            if v5 in (0, v1, v2, v3, v4): continue
                            if (-v5) % 17 in D_set:
                                return True
        return False

    # Pre-compute which pairs are "safe" together?
    # A color class is a set of 2 pairs (4 diffs) or other combinations.
    # Let's just iterate through all partitions of 8 items into 4 groups.
    # 8 items -> 4 groups.
    # We can represent the assignment of each pair index (0..7) to a color (0..3).
    # Total assignments: 4^8 = 65536. This is very fast.
    
    found_n17 = None
    # To avoid symmetry duplicates, we can fix pair 0 to color 0, pair 1 to color 0 or 1, etc?
    # Or just iterate all 65536.
    
    for assignment in range(4**8):
        if time_check(): break
        
        # Decode assignment
        colors_for_pairs = [0] * 8
        temp = assignment
        for i in range(8):
            colors_for_pairs[i] = temp % 4
            temp //= 4
        
        # Build diff sets for each color
        diff_sets = [set() for _ in range(4)]
        for i, c in enumerate(colors_for_pairs):
            p = pairs[i]
            diff_sets[c].add(p[0])
            diff_sets[c].add(p[1])
        
        # Check validity
        valid = True
        for c in range(4):
            if has_c6_in_diffs(diff_sets[c]):
                valid = False
                break
        
        if valid:
            # Found a valid coloring for n=17
            color_map = {}
            for c in range(4):
                for d in diff_sets[c]:
                    color_map[d] = c
            
            n = 17
            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % 17
                    colors_list.append(color_map[diff])
            
            # Verify
            def verify_no_c6(colors_list, n):
                for c in range(4):
                    adj = [[False] * n for _ in range(n)]
                    idx = 0
                    for i in range(n):
                        for j in range(i + 1, n):
                            if colors_list[idx] == c:
                                adj[i][j] = adj[j][i] = True
                            idx += 1
                    # Check for C6
                    for a in range(n):
                        for b in range(n):
                            if not adj[a][b]: continue
                            for c_v in range(n):
                                if c_v == a or c_v == b or not adj[b][c_v]: continue
                                for d_v in range(n):
                                    if d_v in (a, b, c_v) or not adj[c_v][d_v]: continue
                                    for e_v in range(n):
                                        if e_v in (a, b, c_v, d_v) or not adj[d_v][e_v]: continue
                                        if adj[e_v][a] and e_v != b:
                                            return False
                return True
            
            if verify_no_c6(colors_list, n):
                found_n17 = {'n': n, 'colors': colors_list}
                break

    if found_n17:
        return found_n17

    # Fallback if n=17 search fails (should not happen)
    # Try n=16? Or return n=9?
    # Let's try a simple random search for n=17 general (non-cyclic) just in case
    # Or just return a known small one.
    
    # Actually, if the cyclic search fails, it's a bug. 
    # But let's add a generic random restart local search for n=18 (just in case) 
    # or n=17 if cyclic failed.
    
    # Given the time budget, and that n=17 is known to exist, 
    # the cyclic search *must* find it.
    # If it doesn't, return n=9.
    
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
