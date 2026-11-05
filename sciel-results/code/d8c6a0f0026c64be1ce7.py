import time
import itertools

def _has_c6(color_class_adj, n):
    """Check if a color class (given as list of adjacency bitmasks) contains a 6-cycle."""
    # We check for a cycle of length 6: v0-v1-v2-v3-v4-v5-v0
    # v0 < v1 < v2 < v3 < v4 < v5 to avoid duplicates
    # v1 is a neighbor of v0
    # v2 is a neighbor of v1, v2 != v0, v2 > v0
    # v3 is a neighbor of v2, v3 not in {v0, v1, v2}
    # v4 is a neighbor of v3, v4 not in {v0, v1, v2, v3}
    # v5 is a neighbor of v4 and v0, v5 not in {v0, v1, v2, v3, v4}
    
    adj = color_class_adj
    
    for v0 in range(n):
        # v1 must be > v0
        v1_candidates = adj[v0] & ~((1 << (v0 + 1)) - 1)
        while v1_candidates:
            v1 = (v1_candidates & -v1_candidates).bit_length() - 1
            v1_candidates &= v1_candidates - 1
            
            # v2 must be > v0, v2 != v1, neighbor of v1
            # v2 can be < v1 or > v1, but v2 > v0
            v2_candidates = adj[v1] & ~((1 << (v0 + 1)) - 1)
            v2_candidates &= ~(1 << v1)
            # v2 must not be v0
            v2_candidates &= ~(1 << v0)
            
            while v2_candidates:
                v2 = (v2_candidates & -v2_candidates).bit_length() - 1
                v2_candidates &= v2_candidates - 1
                
                # v3 must not be in {v0, v1, v2}, neighbor of v2
                forbidden1 = (1 << v0) | (1 << v1) | (1 << v2)
                v3_candidates = adj[v2] & ~forbidden1
                # v3 must not equal v0, v1, v2
                # Also v3 > v0 to maintain order? No, we need v0 < v1 < v2 < v3 < v4 < v5
                # Actually, the cycle is v0-v1-v2-v3-v4-v5-v0
                # We require v0 < v1 < v2 < v3 < v4 < v5
                # So v3 > v2
                v3_candidates &= ~((1 << (v2 + 1)) - 1)
                
                while v3_candidates:
                    v3 = (v3_candidates & -v3_candidates).bit_length() - 1
                    v3_candidates &= v3_candidates - 1
                    
                    # v4 must not be in {v0, v1, v2, v3}, neighbor of v3
                    forbidden2 = forbidden1 | (1 << v3)
                    v4_candidates = adj[v3] & ~forbidden2
                    # v4 > v3
                    v4_candidates &= ~((1 << (v3 + 1)) - 1)
                    
                    while v4_candidates:
                        v4 = (v4_candidates & -v4_candidates).bit_length() - 1
                        v4_candidates &= v4_candidates - 1
                        
                        # v5 must be neighbor of v4 AND v0
                        # v5 must not be in {v0, v1, v2, v3, v4}
                        # v5 > v4
                        forbidden3 = forbidden2 | (1 << v4)
                        v5_candidates = adj[v4] & adj[v0] & ~forbidden3
                        v5_candidates &= ~((1 << (v4 + 1)) - 1)
                        
                        if v5_candidates:
                            return True
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # We want to maximize n. Best known is 17.
    # Let's try to find a coloring for n=18 or n=19.
    # But first, let's ensure we can quickly return n=17 or better if we find it.
    
    # The seed program gives n=9.
    # Let's try to find a valid coloring for larger n using local search.
    
    best_n = 0
    best_colors = None
    
    # We will try to build a coloring for n=18.
    # If we can't find one in time, we fallback to a known n=17 or n=9.
    
    # Actually, let's just use a known valid coloring for n=17 if possible, or try to extend.
    # A 4-coloring of K17 with no C6 in any color class.
    
    # Let's try random local search for n=18.
    # Variables: color[i][j] for 0 <= i < j < n
    # Objective: Minimize number of C6s in each color class.
    # Constraint: 0 C6s.
    
    n_target = 18
    if time.time() > deadline:
        return {'n': 9, 'colors': [min(j - i, 18 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
        
    # Initialize randomly
    colors = {}
    for i in range(n_target):
        for j in range(i + 1, n_target):
            colors[(i, j)] = i % 4 # Some heuristic init
            
    def count_c6s(colors, n):
        adj = [[0] * n for _ in range(n)]
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[(i, j)] == c:
                        adj[i][j] |= (1 << j)
                        adj[j][i] |= (1 << i)
        total = 0
        for c in range(4):
            if _has_c6(adj, n):
                total += 1
        return total
    
    def local_search(colors, n, timeout):
        t_start = time.time()
        cur_colors = dict(colors)
        cur_c6s = count_c6s(cur_colors, n)
        
        # Precompute adjacencies for speed? No, count_c6s is fast enough for 18 vertices.
        # But we can optimize by only checking if the changed edge affects C6s.
        
        # Simpler: just recompute.
        
        improved = True
        while improved and time.time() < timeout:
            improved = False
            # Randomly pick an edge
            edges = list(cur_colors.keys())
            import random
            random.shuffle(edges)
            for (i, j) in edges:
                old_c = cur_colors[(i, j)]
                for new_c in range(4):
                    if new_c == old_c: continue
                    cur_colors[(i, j)] = new_c
                    new_c6s = count_c6s(cur_colors, n)
                    if new_c6s < cur_c6s:
                        cur_c6s = new_c6s
                        improved = True
                        if cur_c6s == 0:
                            return cur_colors
                    else:
                        cur_colors[(i, j)] = old_c
        return cur_colors
        
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        init_colors = {}
        for i in range(n_target):
            for j in range(i + 1, n_target):
                init_colors[(i, j)] = (i + j) % 4
        res_colors = local_search(init_colors, n_target, deadline)
        c6s = count_c6s(res_colors, n_target)
        if c6s == 0:
            best_n = n_target
            best_colors = res_colors
            break
        elif c6s < count_c6s(init_colors, n_target):
            # Keep best so far?
            pass

    if best_n == 0:
        # Fallback to n=17 if we have one, or n=9
        # Let's try to hardcode a known n=17 coloring if I can recall one?
        # No, I don't have one memorized.
        # Let's just return the seed n=9.
        n = 9
        return {'n': n, 'colors': [min(j - i, 18 - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
        
    # Construct list in lexicographic order
    color_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            color_list.append(best_colors[(i, j)])
            
    return {'n': best_n, 'colors': color_list}
