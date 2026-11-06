def construct():
    n_base = 29
    # Calculate quadratic residues modulo 29
    residues = set()
    for x in range(1, n_base):
        residues.add((x * x) % n_base)
    
    # Base colors for n=29
    base_colors = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % n_base
            if diff in residues:
                base_colors.append(0)
            else:
                base_colors.append(1)
                
    # Helper to get color from base list
    # Index mapping: (i, j) -> i*(n_base-1) + j - (i+1) ... simpler to use a 2D array or dict
    # Let's use a 2D list for speed
    adj = [[0]*n_base for _ in range(n_base)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
            
    # Try to extend to n=30
    n = 30
    # Initialize colors for new vertex (29)
    # We will search for a valid assignment of colors for edges (i, 29)
    
    # Randomize initial guess
    import random
    random.seed(42)
    
    # List of colors for edges to new vertex
    new_colors = [random.randint(0, 1) for _ in range(n_base)]
    
    def is_valid_with_new_vertex(new_colors):
        # Check if adding vertex 29 with these colors creates a mono K2,2,2
        # For each color C in {0, 1}
        for C in [0, 1]:
            # S_C = {u | adj[u][29] == C} i.e., new_colors[u] == C
            S_C = [u for u in range(n_base) if new_colors[u] == C]
            
            # For each u in S_C, u is the partner of 29 in one part of K2,2,2
            for u in S_C:
                # We need to find two disjoint pairs {a,b} and {c,d} from W
                # W = {w | new_colors[w] == C and adj[u][w] == C}
                W = [w for w in S_C if w != u and adj[u][w] == C]
                
                # We need to pick 4 distinct vertices a,b,c,d from W
                # such that all edges between {a,b} and {c,d} are C
                # And {a,b} disjoint from {c,d}
                
                # Iterate over pairs in W
                # To speed up, we can just check all combinations of 4?
                # |W| can be up to 28. 28 choose 4 is ~ 20k.
                # For each u, this is fast enough.
                # Total u's up to 28. Total checks 28 * 20k = 500k. Fast.
                
                w_list = W
                len_w = len(w_list)
                if len_w < 4:
                    continue
                    
                # Find if there exist a,b,c,d distinct in w_list
                # such that adj[a][c]==C, adj[a][d]==C, adj[b][c]==C, adj[b][d]==C
                # And a,b != c,d
                
                # Optimization: Group by neighbors?
                # Simpler: Just nested loops
                for i1 in range(len_w):
                    a = w_list[i1]
                    for i2 in range(i1+1, len_w):
                        b = w_list[i2]
                        # Candidate pair 1: {a,b}
                        # Now find c,d from remaining
                        # c, d must be in w_list, not a or b
                        # and connected to a and b with color C
                        
                        # Precompute intersection of N_C(a) and N_C(b) within W?
                        # Or just loop
                        for i3 in range(len_w):
                            c = w_list[i3]
                            if c == a or c == b: continue
                            if adj[a][c] != C or adj[b][c] != C: continue
                            for i4 in range(i3+1, len_w):
                                d = w_list[i4]
                                if d == a or d == b: continue
                                if d == c: continue # distinct
                                if adj[a][d] != C or adj[b][d] != C: continue
                                if adj[c][d] != C: continue
                                # Found a K2,2,2: {29, u}, {a, b}, {c, d}
                                return False
        return True

    # Local search
    # If initial random fails, flip bits
    max_iters = 5000
    found = False
    if is_valid_with_new_vertex(new_colors):
        found = True
    else:
        for _ in range(max_iters):
            # Flip a random bit
            idx_flip = random.randint(0, n_base - 1)
            new_colors[idx_flip] ^= 1
            if is_valid_with_new_vertex(new_colors):
                found = True
                break

    if found:
        # Construct final list
        final_colors = base_colors.copy()
        # Add colors for (i, 29) in lexicographic order
        # Pairs (0,29), (1,29), ..., (28,29)
        for i in range(n_base):
            final_colors.append(new_colors[i])
        return {'n': 30, 'colors': final_colors}
    else:
        # Fallback to n=29
        return {'n': 29, 'colors': base_colors}
