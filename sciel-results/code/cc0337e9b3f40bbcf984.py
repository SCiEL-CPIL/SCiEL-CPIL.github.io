import time
import random
import sys

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    if n < 2:
        return True
    # Build adjacency sets for each color
    color_adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            color_adj[c][i].add(j)
            color_adj[c][j].add(i)
            idx += 1
    # Check for C4: for each color, no two vertices should have 2+ common neighbors
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = color_adj[c][u] & color_adj[c][v]
                if len(common) >= 2:
                    return False
    return True

def _lz_woldar_coloring(n=26):
    """
    Construct the Lazebnik-Woldar 5-coloring of K_26.
    
    The construction uses the finite field F_25.
    The vertices are F_25 U {infinity}.
    F_25^* is partitioned into 4 cosets of a subgroup H of order 6.
    The 4 cosets correspond to colors 0, 1, 2, 3.
    Color 4 is used for edges involving infinity.
    
    Specifically:
    - Let G be the multiplicative group F_25^* of order 24.
    - Let H be a subgroup of order 6.
    - The cosets of H in G are H, gH, g^2H, g^3H for a generator g.
    - For u, v in F_25, u != v:
      - Let d = v - u.
      - If d is in coset k, assign color k.
    - For edges (infinity, x), assign color 4.
    
    Wait, the Lazebnik-Woldar construction for R_5(C4) > 26 is:
    Vertices: F_25 U {inf}.
    Edges in F_25 are colored by the 4 cosets of a subgroup of order 6 in F_25^*.
    Edges to inf are colored with the 5th color.
    
    Let's verify the C4-free property.
    In F_25, the graph for color c is the Cayley graph Cay(F_25, S_c) where S_c is a coset.
    Since S_c is a coset of H, and H is a subgroup, the graph is a union of cycles?
    Actually, the condition for no C4 in Cay(F_q, S) is that the difference set S has no repeated differences?
    Or more simply, for any two vertices u, v, the number of common neighbors is |S ∩ (S + (v-u))|?
    No, common neighbors of u and v in Cay(G, S) are w such that w-u in S and w-v in S.
    So w-u = s1, w-v = s2 => v-u = s2 - s1.
    So the number of common neighbors is |{(s1, s2) in S x S : s2 - s1 = v-u}|.
    For a coset S = aH, s2 - s1 = a(h2 - h1).
    So the number of common neighbors depends on v-u.
    If v-u is not in aH - aH = H, then 0 common neighbors.
    If v-u is in H, then the number of pairs (h1, h2) in H x H such that h2 - h1 = v-u.
    Since H is a subgroup of order 6 in F_25^*, and F_25 is a field, H is cyclic of order 6.
    The difference set H-H contains 0 and the other elements.
    Actually, for a subgroup H of a field, the difference set is H ∪ {0}? No.
    If h1, h2 in H, h2 - h1 can be any element.
    However, the Lazebnik-Woldar construction is known to be C4-free.
    I will trust the construction and implement it.
    
    Steps:
    1. Define F_25 arithmetic.
    2. Find a subgroup H of order 6.
    3. Partition F_25^* into 4 cosets.
    4. Map cosets to colors 0,1,2,3.
    5. Color edges in F_25 based on difference.
    6. Color edges to vertex 25 (infinity) with color 4.
    7. Return the list of colors for K_26.
    """
    # F_25 representation: a + b*i where i^2 = 2.
    # Elements are pairs (a, b) in Z_5 x Z_5.
    # Multiplication: (a,b)(c,d) = (ac - 2bd, ad + bc).
    
    def add(p, q):
        return ((p[0] + q[0]) % 5, (p[1] + q[1]) % 5)
    
    def sub(p, q):
        return ((p[0] - q[0]) % 5, (p[1] - q[1]) % 5)
    
    def mul(p, q):
        a, b = p
        c, d = q
        return ((a*c - 2*b*d) % 5, (a*d + b*c) % 5)
    
    # Find generator of F_25^*
    # Try elements until we find one with order 24.
    def order(p):
        cur = (1, 0)
        for k in range(1, 25):
            cur = mul(cur, p)
            if cur == (1, 0):
                return k
        return -1
    
    generator = None
    for a in range(5):
        for b in range(5):
            if (a, b) == (0, 0):
                continue
            if order((a, b)) == 24:
                generator = (a, b)
                break
        if generator:
            break
    
    if not generator:
        # Fallback, shouldn't happen
        return [0] * (26*25//2)
    
    # Find subgroup of order 6
    # H = <g^4>
    g4 = mul(mul(generator, generator), mul(generator, generator))
    H = set()
    cur = (1, 0)
    for _ in range(6):
        H.add(cur)
        cur = mul(cur, g4)
    
    # Cosets
    cosets = [set() for _ in range(4)]
    for k in range(4):
        # g^k * H
        gk = (1, 0)
        for _ in range(k):
            gk = mul(gk, generator)
        for h in H:
            cosets[k].add(mul(gk, h))
    
    # Map element to color (0-3)
    elem_to_color = {}
    for c in range(4):
        for e in cosets[c]:
            elem_to_color[e] = c
    
    # Build coloring for K_26
    # Vertices 0..24 correspond to F_25 elements.
    # Vertex 25 is infinity.
    # We need a mapping from index 0..24 to F_25 element.
    # Let's just iterate through all 25 elements.
    elements = []
    for a in range(5):
        for b in range(5):
            elements.append((a, b))
    
    # Build color list
    colors = []
    for i in range(26):
        for j in range(i + 1, 26):
            if i == 25 or j == 25:
                # Edge to infinity
                colors.append(4)
            else:
                # Edge in F_25
                e_i = elements[i]
                e_j = elements[j]
                diff = sub(e_j, e_i)
                # diff is in F_25^* since i != j
                c = elem_to_color.get(diff, 0)
                colors.append(c)
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 seconds margin
    
    best_n = 1
    best_colors = []
    
    # Helper to flatten cm to list
    def flatten(cm, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(cm[i][j])
        return colors

    # Helper to build conflict matrix for current graph
    # conflict[c][u][v] = 1 if u and v have a common neighbor in color c
    def build_conflicts(cm, n):
        adj = [[set() for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i][j]
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        conflicts = [[[0] * n for _ in range(n)] for _ in range(5)]
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[c][u] & adj[c][v]:
                        conflicts[c][u][v] = 1
                        conflicts[c][v][u] = 1
        return conflicts, adj

    # Try to extend coloring from size n to n+1
    def try_extend(cm, n, conflicts, adj, max_attempts=20):
        new_n = n + 1
        # new_cm is a copy of cm extended
        new_cm = [row[:] + [-1] for row in cm]
        new_cm.append([-1] * new_n)
        
        # We need to assign colors to edges (new_n-1, i) for i in 0..n-1
        # Let new_v = new_n - 1
        new_v = new_n - 1
        
        # assigned[i] = color of edge (new_v, i)
        assigned = [-1] * n
        
        # Precompute potential valid colors for each i?
        # A color c is valid for i if for all j where assigned[j]==c, conflicts[c][i][j] == 0.
        # This depends on the assignment of other vertices.
        
        # Backtracking state
        # Order of vertices to assign: random shuffle
        order = list(range(n))
        
        for attempt in range(max_attempts):
            random.shuffle(order)
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    # Success
                    for i in range(n):
                        c = assigned[i]
                        new_cm[new_v][i] = c
                        new_cm[i][new_v] = c
                    return True
                
                u = order[idx]
                # Try colors 0-4
                # Heuristic: try colors that are less constrained?
                # For now, just try 0-4 with random tie breaking or fixed order.
                colors_to_try = [0, 1, 2, 3, 4]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    valid = True
                    # Check against all previously assigned vertices j in order[0..idx-1]
                    # such that assigned[j] == c
                    # Actually, we only need to check against j where assigned[j] == c.
                    # We can iterate through all j < idx in the order, or maintain a list of assigned vertices per color.
                    
                    # Optimization: Maintain assigned_vertices[c] = list of indices assigned color c so far.
                    # But since we are in backtracking, we need to add/remove.
                    # Simpler: just check all j in 0..n-1 where assigned[j] == c.
                    # Since n is small (<30), this is fast.
                    
                    for j in range(n):
                        if assigned[j] == c:
                            if conflicts[c][u][j] == 1:
                                valid = False
                                break
                    
                    if valid:
                        assigned[u] = c
                        if backtrack(idx + 1):
                            return True
                        assigned[u] = -1
                
                return False
            
            if backtrack(0):
                return new_cm, True
        
        return None, False

    # Main loop
    # Start with the Lazebnik-Woldar coloring for n=26
    lw_colors = _lz_woldar_coloring(26)
    # Verify it's valid
    if _is_valid_coloring(26, lw_colors):
        best_n = 26
        best_colors = lw_colors
        # Convert to matrix
        cm = [[-1] * 26 for _ in range(26)]
        idx = 0
        for i in range(26):
            for j in range(i + 1, 26):
                cm[i][j] = lw_colors[idx]
                cm[j][i] = lw_colors[idx]
                idx += 1
        n = 26
        conflicts, adj = build_conflicts(cm, n)
    else:
        # Fallback to n=1 if construction failed (should not happen)
        cm = [[0]]
        n = 1
        conflicts, adj = build_conflicts(cm, n)
        best_n = 1
        best_colors = []

    # Try to extend from n=26 to n=29
    stuck_count = 0
    while n < 29:
        if time.time() > deadline:
            break
            
        success = False
        # Try to extend
        # Use a moderate number of attempts for each extension step
        attempts_for_step = 100
        new_cm, success = try_extend(cm, n, conflicts, adj, attempts_for_step)
        
        if success:
            n += 1
            cm = new_cm
            # Update conflicts and adj
            conflicts, adj = build_conflicts(cm, n)
            stuck_count = 0
            # Update best
            if n > best_n:
                best_n = n
                best_colors = flatten(cm, n)
        else:
            stuck_count += 1
            if stuck_count > 20:
                # If stuck for a while, maybe restart from a smaller base?
                # Or just break and return what we have.
                # Given the time limit, breaking is safer.
                # However, we can try to restart the whole process from n=1 with a different seed?
                # That might be too slow.
                # Let's just break.
                break

    # If we didn't reach a high n, ensure we return a valid small one
    if best_n < 2:
        best_n = 2
        best_colors = [0]
    elif best_n < n:
        # This shouldn't happen as we update best_n on success
        pass

    return {'n': best_n, 'colors': best_colors}
