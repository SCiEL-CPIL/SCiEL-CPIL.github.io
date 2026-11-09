<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Try 49-vertex construction first
    gf49 = _build_gf49()
    if gf49 is not None:
        n = 49
        # Vertices 0..48 correspond to field elements 0..48
        # Color of edge (i, j) is determined by d = i - j (or j - i)
        # We need to map d to a color.
        # Hypothesis: Color(d) = log(d) % 4 for d != 0.
        # Check if this is valid.
        
        matrix = [[0] * n for _ in range(n)]
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                # d = i - j
                # In GF(49), subtraction is add(i, -j)
                # -j is add(0, j) inverted? No, additive inverse.
                # Since char is 7, -j = (7 - (j%7)) % 7 + 7*((7 - (j//7)) % 7)
                # Actually, add_table[i][j] gives i+j.
                # We need i - j.
                # Let's precompute neg table or just use add with negated j.
                # Neg of val a+7b is (7-a)%7 + 7*((7-b)%7)
                
                j_neg = ((7 - (j % 7)) % 7) + 7 * ((7 - (j // 7)) % 7)
                d = gf49['add'][i][j_neg]
                
                if d == 0:
                    # Should not happen for i != j
                    valid = False
                    break
                else:
                    c = gf49['log'][d] % 4
                    matrix[i][j] = c
                    matrix[j][i] = c
            if not valid:
                break
        
        if valid and _is_valid_fast(matrix, n):
            best_n = n
            best_matrix = matrix
        else:
            # If mod 4 doesn't work, try other colorings?
            # For now, fall back to 32
            best_n = 32
            best_matrix = _build_base_32()
    else:
        best_n = 32
        best_matrix = _build_base_32()

    # If we have 49, can we extend to 50?
    # Chung's 50-vertex construction is specific.
    # If we have time, we could try to extend 49 to 50.
    # But extending is hard.
    # Let's just return the best we have.
    
    # Format output
    out_colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            out_colors.append(best_matrix[i][j])

    return {'n': best_n, 'colors': out_colors}
=======
def _try_extend(matrix, n, deadline, max_time=5.0):
    """
    Try to extend an n-vertex coloring to n+1 vertices.
    Returns (new_matrix, new_n) or (None, n) if unsuccessful.
    """
    start = time.time()
    new_n = n + 1
    # For each candidate vertex, try all 4-color assignments to edges
    # This is a constraint satisfaction problem - use backtracking
    
    # Precompute: for each pair (i,j), what colors are forbidden?
    # A color c is forbidden for edge (new, v) if adding it creates a monochromatic triangle.
    # Triangle (new, v, w) with color c on (new,v) requires:
    #   - (new,w) also color c, and (v,w) also color c
    # So for a fixed assignment of colors from new to existing vertices,
    # we need to check consistency.
    
    # Simpler approach: random local search
    best_matrix = None
    best_score = -1
    
    for attempt in range(200):
        if time.time() - start > max_time:
            break
        # Random assignment of colors from new vertex to existing vertices
        new_colors = [random.randint(0, 3) for _ in range(n)]
        
        # Check validity
        valid = True
        for v in range(n):
            cv = new_colors[v]
            for w in range(v + 1, n):
                if matrix[v][w] == cv:
                    # Need new_colors[w] != cv to avoid triangle (new, v, w)
                    if new_colors[w] == cv:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            # Build new matrix
            new_matrix = [row[:] for row in matrix]
            for i in range(n):
                new_matrix.append([0] * new_n)
            for i in range(n):
                new_matrix[i].append(new_colors[i])
            new_matrix[n][n] = 0
            for i in range(n):
                new_matrix[n][i] = new_colors[i]
            best_matrix = new_matrix
            best_score = new_n
            break  # Found a valid extension!
    
    if best_matrix is not None:
        return best_matrix, new_n
    return None, n

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Try 49-vertex construction first
    gf49 = _build_gf49()
    if gf49 is not None:
        n = 49
        # Vertices 0..48 correspond to field elements 0..48
        # Color of edge (i, j) is determined by d = i - j
        # Use log(d) % 4 as color assignment
        
        matrix = [[0] * n for _ in range(n)]
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                d = gf49['add'][i][gf49['neg'][j]]
                
                if d == 0:
                    valid = False
                    break
                else:
                    c = gf49['log'][d] % 4
                    matrix[i][j] = c
                    matrix[j][i] = c
            if not valid:
                break
        
        if valid and _is_valid_fast(matrix, n):
            best_n = n
            best_matrix = matrix
            # Try to extend to 50
            if time.time() < deadline - 10:
                extended_matrix, extended_n = _try_extend(matrix, n, deadline, max_time=8.0)
                if extended_matrix is not None:
                    best_n = extended_n
                    best_matrix = extended_matrix
        else:
            best_n = 32
            best_matrix = _build_base_32()
    else:
        best_n = 32
        best_matrix = _build_base_32()

    # Format output
    out_colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            out_colors.append(best_matrix[i][j])

    return {'n': best_n, 'colors': out_colors}
>>>>>>> REPLACE

