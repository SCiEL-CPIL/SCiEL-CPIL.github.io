import time
import random
import numpy as np

def _is_valid_fast(matrix, n):
    """
    Check if the coloring of K_n defined by matrix (n x n, upper triangle valid) 
    has no monochromatic triangles.
    matrix[i][j] is the color for edge (i,j).
    """
    for c in range(4):
        # Build list of edges with color c
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] == c:
                    edges.append((i, j))
        
        # Build adjacency sets for color c
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        
        # Check for triangles: for each edge (u,v), check if any common neighbor
        for u, v in edges:
            common = adj[u] & adj[v]
            if common:
                return False
    
    return True

def _build_gf49():
    """
    Construct GF(49) using irreducible polynomial x^2 + x + 3 over GF(7).
    Returns dict with add, mul, neg, log tables.
    """
    # x^2 + x + 3 = 0 => x^2 = -x - 3 = 6x + 4 in GF(7)
    def add(v1, v2):
        a1, b1 = v1 % 7, v1 // 7
        a2, b2 = v2 % 7, v2 // 7
        return (a1 + a2) % 7 + 7 * ((b1 + b2) % 7)

    def neg(v):
        a, b = v % 7, v // 7
        return (7 - a) % 7 + 7 * ((7 - b) % 7)

    def mul(v1, v2):
        a1, b1 = v1 % 7, v1 // 7
        a2, b2 = v2 % 7, v2 // 7
        # (a1 + b1*w)*(a2 + b2*w) = a1*a2 + (a1*b2+a2*b1)*w + b1*b2*w^2
        # w^2 = 6w + 4
        a = (a1*a2 + 4*b1*b2) % 7
        b = (a1*b2 + a2*b1 + 6*b1*b2) % 7
        return a + 7*b

    add_table = [[add(i, j) for j in range(49)] for i in range(49)]
    neg_table = [neg(i) for i in range(49)]
    mul_table = [[mul(i, j) for j in range(49)] for i in range(49)]
    
    # Find primitive element (generator of multiplicative group of order 48)
    primitive = -1
    for g in range(2, 49):
        # Check if g has order 48
        # 48 = 2^4 * 3, so check g^(48/2) != 1 and g^(48/3) != 1
        # i.e., g^24 != 1 and g^16 != 1
        x = 1
        for _ in range(24):
            x = mul_table[g][x]
        if x == 1:
            continue
        x = 1
        for _ in range(16):
            x = mul_table[g][x]
        if x == 1:
            continue
        primitive = g
        break
        
    if primitive == -1:
        return None

    # Build log table
    log_table = [0] * 49
    log_table[1] = 0
    x = 1
    for k in range(1, 48):
        x = mul_table[primitive][x]
        log_table[x] = k
        
    return {
        'add': add_table,
        'mul': mul_table,
        'neg': neg_table,
        'log': log_table
    }

def _build_49_coloring():
    """
    Build a 49-vertex 4-coloring using GF(49).
    Color of edge (i,j) = log(i-j) mod 4 where log is discrete log in GF(49).
    """
    gf = _build_gf49()
    if gf is None:
        return None
    
    n = 49
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = gf['add'][i][gf['neg'][j]]
            if d == 0:
                return None  # Should not happen since i != j
            c = gf['log'][d] % 4
            matrix[i][j] = c
            matrix[j][i] = c
    
    # Verify
    if not _is_valid_fast(matrix, n):
        return None
    
    return matrix

def _build_25_coloring():
    """
    Build a 25-vertex 4-coloring using GF(25).
    GF(25) = GF(5)[x]/(x^2 + 2) since x^2 + 2 is irreducible over GF(5).
    x^2 = -2 = 3 in GF(5).
    Multiplicative group has order 24, divisible by 4.
    """
    def add(v1, v2):
        a1, b1 = v1 % 5, v1 // 5
        a2, b2 = v2 % 5, v2 // 5
        return (a1 + a2) % 5 + 5 * ((b1 + b2) % 5)

    def neg(v):
        a, b = v % 5, v // 5
        return (5 - a) % 5 + 5 * ((5 - b) % 5)

    def mul(v1, v2):
        a1, b1 = v1 % 5, v1 // 5
        a2, b2 = v2 % 5, v2 // 5
        # w^2 = 3
        a = (a1*a2 + 3*b1*b2) % 5
        b = (a1*b2 + a2*b1) % 5
        return a + 5*b

    add_table = [[add(i, j) for j in range(25)] for i in range(25)]
    neg_table = [neg(i) for i in range(25)]
    mul_table = [[mul(i, j) for j in range(25)] for i in range(25)]
    
    # Find primitive element of order 24
    primitive = -1
    for g in range(2, 25):
        # 24 = 2^3 * 3, check g^12 != 1 and g^8 != 1
        x = 1
        for _ in range(12):
            x = mul_table[g][x]
        if x == 1:
            continue
        x = 1
        for _ in range(8):
            x = mul_table[g][x]
        if x == 1:
            continue
        primitive = g
        break
    
    if primitive == -1:
        return None

    log_table = [0] * 25
    log_table[1] = 0
    x = 1
    for k in range(1, 24):
        x = mul_table[primitive][x]
        log_table[x] = k
    
    n = 25
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = add_table[i][neg_table[j]]
            if d == 0:
                return None
            c = log_table[d] % 4
            matrix[i][j] = c
            matrix[j][i] = c
    
    if not _is_valid_fast(matrix, n):
        return None
    
    return matrix

def _build_9_coloring():
    """
    Build a 9-vertex 4-coloring using GF(9).
    GF(9) = GF(3)[x]/(x^2 + 1) since x^2 + 1 is irreducible over GF(3).
    x^2 = -1 = 2 in GF(3).
    Multiplicative group has order 8, divisible by 4.
    """
    def add(v1, v2):
        a1, b1 = v1 % 3, v1 // 3
        a2, b2 = v2 % 3, v2 // 3
        return (a1 + a2) % 3 + 3 * ((b1 + b2) % 3)

    def neg(v):
        a, b = v % 3, v // 3
        return (3 - a) % 3 + 3 * ((3 - b) % 3)

    def mul(v1, v2):
        a1, b1 = v1 % 3, v1 // 3
        a2, b2 = v2 % 3, v2 // 3
        # w^2 = 2
        a = (a1*a2 + 2*b1*b2) % 3
        b = (a1*b2 + a2*b1) % 3
        return a + 3*b

    add_table = [[add(i, j) for j in range(9)] for i in range(9)]
    neg_table = [neg(i) for i in range(9)]
    mul_table = [[mul(i, j) for j in range(9)] for i in range(9)]
    
    # Find primitive element of order 8
    primitive = -1
    for g in range(2, 9):
        # 8 = 2^3, check g^4 != 1
        x = 1
        for _ in range(4):
            x = mul_table[g][x]
        if x == 1:
            continue
        primitive = g
        break
    
    if primitive == -1:
        return None

    log_table = [0] * 9
    log_table[1] = 0
    x = 1
    for k in range(1, 8):
        x = mul_table[primitive][x]
        log_table[x] = k
    
    n = 9
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = add_table[i][neg_table[j]]
            if d == 0:
                return None
            c = log_table[d] % 4
            matrix[i][j] = c
            matrix[j][i] = c
    
    if not _is_valid_fast(matrix, n):
        return None
    
    return matrix

def _build_5_coloring():
    """
    Build a 5-vertex 4-coloring.
    """
    n = 5
    matrix = [[0] * n for _ in range(n)]
    # Simple assignment that avoids monochromatic triangles
    # K5 with 4 colors, no monochromatic triangle
    # Use: 0-1:0, 0-2:1, 0-3:2, 0-4:3, 1-2:2, 1-3:3, 1-4:0, 2-3:0, 2-4:1, 3-4:1
    colors = {
        (0,1): 0, (0,2): 1, (0,3): 2, (0,4): 3,
        (1,2): 2, (1,3): 3, (1,4): 0,
        (2,3): 0, (2,4): 1,
        (3,4): 1
    }
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[(i, j)]
            matrix[i][j] = c
            matrix[j][i] = c
    
    if not _is_valid_fast(matrix, n):
        return None
    return matrix

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try constructions from largest to smallest
    best_n = 0
    best_matrix = None
    
    # Try 49
    if time.time() < deadline - 5:
        m49 = _build_49_coloring()
        if m49 is not None:
            best_n = 49
            best_matrix = m49
    
    # Try 25
    if best_n < 25 and time.time() < deadline - 5:
        m25 = _build_25_coloring()
        if m25 is not None:
            best_n = 25
            best_matrix = m25
    
    # Try 9
    if best_n < 9 and time.time() < deadline - 5:
        m9 = _build_9_coloring()
        if m9 is not None:
            best_n = 9
            best_matrix = m9
    
    # Try 5
    if best_n < 5 and time.time() < deadline - 5:
        m5 = _build_5_coloring()
        if m5 is not None:
            best_n = 5
            best_matrix = m5
    
    # Fallback: n=4 (always valid)
    if best_n < 4:
        best_n = 4
        best_matrix = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                best_matrix[i][j] = (i + j) % 4
                best_matrix[j][i] = best_matrix[i][j]
    
    # Format output
    out_colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            out_colors.append(best_matrix[i][j])

    return {'n': best_n, 'colors': out_colors}
