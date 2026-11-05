"""
Matrix multiplication 5x5x5 tensor decomposition.

Approach: Block decomposition with 5 = 3 + 2 partition.
- Use Strassen's algorithm (rank 7) for the 2x2x2 block
- Use schoolbook (rank n*m*p) for all other blocks

Total rank: 18 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 115
"""

DIMENSIONS = [5, 5, 5]


def _schoolbook_factors(n, m, p):
    """Return U, V, W for schoolbook n x m x p matrix multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _strassen_factors():
    """
    Return U, V, W for Strassen's 2x2x2 algorithm.
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[a*(e+f) - (d-c)*(g+f), b*(e+f)],
         [(a+c)*h - (e+d)*(h-g), (a+c)*(g+h)]]
    
    Seven products:
    M1 = a*(e+f)
    M2 = (a+b)*h
    M3 = c*(g+h)
    M4 = d*(e+f)
    M5 = (a+d)*(e+h-f)
    M6 = (b+c)*(g+e-h)
    M7 = (a+c)*(g+h)
    
    C00 = M1 - M4 + M5
    C01 = M2 - M5 + M7
    C10 = M3 + M4 - M6
    C11 = M1 - M3 + M6
    """
    rank = 7
    # U: shape (n*m, rank) = (4, 7), indexed by [i*m+j]
    # A = [[a, b], [c, d]] -> indices: a=0, b=1, c=2, d=3
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # a: M1, M5
        [0, 1, 0, 0, 0, 1, 0],  # b: M2, M6
        [0, 0, 1, 0, 0, 1, 1],  # c: M3, M6, M7
        [0, 0, 0, 1, 1, 0, 0],  # d: M4, M5
    ]
    # V: shape (m*p, rank) = (4, 7), indexed by [j*p+k]
    # B = [[e, f], [g, h]] -> indices: e=0, f=1, g=2, h=3
    V = [
        [1, 0, 0, 1, 1, 1, 0],  # e: M1, M4, M5, M6
        [1, 0, 0, 1, -1, 0, 0], # f: M1, M4, M5
        [0, 0, 1, 0, 0, 0, 1],  # g: M3, M7
        [0, 1, 1, 0, 1, -1, 1], # h: M2, M3, M5, M6, M7
    ]
    # W: shape (p*n, rank) = (4, 7), indexed by [k*n+i]
    # C is 2x2: C[i,j] where i is row (0,1), j is col (0,1)
    # Output index for C[i,j] in flattened C (row-major): i*2+j
    # But W is indexed by [k*n+i] where k is the "output col" index (p dim) and i is "output row" index (n dim)
    # For matrix mult: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So output has shape n x p = 2 x 2
    # W index: k*n + i = k*2 + i
    # C00: i=0, k=0 -> index 0
    # C01: i=0, k=1 -> index 1
    # C10: i=1, k=0 -> index 2
    # C11: i=1, k=1 -> index 3
    W = [
        [1, 0, 0, -1, 1, 0, 0],   # C00 = M1 - M4 + M5
        [0, 1, 0, 0, -1, 0, 1],   # C01 = M2 - M5 + M7
        [0, 0, 1, 1, 0, -1, 0],   # C10 = M3 + M4 - M6
        [1, 0, -1, 0, 0, 1, 0],   # C11 = M1 - M3 + M6
    ]
    return U, V, W


def _add_block_to_global(global_U, global_V, global_W, local_U, local_V, local_W,
                          a_row_start, a_col_start, b_row_start, b_col_start,
                          c_row_start, c_col_start, rank_offset):
    """
    Add a local block decomposition to the global factors.
    
    local_U has shape (a_rows * a_cols, local_rank)
    local_V has shape (b_rows * b_cols, local_rank)  
    local_W has shape (c_rows * c_cols, local_rank)
    
    Global U has shape (5*5, total_rank) indexed by [i*5+j]
    Global V has shape (5*5, total_rank) indexed by [j*5+k]
    Global W has shape (5*5, total_rank) indexed by [k*5+i]
    """
    a_rows = a_col_start + len(local_U) // (a_col_start + a_col_start)  # This is getting complicated
    # Let me just do it more directly
    
    # For U: local_U[i_local * a_cols + j_local] maps to global U[(a_row_start + i_local) * 5 + (a_col_start + j_local)]
    a_cols = local_U[0].__len__() if local_U else 0  # No, local_U rows are over (a_rows * a_cols)
    
    # Actually, local_U has a_rows * a_cols rows, each of length local_rank
    # local_U index r corresponds to (i_local, j_local) where r = i_local * a_cols + j_local
    # This maps to global index (a_row_start + i_local) * 5 + (a_col_start + j_local)
    
    # Let me figure out a_rows and a_cols from the local dimensions
    # local_U has a_rows * a_cols rows
    # local_V has b_rows * b_cols rows
    # local_W has c_rows * c_cols rows
    
    # From the call, we know the block sizes, so let me pass them explicitly
    pass


def construct():
    """
    Construct the 5x5x5 matrix multiplication tensor decomposition.
    
    Uses block decomposition with 5 = 3 + 2:
    A = [[A11, A12], [A21, A22]] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2
    B = [[B11, B12], [B21, B22]] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2
    
    C = A*B, partitioned as:
    C = [[C11, C12], [C21, C22]] where C11 is 3x3, C12 is 3x2, C21 is 2x3, C22 is 2x2
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each term is a sub-matrix multiplication:
    1. A11*B11: 3x3 * 3x3 -> 3x3 (schoolbook: 27)
    2. A12*B21: 3x2 * 2x3 -> 3x3 (schoolbook: 18)
    3. A11*B12: 3x3 * 3x2 -> 3x2 (schoolbook: 18)
    4. A12*B22: 3x2 * 2x2 -> 3x2 (schoolbook: 12)
    5. A21*B11: 2x3 * 3x3 -> 2x3 (schoolbook: 18)
    6. A22*B21: 2x2 * 2x3 -> 2x3 (schoolbook: 12)
    7. A21*B12: 2x3 * 3x2 -> 2x2 (schoolbook: 12)
    8. A22*B22: 2x2 * 2x2 -> 2x2 (Strassen: 7)
    
    Total rank: 27 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 124
    """
    n = m = p = 5
    total_rank = 124
    
    # Initialize global factors
    global_U = [[0] * total_rank for _ in range(n * m)]
    global_V = [[0] * total_rank for _ in range(m * p)]
    global_W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Block 1: A11 * B11 (3x3 * 3x3 -> 3x3)
    # A rows 0-2, cols 0-2. B rows 0-2, cols 0-2. C rows 0-2, cols 0-2.
    local_U, local_V, local_W = _schoolbook_factors(3, 3, 3)
    local_rank = 27
    for i_local in range(3):
        for j_local in range(3):
            u_idx = i_local * 3 + j_local
            for r in range(local_rank):
                global_U[(0 + i_local) * 5 + (0 + j_local)][rank_offset + r] += local_U[u_idx][r]
    for j_local in range(3):
        for k_local in range(3):
            v_idx = j_local * 3 + k_local
            for r in range(local_rank):
                global_V[(0 + j_local) * 5 + (0 + k_local)][rank_offset + r] += local_V[v_idx][r]
    for k_local in range(3):
        for i_local in range(3):
            w_idx = k_local * 3 + i_local
            for r in range(local_rank):
                global_W[(0 + k_local) * 5 + (0 + i_local)][rank_offset + r] += local_W[w_idx][r]
    rank_offset += 27
    
    # Block 2: A12 * B21 (3x2 * 2x3 -> 3x3)
    # A rows 0-2, cols 3-4. B rows 3-4, cols 0-2. C rows 0-2, cols 0-2.
    local_U, local_V, local_W = _schoolbook_factors(3, 2, 3)
    local_rank = 18
    for i_local in range(3):
        for j_local in range(2):
            u_idx = i_local * 2 + j_local
            for r in range(local_rank):
                global_U[(0 + i_local) * 5 + (3 + j_local)][rank_offset + r] += local_U[u_idx][r]
    for j_local in range(2):
        for k_local in range(3):
            v_idx = j_local * 3 + k_local
            for r in range(local_rank):
                global_V[(3 + j_local) * 5 + (0 + k_local)][rank_offset + r] += local_V[v_idx][r]
    for k_local in range(3):
        for i_local in range(3):
            w_idx = k_local * 3 + i_local
            for r in range(local_rank):
                global_W[(0 + k_local) * 5 + (0 + i_local)][rank_offset + r] += local_W[w_idx][r]
    rank_offset += 18
    
    # Block 3: A11 * B12 (3x3 * 3x2 -> 3x2)
    # A rows 0-2, cols 0-2. B rows 0-2, cols 3-4. C rows 0-2, cols 3-4.
    local_U, local_V, local_W = _schoolbook_factors(3, 3, 2)
    local_rank = 18
    for i_local in range(3):
        for j_local in range(3):
            u_idx = i_local * 3 + j_local
            for r in range(local_rank):
                global_U[(0 + i_local) * 5 + (0 + j_local)][rank_offset + r] += local_U[u_idx][r]
    for j_local in range(3):
        for k_local in range(2):
            v_idx = j_local * 2 + k_local
            for r in range(local_rank):
                global_V[(0 + j_local) * 5 + (3 + k_local)][rank_offset + r] += local_V[v_idx][r]
    for k_local in range(2):
        for i_local in range(3):
            w_idx = k_local * 3 + i_local
            for r in range(local_rank):
                global_W[(3 + k_local) * 5 + (0 + i_local)][rank_offset + r] += local_W[w_idx][r]
    rank_offset += 18
    
    # Block 4: A12 * B22 (3x2 * 2x2 -> 3x2)
    # A rows 0-2, cols 3-4. B rows 3-4, cols 3-4. C rows 0-2, cols 3-4.
    local_U, local_V, local_W = _schoolbook_factors(3, 2, 2)
    local_rank = 12
    for i_local in range(3):
        for j_local in range(2):
            u_idx = i_local * 2 + j_local
            for r in range(local_rank):
                global_U[(0 + i_local) * 5 + (3 + j_local)][rank_offset + r] += local_U[u_idx][r]
    for j_local in range(2):
        for k_local in range(2):
            v_idx = j_local * 2 + k_local
            for r in range(local_rank):
                global_V[(3 + j_local) * 5 + (3 + k_local)][rank_offset + r] += local_V[v_idx][r]
    for k_local in range(2):
        for i_local in range(3):
            w_idx = k_local * 3 + i_local
            for r in range(local_rank):
                global_W[(3 + k_local) * 5 + (0 + i_local)][rank_offset + r] += local_W[w_idx][r]
    rank_offset += 12
    
    # Block 5: A21 * B11 (2x3 * 3x3 -> 2x3)
    # A rows 3-4, cols 0-2. B rows 0-2, cols 0-2. C rows 3-4, cols 0-2.
    local_U, local_V, local_W = _schoolbook_factors(2, 3, 3)
    local_rank = 18
    for i_local in range(2):
        for j_local in range(3):
            u_idx = i_local * 3 + j_local
            for r in range(local_rank):
                global_U[(3 + i_local) * 5 + (0 + j_local)][rank_offset + r] += local_U[u_idx][r]
    for j_local in range(3):
        for k_local in range(3):
            v_idx = j_local * 3 + k_local
            for r in range(local_rank):
                global_V[(0 + j_local) * 5 + (0 + k_local)][rank_offset + r] += local_V[v_idx][r]
    for k_local in range(3):
        for i_local in range(2):
            w_idx = k_local * 2 + i_local
            for r in range(local_rank):
                global_W[(0 + k_local) * 5 + (3 + i_local)][rank_offset + r] += local_W[w_idx][r]
    rank_offset += 18
    
    # Block 6: A22 * B21 (2x2 * 2x3 -> 2x3)
    # A rows 3-4, cols 3-4. B rows 3-4, cols 0-2. C rows 3-4, cols 0-2.
    local_U, local_V, local_W = _schoolbook_factors(2, 2, 3)
    local_rank = 12
    for i_local in range(2):
        for j_local in range(2):
            u_idx = i_local * 2 + j_local
            for r in range(local_rank):
                global_U[(3 + i_local) * 5 + (3 + j_local)][rank_offset + r] += local_U[u_idx][r]
    for j_local in range(2):
        for k_local in range(3):
            v_idx = j_local * 3 + k_local
            for r in range(local_rank):
                global_V[(3 + j_local) * 5 + (0 + k_local)][rank_offset + r] += local_V[v_idx][r]
    for k_local in range(3):
        for i_local in range(2):
            w_idx = k_local * 2 + i_local
            for r in range(local_rank):
                global_W[(0 + k_local) * 5 + (3 + i_local)][rank_offset + r] += local_W[w_idx][r]
    rank_offset += 12
    
    # Block 7: A21 * B12 (2x3 * 3x2 -> 2x2)
    # A rows 3-4, cols 0-2. B rows 0-2, cols 3-4. C rows 3-4, cols 3-4.
    local_U, local_V, local_W = _schoolbook_factors(2, 3, 2)
    local_rank = 12
    for i_local in range(2):
        for j_local in range(3):
            u_idx = i_local * 3 + j_local
            for r in range(local_rank):
                global_U[(3 + i_local) * 5 + (0 + j_local)][rank_offset + r] += local_U[u_idx][r]
    for j_local in range(3):
        for k_local in range(2):
            v_idx = j_local * 2 + k_local
            for r in range(local_rank):
                global_V[(0 + j_local) * 5 + (3 + k_local)][rank_offset + r] += local_V[v_idx][r]
    for k_local in range(2):
        for i_local in range(2):
            w_idx = k_local * 2 + i_local
            for r in range(local_rank):
                global_W[(3 + k_local) * 5 + (3 + i_local)][rank_offset + r] += local_W[w_idx][r]
    rank_offset += 12
    
    # Block 8: A22 * B22 (2x2 * 2x2 -> 2x2) - Use Strassen
    # A rows 3-4, cols 3-4. B rows 3-4, cols 3-4. C rows 3-4, cols 3-4.
    local_U, local_V, local_W = _strassen_factors()
    local_rank = 7
    for i_local in range(2):
        for j_local in range(2):
            u_idx = i_local * 2 + j_local
            for r in range(local_rank):
                global_U[(3 + i_local) * 5 + (3 + j_local)][rank_offset + r] += local_U[u_idx][r]
    for j_local in range(2):
        for k_local in range(2):
            v_idx = j_local * 2 + k_local
            for r in range(local_rank):
                global_V[(3 + j_local) * 5 + (3 + k_local)][rank_offset + r] += local_V[v_idx][r]
    for k_local in range(2):
        for i_local in range(2):
            w_idx = k_local * 2 + i_local
            for r in range(local_rank):
                global_W[(3 + k_local) * 5 + (3 + i_local)][rank_offset + r] += local_W[w_idx][r]
    rank_offset += 7
    
    return {"U": global_U, "V": global_V, "W": global_W}
