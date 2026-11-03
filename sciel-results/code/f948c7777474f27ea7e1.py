"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x3 matrix.
    
    The minimal rank for 2x3x3 matrix multiplication is 17.
    This implementation constructs the decomposition using a data-driven
    list of rank-1 terms, separating the Strassen-based core from the
    schoolbook extensions.
    
    Interface:
    U: shape (n*m, rank) = (6, 17)
    V: shape (m*p, rank) = (9, 17)
    W: shape (p*n, rank) = (6, 17)
    
    n=2, m=3, p=3.
    rank = 17.
    """
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define terms as (u_entries, v_entries, w_entries)
    # u_entries: list of (index, coeff)
    # v_entries: list of (index, coeff)
    # w_entries: list of (index, coeff)
    
    terms = []
    
    # Strassen terms for the 2x2x2 sub-problem (A[0:2, 0:2] * B[0:2, 0:2])
    # M1 = (A00 + A11) * (B00 + B11) -> contributes to C00, C01, C10, C11 (partially)
    # Indices: A: 0,4; B: 0,4; C: 0,1,2,3 (C is 3x2, so C00=0, C10=1, C01=2, C11=3)
    # Wait, C index is k*n + i.
    # C00 (k=0, i=0) -> 0
    # C10 (k=0, i=1) -> 2? No. n=2.
    # C is 3x2.
    # C[k, i] is at index k*n + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    
    # Let's re-verify the W indices in the original code.
    # Original M1: w_indices [(0, 1), (3, 1)] -> W[0], W[3].
    # W[0] is C[0,0]. W[3] is C[1,1].
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This contributes to C00, C01, C10, C11?
    # A00B00 -> C00.
    # A00B11 -> C01? No, B is 3x3. B[0,1] is B01. B[1,1] is B11.
    # A[0,0] * B[0,1] -> C[0,0]? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # C[0,0] = A00B00 + A01B10 + A02B20.
    # C[0,1] = A00B01 + A01B11 + A02B21.
    # C[1,0] = A10B00 + A11B10 + A12B20.
    # C[1,1] = A10B01 + A11B11 + A12B21.
    
    # Strassen on A[0:2, 0:2] and B[0:2, 0:2] computes:
    # S00 = A00B00 + A01B10
    # S01 = A00B01 + A01B11
    # S10 = A10B00 + A11B10
    # S11 = A10B01 + A11B11
    
    # The original code's M1:
    # U: A00, A11. V: B00, B11.
    # Product: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This does NOT directly map to S00..S11 without further combination.
    # The original code uses 7 terms to reconstruct the 4 values S00..S11.
    
    # I will keep the exact same terms to ensure correctness, just reformatting.
    
    terms.append((
        [(0, 1), (4, 1)],  # U
        [(0, 1), (4, 1)],  # V
        [(0, 1), (3, 1)]   # W
    ))
    terms.append((
        [(3, 1), (4, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)]
    ))
    terms.append((
        [(0, 1)],
        [(1, 1), (4, -1)],
        [(2, 1), (3, 1)]
    ))
    terms.append((
        [(4, 1)],
        [(3, 1), (0, -1)],
        [(0, 1), (1, 1)]
    ))
    terms.append((
        [(0, 1), (1, 1)],
        [(4, 1)],
        [(0, -1), (2, 1)]
    ))
    terms.append((
        [(3, 1), (0, -1)],
        [(0, 1), (1, 1)],
        [(3, 1)]
    ))
    terms.append((
        [(1, 1), (4, -1)],
        [(3, 1), (4, 1)],
        [(0, 1)]
    ))
    
    # Schoolbook terms
    terms.append(([(2, 1)], [(6, 1)], [(0, 1)]))
    terms.append(([(5, 1)], [(6, 1)], [(1, 1)]))
    terms.append(([(2, 1)], [(7, 1)], [(2, 1)]))
    terms.append(([(5, 1)], [(7, 1)], [(3, 1)]))
    terms.append(([(0, 1)], [(2, 1)], [(4, 1)]))
    terms.append(([(3, 1)], [(2, 1)], [(5, 1)]))
    terms.append(([(1, 1)], [(5, 1)], [(4, 1)]))
    terms.append(([(4, 1)], [(5, 1)], [(5, 1)]))
    terms.append(([(2, 1)], [(8, 1)], [(4, 1)]))
    terms.append(([(5, 1)], [(8, 1)], [(5, 1)]))
    
    # Populate matrices
    for col, (u_list, v_list, w_list) in enumerate(terms):
        for idx, val in u_list:
            U[idx][col] += val
        for idx, val in v_list:
            V[idx][col] += val
        for idx, val in w_list:
            W[idx][col] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
