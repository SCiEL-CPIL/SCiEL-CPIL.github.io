"""Rewritten program for matrix multiplication 2x2 by 2x16 using Strassen-based decomposition."""
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for multiplying a 2x2 matrix A
    by a 2x16 matrix B.
    
    Strategy: Group the 16 columns of B into 8 blocks of 2 columns each.
    For each 2x2 block, use the rank-7 Strassen decomposition.
    Total rank = 8 * 7 = 56.
    
    The tensor for (n,m,p) = (2,2,16) multiplication has:
    - U: shape (n*m, rank) = (4, 56)
    - V: shape (m*p, rank) = (32, 56)  
    - W: shape (p*n, rank) = (32, 56)
    
    Index conventions:
    - U[i*m + j, c] where i in [0,n), j in [0,m), c in [0,rank)
    - V[j*p + k, c] where j in [0,m), k in [0,p), c in [0,rank)
    - W[k*n + i, c] where k in [0,p), i in [0,n), c in [0,rank)
    
    The multiplication tensor T_{i,j,k} = 1 if and only if the product C[i,k] 
    involves A[i,j] * B[j,k].
    
    For Strassen on 2x2 * 2x2, the 7 products are:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B11)
    M4 = A11(B01 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    And the output:
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M7
    """
    n, m, p = DIMENSIONS
    # Group p=16 columns into p/2=8 blocks of size 2
    num_blocks = p // 2  # 8 blocks
    strassen_rank = 7
    rank = num_blocks * strassen_rank  # 56
    
    U = [[0] * rank for _ in range(n * m)]   # (4, 56)
    V = [[0] * rank for _ in range(m * p)]   # (32, 56)
    W = [[0] * rank for _ in range(p * n)]   # (32, 56)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # For each Strassen term t in 0..6:
    # A-side: which (i,j) entries of A and with what coefficients
    # B-side: which (j,k) entries of B and with what coefficients
    # C-side: which (i',k') entries of C and with what coefficients
    
    # Strassen terms: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: list of (i, j, coeff) for A entries
    # b_coeffs: list of (j, k, coeff) for B entries  
    # c_coeffs: list of (i, k, coeff) for C entries
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        {
            'a': [(0, 0, 1), (1, 1, 1)],
            'b': [(0, 0, 1), (1, 1, 1)],
            'c': [(0, 0, 1), (1, 1, 1)]
        },
        # M2 = (A10 + A11)B00
        {
            'a': [(1, 0, 1), (1, 1, 1)],
            'b': [(0, 0, 1)],
            'c': [(1, 0, 1), (1, 1, -1)]
        },
        # M3 = A00(B10 - B11)
        {
            'a': [(0, 0, 1)],
            'b': [(1, 0, 1), (1, 1, -1)],
            'c': [(0, 1, 1), (1, 1, 1)]
        },
        # M4 = A11(B01 - B00)
        {
            'a': [(1, 1, 1)],
            'b': [(0, 1, 1), (0, 0, -1)],
            'c': [(0, 0, 1), (1, 0, 1)]
        },
        # M5 = (A00 + A01)B11
        {
            'a': [(0, 0, 1), (0, 1, 1)],
            'b': [(1, 1, 1)],
            'c': [(0, 0, -1), (0, 1, 1)]
        },
        # M6 = (A10 - A00)(B00 + B01)
        {
            'a': [(1, 0, 1), (0, 0, -1)],
            'b': [(0, 0, 1), (0, 1, 1)],
            'c': []  # M6 doesn't directly appear in standard Strassen output
            # Wait, let me recheck. Standard Strassen:
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            # M6 is not used in the standard 7-product Strassen!
            # Let me use the correct 7 terms.
        },
        # M7 = (A01 - A11)(B10 + B11)
        {
            'a': [(0, 1, 1), (1, 1, -1)],
            'b': [(1, 0, 1), (1, 1, 1)],
            'c': [(0, 0, -1), (1, 1, 1)]
        },
    ]
    
    # Actually, let me use the standard 7-term Strassen correctly.
    # The 7 products are:
    # P1 = (A11 - A00)(B00 + B01)
    # P2 = (A00 + A11)B11
    # P3 = (A00 + A01)B00
    # P4 = (A10 + A11)B10
    # P5 = A00(B01 - B11)
    # P6 = A11(B10 - B00)
    # P7 = (A01 - A11)(B10 + B11)
    #
    # C00 = P5 + P6 + P7 + P2
    # C01 = P1 + P2
    # C10 = P3 + P4
    # C11 = P1 - P4 + P5 + P3
    
    # Let me redefine properly
    strassen_terms = [
        # P1 = (A11 - A00)(B00 + B01)
        {
            'a': [(1, 1, 1), (0, 0, -1)],
            'b': [(0, 0, 1), (0, 1, 1)],
            'c': [(0, 1, 1), (1, 1, 1)]
        },
        # P2 = (A00 + A11)B11
        {
            'a': [(0, 0, 1), (1, 1, 1)],
            'b': [(1, 1, 1)],
            'c': [(0, 0, 1), (0, 1, 1)]
        },
        # P3 = (A00 + A01)B00
        {
            'a': [(0, 0, 1), (0, 1, 1)],
            'b': [(0, 0, 1)],
            'c': [(1, 0, 1), (1, 1, 1)]
        },
        # P4 = (A10 + A11)B10
        {
            'a': [(1, 0, 1), (1, 1, 1)],
            'b': [(1, 0, 1)],
            'c': [(1, 0, 1), (1, 1, -1)]
        },
        # P5 = A00(B01 - B11)
        {
            'a': [(0, 0, 1)],
            'b': [(0, 1, 1), (1, 1, -1)],
            'c': [(0, 0, 1), (1, 1, 1)]
        },
        # P6 = A11(B10 - B00)
        {
            'a': [(1, 1, 1)],
            'b': [(1, 0, 1), (0, 0, -1)],
            'c': [(0, 0, 1), (1, 0, 1)]
        },
        # P7 = (A01 - A11)(B10 + B11)
        {
            'a': [(0, 1, 1), (1, 1, -1)],
            'b': [(1, 0, 1), (1, 1, 1)],
            'c': [(0, 0, 1), (1, 1, -1)]
        },
    ]
    
    # Now fill in U, V, W for each block and each Strassen term
    for block in range(num_blocks):
        for t in range(strassen_rank):
            col = block * strassen_rank + t
            term = strassen_terms[t]
            
            # Fill U: U[i*m + j, col] += coeff_a
            for (i, j, coeff) in term['a']:
                U[i * m + j][col] += coeff
            
            # Fill V: V[j*p + k, col] += coeff_b
            # k is the column index within this block: k = block*2 + local_k
            for (j, local_k, coeff) in term['b']:
                k = block * 2 + local_k
                V[j * p + k][col] += coeff
            
            # Fill W: W[k*n + i, col] += coeff_c
            # k is the column index in C, i is the row index in C
            for (i, local_k, coeff) in term['c']:
                k = block * 2 + local_k
                W[k * n + i][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
