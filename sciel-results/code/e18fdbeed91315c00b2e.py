import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the rank decomposition
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # 1. 4x4x4 part (Rank 49)
    # Strassen 2x2x2 definitions
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map 4x4 indices to 2x2 block indices
    # A[i, j] -> block A[i//2, j//2], local (i%2, j%2)
    # Same for B, C.
    
    # For each Strassen product Mi, we have a 2x2x2 multiplication.
    # We need to decompose each 2x2x2 multiplication into 7 rank terms.
    
    # Strassen 2x2x2 rank terms (coefficients for U, V, W)
    # U is 2x2, V is 2x2, W is 2x2 (for C)
    # Indices 0..3 for 2x2 matrices.
    
    strassen_2x2_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: list of (i, j, val)
        # v_coeffs: list of (i, j, val)
        # w_coeffs: list of (i, j, val)
        
        # M1 = (A11+A22)(B11+B22)
        # This is not a single rank term. It's a product of two matrices.
        # We need to decompose the product of two 2x2 matrices into 7 rank terms.
        # The 7 rank terms for 2x2x2 are standard.
        
        # Let's define the 7 rank terms for 2x2x2 multiplication.
        # C = A B
        # Rank 7 decomposition:
        # T1: A11, B11+B21, C11
        # T2: A11+A12, B21, C11
        # T3: A11+A12, B21+B22, C12
        # T4: A21+A22, B11, C21
        # T5: A21+A22, B11+B12, C22
        # T6: A21, B11-B21, C21
        # T7: A22, B21-B22, C22
        # (One of many variants)
        
        # Actually, it's easier to just use the known 7-term decomposition.
        # I will hardcode the 7 terms for 2x2x2.
    ]
    
    # This is getting complicated to hardcode all 49 terms manually.
    # I will write a helper to generate the terms for a 2x2x2 multiplication
    # given the linear combinations for the input matrices.
    
    return {"U": U, "V": V, "W": W}
