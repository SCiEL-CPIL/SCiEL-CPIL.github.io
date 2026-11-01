"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for pairs of columns and schoolbook for the remainder.
    # p = 13. 6 pairs (0-1, 2-3, 4-5, 6-7, 8-9, 10-11) and 1 single (12).
    # Rank per pair (Strassen 2x2x2) is 7.
    # Rank per single (Schoolbook 2x2x1) is 4.
    # Total rank = 6 * 7 + 4 = 46.
    
    rank = 6 * 7 + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1
        
    # Indices for A
    # A[0,0] -> 0
    # A[0,1] -> 1
    # A[1,0] -> 2
    # A[1,1] -> 3
    
    # Process pairs of columns
    for r in range(6):
        k0 = 2 * r
        k1 = 2 * r + 1
        
        # B indices
        # B[0, k0] -> 0 * 13 + k0 = k0
        # B[1, k0] -> 1 * 13 + k0 = 13 + k0
        # B[0, k1] -> k1
        # B[1, k1] -> 13 + k1
        
        # C indices
        # C[0, k0] -> k0 * 2 + 0 = 2 * k0
        # C[1, k0] -> k0 * 2 + 1 = 2 * k0 + 1
        # C[0, k1] -> k1 * 2 + 0 = 2 * k1
        # C[1, k1] -> k1 * 2 + 1 = 2 * k1 + 1
        
        # Strassen M1 = (A00 + A11) * (B0k0 + B1k0) ... wait, Strassen is for 2x2 * 2x2.
        # Here B is 2x2 block:
        # [ B0k0, B0k1 ]
        # [ B1k0, B1k1 ]
        # Let e = B0k0, f = B0k1, g = B1k0, h = B1k1.
        
        # M1 = (a+d)(e+g)
        # a=A00, d=A11, e=B0k0, g=B1k0
        # This term contributes to C0k0 and C1k1?
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map the linear combinations of A and B to the U and V matrices.
        # Each "M" is a product of a linear combo of A and a linear combo of B.
        # So for each M, we add 1 term to the decomposition.
        
        # M1: (A00 + A11) * (B0k0 + B1k0)
        # U indices: A00 (0), A11 (3). Both coeff 1.
        # V indices: B0k0 (k0), B1k0 (13+k0). Both coeff 1.
        # W indices: 
        #   C0k0 (2*k0) gets +1
        #   C1k1 (2*k1+1) gets +1
        # Wait, M1 contributes to C00 (+1) and C11 (+1).
        # So W[2*k0][col] = 1, W[2*k1+1][col] = 1.
        
        U[0][col] = 1
        U[3][col] = 1
        V[k0][col] = 1
        V[13 + k0][col] = 1
        W[2 * k0][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1
        
        # M2: (A10 + A11) * B0k0
        # U: A10 (2), A11 (3). Coeff 1.
        # V: B0k0 (k0). Coeff 1.
        # W: C1k0 (2*k0+1) gets +1.
        
        U[2][col] = 1
        U[3][col] = 1
        V[k0][col] = 1
        W[2 * k0 + 1][col] = 1
        col += 1
        
        # M3: A00 * (B0k1 - B1k1)
        # U: A00 (0). Coeff 1.
        # V: B0k1 (k1) Coeff 1, B1k1 (13+k1) Coeff -1.
        # W: C0k1 (2*k1) gets +1.
        
        U[0][col] = 1
        V[k1][col] = 1
        V[13 + k1][col] = -1
        W[2 * k1][col] = 1
        col += 1
        
        # M4: A11 * (B1k1 - B0k1)
        # U: A11 (3). Coeff 1.
        # V: B1k1 (13+k1) Coeff 1, B0k1 (k1) Coeff -1.
        # W: C0k0 (2*k0) gets +1, C1k0 (2*k0+1) gets +1.
        
        U[3][col] = 1
        V[13 + k1][col] = 1
        V[k1][col] = -1
        W[2 * k0][col] = 1
        W[2 * k0 + 1][col] = 1
        col += 1
        
        # M5: (A00 + A01) * B1k1
        # U: A00 (0), A01 (1). Coeff 1.
        # V: B1k1 (13+k1). Coeff 1.
        # W: C0k0 (2*k0) gets -1, C0k1 (2*k1) gets +1.
        
        U[0][col] = 1
        U[1][col] = 1
        V[13 + k1][col] = 1
        W[2 * k0][col] = -1
        W[2 * k1][col] = 1
        col += 1
        
        # M6: (A10 - A00) * (B0k0 + B0k1)
        # U: A10 (2) Coeff 1, A00 (0) Coeff -1.
        # V: B0k0 (k0) Coeff 1, B0k1 (k1) Coeff 1.
        # W: C1k1 (2*k1+1) gets +1.
        
        U[2][col] = 1
        U[0][col] = -1
        V[k0][col] = 1
        V[k1][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1
        
        # M7: (A01 - A11) * (B1k0 + B1k1)
        # U: A01 (1) Coeff 1, A11 (3) Coeff -1.
        # V: B1k0 (13+k0) Coeff 1, B1k1 (13+k1) Coeff 1.
        # W: C0k0 (2*k0) gets +1.
        
        U[1][col] = 1
        U[3][col] = -1
        V[13 + k0][col] = 1
        V[13 + k1][col] = 1
        W[2 * k0][col] = 1
        col += 1

    # Process last single column (k=12)
    k = 12
    # Schoolbook:
    # C0k = A00 B0k + A01 B1k
    # C1k = A10 B0k + A11 B1k
    
    # Term 1: A00 B0k -> C0k
    U[0][col] = 1
    V[k][col] = 1
    W[2 * k][col] = 1
    col += 1
    
    # Term 2: A01 B1k -> C0k
    U[1][col] = 1
    V[13 + k][col] = 1
    W[2 * k][col] = 1
    col += 1
    
    # Term 3: A10 B0k -> C1k
    U[2][col] = 1
    V[k][col] = 1
    W[2 * k + 1][col] = 1
    col += 1
    
    # Term 4: A11 B1k -> C1k
    U[3][col] = 1
    V[13 + k][col] = 1
    W[2 * k + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
