"""
Rank-7 exact tensor decomposition for 2x2 matrix multiplication (Strassen).
This is the known optimal rank for 2x2 matrix multiplication.
"""

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns U, V, W matrices for the rank-7 Strassen decomposition.
    
    Indexing convention:
      U[i*m + j, r]  — coefficient for A[i,j] in product r
      V[j*p + k, r]  — coefficient for B[j,k] in product r
      W[k*n + i, r]  — coefficient for C[i,k] in product r  (trace convention)
    
    Tensor identity: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's 7 intermediate products.
    # Each entry: (U_coeffs, V_coeffs, W_coeffs)
    #   U_coeffs: {A_flat_index: coeff}  where A_flat_index = i*m + j
    #   V_coeffs: {B_flat_index: coeff}  where B_flat_index = j*p + k
    #   W_coeffs: {C_flat_index: coeff}  where C_flat_index = k*n + i
    #
    # A flat indices: A00=0, A01=1, A10=2, A11=3
    # B flat indices: B00=0, B01=1, B10=2, B11=3
    # C flat indices (W[k*n+i]): 
    #   k=0: C00=0, C01=1 (Wait, W[k*n+i]. If k=0, i=0 -> 0. If k=0, i=1 -> 1. So C00=0, C10=1? No.)
    #   Let's re-verify the trace convention W[k*n+i].
    #   Usually C[i,k] = sum A[i,j] B[j,k].
    #   The tensor is T[i,j,k].
    #   The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    #   This implies the index of W corresponds to the position in the flattened C matrix.
    #   If C is stored row-major as C[i,k], the flat index is i*p + k.
    #   But the prompt says W[k*n+i]. This is column-major indexing of C? Or is it defining the tensor index order differently?
    #   Let's check the current code's comment:
    #   "C flat indices (W[k*n+i]): C00=0, C10=1, C01=2, C11=3"
    #   If W index is k*n + i:
    #     k=0, i=0 -> 0 (C00)
    #     k=0, i=1 -> 1 (C10) -> This corresponds to C[1,0]
    #     k=1, i=0 -> 2 (C01) -> This corresponds to C[0,1]
    #     k=1, i=1 -> 3 (C11) -> This corresponds to C[1,1]
    #   
    #   Standard matrix multiplication:
    #   C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    #   C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    #   C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    #   C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    #
    #   Let's verify the current code's Strassen mapping against this W indexing.
    #   M0 = (A00+A11)(B00+B11)
    #   Current code: W coeffs {0: 1, 3: 1}.
    #   Index 0 is C[0,0]. Index 3 is C[1,1].
    #   So M0 adds to C[0,0] and C[1,1]. This is correct for Strassen M0.
    #
    #   M1 = (A10+A11)B00
    #   Current code: W coeffs {1: 1, 3: -1}.
    #   Index 1 is C[1,0]. Index 3 is C[1,1].
    #   So M1 adds to C[1,0] and subtracts from C[1,1].
    #   Strassen M1 usually is (A10+A11)B00.
    #   C[1,0] gets M1.
    #   C[1,1] gets -M1.
    #   Let's check standard Strassen:
    #   M1 = (A10 + A11) B00
    #   C10 += M1
    #   C11 -= M1
    #   This matches.
    #
    #   M2 = A00(B01-B11)
    #   Current code: W coeffs {2: 1, 3: 1}.
    #   Index 2 is C[0,1]. Index 3 is C[1,1].
    #   C[0,1] += M2.
    #   C[1,1] += M2.
    #   Standard Strassen:
    #   M2 = A00(B01 - B11)
    #   C01 += M2
    #   C11 += M2
    #   This matches.
    #
    #   M3 = A11(B10-B00)
    #   Current code: W coeffs {0: 1, 1: 1}.
    #   Index 0 is C[0,0]. Index 1 is C[1,0].
    #   C[0,0] += M3.
    #   C[1,0] += M3.
    #   Standard Strassen:
    #   M3 = A11(B10 - B00)
    #   C00 += M3
    #   C10 += M3
    #   This matches.
    #
    #   M4 = (A00+A01)B11
    #   Current code: W coeffs {0: -1, 2: 1}.
    #   Index 0 is C[0,0]. Index 2 is C[0,1].
    #   C[0,0] -= M4.
    #   C[0,1] += M4.
    #   Standard Strassen:
    #   M4 = (A00 + A01) B11
    #   C00 -= M4
    #   C01 += M4
    #   This matches.
    #
    #   M5 = (A10-A00)(B00+B01)
    #   Current code: W coeffs {3: 1}.
    #   Index 3 is C[1,1].
    #   C[1,1] += M5.
    #   Standard Strassen:
    #   M5 = (A10 - A00)(B00 + B01)
    #   C11 += M5
    #   This matches.
    #
    #   M6 = (A01-A11)(B10+B11)
    #   Current code: W coeffs {0: 1}.
    #   Index 0 is C[0,0].
    #   C[0,0] += M6.
    #   Standard Strassen:
    #   M6 = (A01 - A11)(B10 + B11)
    #   C00 += M6
    #   This matches.
    
    strassen_products = [
        # M0
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M1
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # M2
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_c, v_c, w_c) in enumerate(strassen_products):
        for idx, val in u_c.items():
            U[idx][r] = val
        for idx, val in v_c.items():
            V[idx][r] = val
        for idx, val in w_c.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
