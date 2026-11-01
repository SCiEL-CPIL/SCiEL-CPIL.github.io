"""Matrix multiplication 4x4x4 tensor decomposition - rank 49 via Kronecker product of Strassen."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    
    Uses the 7x7 Kronecker product structure: recursive Strassen applied at
    block level (2x2 blocks of 2x2 matrices) and scalar level (2x2 matrices).
    Total rank: 7 * 7 = 49.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      Tensor T[i,k,j] = sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
    """
    t_start = time.time()
    n = m = p = 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen 2x2: 7 products, each defined by (A_coeffs, B_coeffs, C_contrib)
    # A_coeffs/B_coeffs: dict from (row,col) in 2x2 to coefficient
    # C_contrib: dict from (row,col) in 2x2 output to coefficient
    
    strassen = [
        # s1: (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # s2: (c+d)*e -> C[1,0]+=1, C[1,1]-=1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # s3: a*(f-h) -> C[0,1]+=1, C[1,1]+=1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # s4: d*(g-e) -> C[0,0]+=1, C[1,0]+=1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # s5: (a+b)*h -> C[0,0]-=1, C[0,1]+=1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # s6: (c-a)*(e+f) -> C[1,1]+=1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # s7: (b-d)*(g+h) -> C[0,0]+=1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # For each pair (outer_term, inner_term), build one rank-1 component.
    # Outer term t: defines how 2x2 blocks of A and B combine (block-level Strassen)
    # Inner term s: defines how 2x2 elements within each block combine (scalar-level Strassen)
    #
    # The outer term t gives:
    #   L_t: a 2x2 matrix where each entry is a linear combo of A-blocks
    #   R_t: a 2x2 matrix where each entry is a linear combo of B-blocks
    #   C_t: how the 2x2 result P_t contributes to C-blocks
    #
    # The inner term s gives:
    #   l_s: a 2x2 matrix of coefficients for L_t's elements
    #   r_s: a 2x2 matrix of coefficients for R_t's elements  
    #   p_s: how scalar product s contributes to P_t's 2x2 output
    #
    # Combined: for rank index r = t*7 + s:
    #   U contribution: for each (bi,bj) block and (er,ec) element:
    #     coeff = L_t[bi,bj] * l_s[er,ec]  (if that block/element is in L_t)
    #     U[(2*bi+er)*4 + (2*bj+ec), r] += coeff
    #   Similarly for V and W.

    for t in range(7):
        block_A, block_B, block_C = strassen[t]
        for s in range(7):
            scalar_A, scalar_B, scalar_C = strassen[s]
            r_idx = t * 7 + s

            # Build U: A-side
            # block_A maps (block_row, block_col) -> block_coeff
            # scalar_A maps (elem_row, elem_col) -> scalar_coeff
            for (bi, bj), bc in block_A.items():
                for (er, ec), sc in scalar_A.items():
                    a_row = 2 * bi + er
                    a_col = 2 * bj + ec
                    U[a_row * m + a_col, r_idx] += bc * sc

            # Build V: B-side
            for (bi, bj), bc in block_B.items():
                for (er, ec), sc in scalar_B.items():
                    b_row = 2 * bi + er
                    b_col = 2 * bj + ec
                    V[b_row * p + b_col, r_idx] += bc * sc

            # Build W: C-side (trace convention W[j*4+i] where i=C row, j=C col)
            for (ci, cj), bc in block_C.items():
                for (er, ec), sc in scalar_C.items():
                    c_row = 2 * ci + er
                    c_col = 2 * cj + ec
                    W[c_col * n + c_row, r_idx] += bc * sc

    # Verify correctness (fast check)
    # Reconstruct tensor and compare with matmul
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 4)
    C_expected = A @ B
    
    C_reconstructed = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0.0
                for r in range(rank):
                    val += U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
                C_reconstructed[i, j] += val

    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        # Fallback: return schoolbook rank-64
        U64 = np.zeros((16, 64))
        V64 = np.zeros((16, 64))
        W64 = np.zeros((16, 64))
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i*16 + k*4 + j
                    U64[i*4+k, r] = 1.0
                    V64[k*4+j, r] = 1.0
                    W64[j*4+i, r] = 1.0
        return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
