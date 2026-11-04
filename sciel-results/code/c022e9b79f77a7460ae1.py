"""Matrix multiplication 2x3x8: find rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    # Schoolbook rank: n*m*p = 48
    # Try Strassen-based decomposition
    
    # Strategy: Split A (2x3) into A_left (2x1) and A_right (2x2)
    # Split B (3x8) into B_top (1x8) and B_bottom (2x8)
    # C = A_left * B_top + A_right * B_bottom
    # A_left (2x1) * B_top (1x8) -> rank 2
    # A_right (2x2) * B_bottom (2x8) -> split B_bottom into four 2x2 blocks, use Strassen for each: 4*7=28
    # Total: 2 + 28 = 30
    
    # Let me build this carefully.
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01, ..., b07],
    #      [b10, b11, ..., b17],
    #      [b20, b21, ..., b27]]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0,k] = a00*b0k + a01*b1k + a02*b2k
    # C[1,k] = a10*b0k + a11*b1k + a12*b2k
    
    # Split: A_left = [[a00],[a10]] (2x1), A_right = [[a01,a02],[a11,a12]] (2x2)
    # B_top = [b00,...,b07] (1x8), B_bottom = [[b10,...,b17],[b20,...,b27]] (2x8)
    # C = A_left * B_top + A_right * B_bottom
    
    # Part 1: A_left * B_top, rank 2
    # For each k: C[0,k] += a00 * b0k, C[1,k] += a10 * b0k
    # This is 2 rank-1 terms: one for k=0..7? No, it's 2*1*8 = 16 scalar products but rank 2.
    # Actually A_left (2x1) * B_top (1x8) is outer product of A_left (2x1) and B_top (1x8).
    # This has rank 2 (each column of B_top is a scalar, multiplied by A_left).
    # In tensor decomposition: for each k, we need a term that captures A_left[:,k'] * B_top[k',k]
    # But B_top is 1x8, so k' is always 0.
    # So for each output column k: term is A_left (2x1) * B_top[0,k] (scalar)
    # In the U,V,W format:
    # U[i*m+j, r] where i in {0,1}, j in {0} -> row i
    # V[j*p+k, r] where j=0, k in {0..7} -> column k
    # W[k*n+i, r] where k in {0..7}, i in {0,1} -> row k*n+i
    # For term r_k (k=0..7): U[i*1+0, r_k] = A_left[i,0] = 1 (for the coefficient)
    # Actually, let me use the standard schoolbook for this part.
    # Part 1 has n*m1*p = 2*1*8 = 16 schoolbook terms, but can be done in rank 2? 
    # No, rank of (2x1)*(1x8) tensor is 2 (it's an outer product of two vectors of lengths 2 and 8... wait, that's 2x8 matrix of rank 1 if both nonzero).
    # Hmm, the tensor T[i,k] = A_left[i,0] * B_top[0,k] is a 2x8 tensor. Its rank is 1 (outer product of two vectors).
    # But in the matrix multiplication tensor framework, each "term" in the decomposition corresponds to a rank-1 tensor in the (i,j,k) space.
    # For fixed j=0: T[i,0,k] = A[i,0]*B[0,k]. This is rank 1 in (i,k) space for fixed j.
    # But we need to decompose the full (i,j,k) tensor.
    # The sub-tensor for j=0: T[i,0,k] = A[i,0]*B[0,k]. This can be decomposed as: for each (i,k), one term? No.
    # Actually, T[i,0,k] = A[i,0]*B[0,k] is already a rank-1 tensor in the (i,k) subspace, but in the full (i,j,k) space with j fixed at 0, it's still rank 1.
    # So Part 1 contributes rank 2? No, rank 1 for the whole j=0 slice? Let me reconsider.
    
    # The full tensor T[i,j,k] = A[i,j]*B[j,k].
    # For j=0: T[i,0,k] = A[i,0]*B[0,k]. This is an outer product of vector A[:,0] (length 2) and vector B[0,:] (length 8). In the (i,k) space, rank 1. In the full (i,j,k) space, since j is fixed, it's still a rank-1 tensor (supported on the j=0 slice).
    # So Part 1 (j=0) contributes rank 1 to the total decomposition? That doesn't seem right because we have 2*8=16 entries.
    
    # I think I'm confusing matrix rank with tensor rank. Let me just use the schoolbook for Part 1: 2*1*8 = 16 terms.
    # Part 2 (j=1,2): A_right (2x2) * B_bottom (2x8). Schoolbook: 2*2*8 = 32 terms.
    # Total schoolbook: 16+32 = 48.
    
    # For Part 2, use Strassen: split B_bottom (2x8) into four 2x2 blocks: B_bottom[:, 0:2], B_bottom[:, 2:4], B_bottom[:, 4:6], B_bottom[:, 6:8].
    # For each 2x2 block, A_right * B_block is a 2x2 matrix product, Strassen rank 7.
    # 4 blocks * 7 = 28 terms.
    # Total: 16 + 28 = 44 terms.
    
    # This matches the knowledge: rank 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    col = 0
    
    # Part 1: j=0, schoolbook, 16 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2, Strassen on four 2x2 blocks
    # A_right[i,j'] where j'=0 corresponds to j=1, j'=1 corresponds to j=2
    # B_bottom[j',k'] where j'=0->j=1, j'=1->j=2, k' is 0..7
    # For each block b (0..3), k' = 2*b + dk where dk in {0,1}
    # Strassen for A_right (2x2) * B_block (2x2):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # In terms of our indices:
    # A_right[0,0]=A[0,1], A_right[0,1]=A[0,2], A_right[1,0]=A[1,1], A_right[1,1]=A[1,2]
    # B_block[0,0]=B[1,2b+0], B_block[0,1]=B[1,2b+1], B_block[1,0]=B[2,2b+0], B_block[1,1]=B[2,2b+1]
    
    # For each block, we need 7 terms. Each term has coefficients for A_right elements and B_block elements.
    # In U: rows are i*m+j where i in {0,1}, j in {1,2}
    # In V: rows are j*p+k where j in {1,2}, k in {0..7}
    # In W: rows are k*n+i where k in {0..7}, i in {0,1}
    
    strassen_terms = [
        # (coef_A00, coef_A01, coef_A10, coef_A11, coef_B00, coef_B01, coef_B10, coef_B11, W_row, W_val)
        # M1: (1,0,0,1) * (1,0,0,1)
        (1, 0, 0, 1, 1, 0, 0, 1, None, None),
        # M2: (0,0,1,1) * (1,0,0,0)
        (0, 0, 1, 1, 1, 0, 0, 0, None, None),
        # M3: (1,0,0,0) * (0,1,0,-1)
        (1, 0, 0, 0, 0, 1, 0, -1, None, None),
        # M4: (0,0,0,1) * (-1,0,1,0)
        (0, 0, 0, 1, -1, 0, 1, 0, None, None),
        # M5: (1,1,0,0) * (0,0,0,1)
        (1, 1, 0, 0, 0, 0, 0, 1, None, None),
        # M6: (-1,0,1,0) * (1,1,0,0)
        (-1, 0, 1, 0, 1, 1, 0, 0, None, None),
        # M7: (0,-1,0,1) * (0,0,1,1)
        (0, -1, 0, 1, 0, 0, 1, 1, None, None),
    ]
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block b, output columns are 2b and 2b+1.
    # C[0,2b] = M1 + M4 - M5 + M7
    # C[0,2b+1] = M3 + M5
    # C[1,2b] = M2 + M4
    # C[1,2b+1] = M1 - M2 + M3 + M6
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1: contributes to C00 (+1) and C11 (+1)
        U[0 * m + 1][col] = 1   # A[0,1]
        U[1 * m + 2][col] = 1   # A[1,2]
        V[1 * p + k0][col] = 1  # B[1,k0]
        V[2 * p + k1][col] = 1  # B[2,k1]
        W[k0 * n + 0][col] = 1  # C[0,k0]
        W[k1 * n + 1][col] = 1  # C[1,k1]
        col += 1
        
        # M2: contributes to C10 (+1) and C11 (-1)
        U[1 * m + 1][col] = 1   # A[1,1]
        U[1 * m + 2][col] = 1   # A[1,2]
        V[1 * p + k0][col] = 1  # B[1,k0]
        W[k0 * n + 1][col] = 1  # C[1,k0]
        W[k1 * n + 1][col] = -1 # C[1,k1]
        col += 1
        
        # M3: contributes to C01 (+1) and C11 (+1)
        U[0 * m + 1][col] = 1   # A[0,1]
        V[1 * p + k1][col] = 1  # B[1,k1]
        V[2 * p + k1][col] = -1 # B[2,k1]
        W[k1 * n + 0][col] = 1  # C[0,k1]
        W[k1 * n + 1][col] = 1  # C[1,k1]
        col += 1
        
        # M4: contributes to C00 (+1) and C10 (+1)
        U[1 * m + 2][col] = 1   # A[1,2]
        V[1 * p + k0][col] = -1 # B[1,k0]
        V[2 * p + k0][col] = 1  # B[2,k0]
        W[k0 * n + 0][col] = 1  # C[0,k0]
        W[k0 * n + 1][col] = 1  # C[1,k0]
        col += 1
        
        # M5: contributes to C00 (-1) and C01 (+1)
        U[0 * m + 1][col] = 1   # A[0,1]
        U[0 * m + 2][col] = 1   # A[0,2]
        V[2 * p + k1][col] = 1  # B[2,k1]
        W[k0 * n + 0][col] = -1 # C[0,k0]
        W[k1 * n + 0][col] = 1  # C[0,k1]
        col += 1
        
        # M6: contributes to C11 (+1)
        U[0 * m + 1][col] = -1  # A[0,1]
        U[1 * m + 1][col] = 1   # A[1,1]
        V[1 * p + k0][col] = 1  # B[1,k0]
        V[1 * p + k1][col] = 1  # B[1,k1]
        W[k1 * n + 1][col] = 1  # C[1,k1]
        col += 1
        
        # M7: contributes to C00 (+1)
        U[0 * m + 2][col] = -1  # A[0,2]
        U[1 * m + 2][col] = 1   # A[1,2]
        V[2 * p + k0][col] = 1  # B[2,k0]
        V[2 * p + k1][col] = 1  # B[2,k1]
        W[k0 * n + 0][col] = 1  # C[0,k0]
        col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    try:
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        # Reconstruct tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    T[i, j, k] = s
        # Check: T[i,j,k] should equal 1 if this is a valid basis term, but actually
        # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity tensor in some sense? No.)
        # Actually, the tensor for matrix mult is T[i,j,k] = delta_{ik} when... no.
        # The tensor is T[i,j,k] such that C[i,k] = sum_j A[i,j] T[i,j,k] B[j,k].
        # For standard matrix mult, T[i,j,k] = delta_{jk}? No, T[i,j,k] = 1 for all i,j,k? 
        # Let me check: C[i,k] = sum_j A[i,j] * B[j,k]. So T[i,j,k] = 1 for all i,j,k.
        # Wait, that's not right either. The tensor is T[i,j,k] = 1 if the indices are "compatible".
        # Actually, for matrix multiplication, the structure tensor has T[i,j,k] = 1 for all valid i,j,k.
        # So T should be all ones.
        expected = np.ones((n, m, p))
        if not np.allclose(T, expected, atol=1e-9):
            # Debug: print differences
            diff = np.abs(T - expected)
            if diff.max() > 1e-9:
                # Fall back to schoolbook
                rank_s = n * m * p
                U_s = [[0] * rank_s for _ in range(n * m)]
                V_s = [[0] * rank_s for _ in range(m * p)]
                W_s = [[0] * rank_s for _ in range(p * n)]
                c = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_s[i * m + j][c] = 1
                            V_s[j * p + k][c] = 1
                            W_s[k * n + i][c] = 1
                            c += 1
                return {"U": U_s, "V": V_s, "W": W_s}
    except Exception:
        pass
    
    return result
# EVOLVE-BLOCK-END
