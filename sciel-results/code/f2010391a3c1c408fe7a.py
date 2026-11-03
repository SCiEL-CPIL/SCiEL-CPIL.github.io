"""Matrix multiplication 4x4x8 with Strassen-based rank-98 decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _build_2x2x2_strassen():
    """Build the rank-7 Strassen decomposition for 2x2x2."""
    # Standard Strassen: M1..M7
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a-d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-d)(f+h), M7 = (a-c)(e+g)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    #
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    # (i,j) in {0,1}^2 for A, (j,k) in {0,1}^2 for B, (k,i) in {0,1}^2 for C
    
    # Each rank term r has U_val, V_val, W_val as 4-element vectors
    # U_val[i*2+j], V_val[j*2+k], W_val[k*2+i]
    
    terms = []
    
    # M1 = (a-d)(e+h)
    # U: a=1, d=-1 -> U[0]=1, U[3]=-1
    # V: e=1, h=1 -> V[0]=1, V[3]=1
    # W: M1 contributes to C11 and C22
    # C11 = M1+M4-M5+M7 -> W[0]=1 (k=0,i=0)
    # C22 = M1-M2+M3+M6 -> W[3]=1 (k=1,i=1)
    terms.append(([1, 0, 0, -1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # M2 = (c+d)e
    # U: c=1, d=1 -> U[2]=1, U[3]=1
    # V: e=1 -> V[0]=1
    # W: M2 contributes to C21 and C22
    # C21 = M2+M4 -> W[2]=1 (k=1,i=0)
    # C22 = M1-M2+M3+M6 -> W[3]=-1 (k=1,i=1)
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    
    # M3 = a(f-h)
    # U: a=1 -> U[0]=1
    # V: f=1, h=-1 -> V[1]=1, V[3]=-1
    # W: M3 contributes to C12 and C22
    # C12 = M3+M5 -> W[1]=1 (k=0,i=1)
    # C22 = M1-M2+M3+M6 -> W[3]=1 (k=1,i=1)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    
    # M4 = d(g-e)
    # U: d=1 -> U[3]=1
    # V: g=1, e=-1 -> V[0]=-1, V[2]=1
    # W: M4 contributes to C11 and C21
    # C11 = M1+M4-M5+M7 -> W[0]=1 (k=0,i=0)
    # C21 = M2+M4 -> W[2]=1 (k=1,i=0)
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    
    # M5 = (a+b)h
    # U: a=1, b=1 -> U[0]=1, U[1]=1
    # V: h=1 -> V[3]=1
    # W: M5 contributes to C11 and C12
    # C11 = M1+M4-M5+M7 -> W[0]=-1 (k=0,i=0)
    # C12 = M3+M5 -> W[1]=1 (k=0,i=1)
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # M6 = (c-d)(f+h)
    # U: c=1, d=-1 -> U[2]=1, U[3]=-1
    # V: f=1, h=1 -> V[1]=1, V[3]=1
    # W: M6 contributes to C22
    # C22 = M1-M2+M3+M6 -> W[3]=1 (k=1,i=1)
    terms.append(([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]))
    
    # M7 = (a-c)(e+g)
    # U: a=1, c=-1 -> U[0]=1, U[2]=-1
    # V: e=1, g=1 -> V[0]=1, V[2]=1
    # W: M7 contributes to C11
    # C11 = M1+M4-M5+M7 -> W[0]=1 (k=0,i=0)
    terms.append(([1, 0, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]))
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    return U, V, W


def _build_4x4x4_strassen2():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen applied to 2x2 blocks.
    This uses the standard approach: treat A, B, C as 2x2 blocks of 2x2 submatrices,
    apply Strassen to get 7 products of 2x2 matrices, then recursively apply Strassen.
    Total rank = 7 * 7 = 49.
    """
    n = 4
    # U: 16 x 49, V: 16 x 49, W: 16 x 49
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _build_2x2x2_strassen()
    
    # The 7 Strassen terms at the block level:
    # For block indices (i_block, j_block) of A, (j_block, k_block) of B, (k_block, i_block) of C
    # i_block, j_block, k_block in {0, 1}
    # A_block[ib, jb], B_block[jb, kb], C_block[kb, ib]
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A00 - A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A11)(B01 + B11)
    # M7 = (A00 - A10)(B00 + B10)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 7 block-level terms, we need to compute the product
    # of two 2x2 matrices (the block factors), which gives 4 output blocks.
    # But wait - Strassen gives us the products directly. Let me reconsider.
    
    # Actually, the standard way: apply Strassen to compute C = A*B where A, B are 4x4.
    # Split into 2x2 blocks of 2x2.
    # The 7 M terms are 2x2 matrix products.
    # Each 2x2 matrix product can be done with rank-7 Strassen.
    # So total rank = 7 * 7 = 49.
    
    # But we need to be careful: the M terms are linear combinations of block products.
    # Let me think about this differently.
    
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Strassen:
    # M1 = (A00-A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A11)(B01+B11)
    # M7 = (A00-A10)(B00+B10)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each M is a 2x2 matrix product. To compute a 2x2 matrix product X*Y
    # where X = x1*A00 + x2*A01 + x3*A10 + x4*A11 (linear combo of A blocks)
    # and Y = y1*B00 + y2*B01 + y3*B10 + y4*B11 (linear combo of B blocks)
    
    # This is getting complex. Let me use a different approach:
    # Directly construct the rank-49 decomposition using the known Strassen-2 formula.
    
    # Alternative: Use the fact that rank(4x4x4) <= 49 via recursive Strassen.
    # Let me build it step by step.
    
    # For each of the 7 Strassen terms at block level, the "left factor" is a linear
    # combination of A's 2x2 blocks, and the "right factor" is a linear combination
    # of B's 2x2 blocks.
    
    # Let me define the 7 block-level operations as (left_coeffs, right_coeffs):
    # left_coeffs: coefficients for A00, A01, A10, A11
    # right_coeffs: coefficients for B00, B01, B10, B11
    
    # A block index: (ib, jb) -> local 2x2 position. ib in {0,1}, jb in {0,1}
    # A_flat index: ib*2+jb -> 0=A00, 1=A01, 2=A10, 3=A11
    # B_flat index: jb*2+kb -> 0=B00, 1=B01, 2=B10, 3=B11
    
    block_ops = [
        # (left: [A00, A01, A10, A11], right: [B00, B01, B10, B11])
        ([1, 0, 0, -1], [1, 0, 0, 1]),    # M1 = (A00-A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0]),     # M2 = (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1]),    # M3 = A00(B01-B11)
        ([0, 0, 0, 1], [0, 0, 1, -1]),    # M4 = A11(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1]),     # M5 = (A00+A01)B11
        ([0, 0, 1, -1], [0, 1, 0, 1]),    # M6 = (A10-A11)(B01+B11)
        ([1, 0, -1, 0], [1, 0, 1, 0]),    # M7 = (A00-A10)(B00+B10)
    ]
    
    # C output block combinations:
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    c_combos = [
        [1, 0, 0, 1, -1, 0, 1],    # C00
        [0, 0, 1, 0, 1, 0, 0],     # C01
        [0, 1, 0, 1, 0, 0, 0],     # C10
        [1, -1, 1, 0, 0, 1, 0],    # C11
    ]
    
    # Now, for each of the 7 block operations, we need to compute the 2x2 matrix product.
    # The left factor is a 2x2 matrix L = sum_a left_coeffs[a] * A_block[a]
    # The right factor is a 2x2 matrix R = sum_b right_coeffs[b] * B_block[b]
    
    # To compute L*R as a 2x2 product, we use the rank-7 Strassen decomposition for 2x2.
    # The U2, V2, W2 from _build_2x2x2_strassen() give us:
    # For each rank term t in 0..6:
    #   U2[2i+j, t] gives coefficient of A[i,j] in the t-th Strassen term
    #   V2[2j+k, t] gives coefficient of B[j,k] in the t-th Strassen term
    #   W2[2k+i, t] gives coefficient for output C[i,k]
    
    # So for block operation op with left_coeffs lc and right_coeffs rc:
    # L[i,j] = sum_a lc[a] * A_block[a][i,j] where a = ib*2+jb
    # R[j,k] = sum_b rc[b] * B_block[b][j,k] where b = jb*2+kb
    
    # The 2x2 product L*R has entries (L*R)[i,k] = sum_j L[i,j] * R[j,k]
    # Using Strassen rank-7:
    # (L*R)[i,k] = sum_t U2[2i+j... 
    # Wait, U2 index is i*2+j for the 2x2 matrix.
    
    # For the block-level 2x2 product L*R:
    # L is indexed by (li, lj) where li, lj in {0,1}
    # R is indexed by (lj, lk) where lj, lk in {0,1}
    # U2 index: li*2 + lj
    # V2 index: lj*2 + lk
    # W2 index: lk*2 + li
    
    # L[li, lj] = sum_a lc[a] * A_block[a][li, lj]
    # A_block[a] is at position (a//2, a%2) in the 2x2 block grid of A.
    # So A_block[a][li, lj] is element (li, lj) of the 2x2 submatrix A[a//2, a%2].
    # In the full 4x4 matrix A, this is A[a//2*2+li, a%2*2+lj] = A[2*(a//2)+li, 2*(a%2)+lj]
    
    # Similarly for R.
    
    # So for block operation op with lc, rc, and for each Strassen term t:
    # For each (li, lj) in {0,1}^2:
    #   U_full[row, col] += U2[2*li+lj, t] * lc[a] where a = (block_row, block_col)
    #   But wait, lc[a] multiplies A_block[a], and A_block[a] contributes to L[li,lj].
    #   L[li,lj] = sum_a lc[a] * A[2*(a//2)+li, 2*(a%2)+lj]
    #   So the coefficient of A[2*(a//2)+li, 2*(a%2)+lj] in U is lc[a] * U2[2*li+lj, t]
    
    # Let me just iterate:
    for op_idx in range(7):
        lc, rc = block_ops[op_idx]
        for t in range(7):
            col = op_idx * 7 + t
            # Fill U
            for a in range(4):  # A block index
                if lc[a] != 0:
                    a_row = a // 2  # 0 or 1
                    a_col = a % 2    # 0 or 1
                    for li in range(2):
                        for lj in range(2):
                            u_idx = (a_row * 2 + li) * 4 + (a_col * 2 + lj)
                            U[u_idx, col] += lc[a] * U2[2 * li + lj, t]
            
            # Fill V
            for b in range(4):  # B block index
                if rc[b] != 0:
                    b_row = b // 2  # 0 or 1
                    b_col = b % 2    # 0 or 1
                    for lj in range(2):
                        for lk in range(2):
                            v_idx = (b_row * 2 + lj) * 4 + (b_col * 2 + lk)
                            V[v_idx, col] += rc[b] * V2[2 * lj + lk, t]
            
            # Fill W
            for c_blk in range(4):  # C block index
                c_coeff = c_combos[c_blk][op_idx]
                if c_coeff == 0:
                    continue
                c_row = c_blk // 2  # 0 or 1
                c_col = c_blk % 2    # 0 or 1
                for lk in range(2):
                    for li in range(2):
                        w_idx = (c_row * 2 + lk) * 4 + (c_col * 2 + li)
                        W[w_idx, col] += c_coeff * W2[2 * lk + li, t]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build 4x4x4 rank-49 decomposition
    U44, V44, W44 = _build_4x4x4_strassen2()
    
    # For 4x4x8: C = A * B where A is 4x4, B is 4x8
    # Split B into B1 (4x4, cols 0-3) and B2 (4x4, cols 4-7)
    # C1 = A * B1 (4x4), C2 = A * B2 (4x4)
    # C = [C1 | C2] (4x8)
    
    # U: 16 x 98
    # V: 32 x 98 (rows 0-15 for B1, rows 16-31 for B2)
    # W: 32 x 98 (rows 0-15 for C1, rows 16-31 for C2)
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First half (cols 0-48): C1 = A * B1
    U[:, 0:49] = U44
    V[0:16, 0:49] = V44
    W[0:16, 0:49] = W44
    
    # Second half (cols 49-97): C2 = A * B2
    U[:, 49:98] = U44
    V[16:32, 49:98] = V44
    W[16:32, 49:98] = W44
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
