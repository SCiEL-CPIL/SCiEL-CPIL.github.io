import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0

def _schoolbook_rank(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222_block(i_off, j_off, k_off, n, m, p):
    """
    Strassen 2x2x2 block with global offsets.
    Returns list of (U_col, V_col, W_col) triples.
    """
    cols = []
    # Strassen 7 multiplications
    # Each multiplication M_c = (linear combo of A entries) * (linear combo of B entries)
    # contributes to specific C entries with specific coefficients.
    
    # M1 = (A[i,j] + A[i+1,j+1]) * (B[j,k] + B[j+1,k+1])
    #   A entries: (i_off, j_off)=1, (i_off+1, j_off+1)=1
    #   B entries: (j_off, k_off)=1, (j_off+1, k_off+1)=1
    #   C contributions:
    #     C[i_off, k_off] += 1 (from A[i,j]*B[j,k])
    #     C[i_off, k_off+1] += 1 (from A[i,j]*B[j+1,k+1])  -- wait no
    #     Actually: (A[i,j]+A[i+1,j+1])*(B[j,k]+B[j+1,k+1])
    #     = A[i,j]*B[j,k] + A[i,j]*B[j+1,k+1] + A[i+1,j+1]*B[j,k] + A[i+1,j+1]*B[j+1,k+1]
    #     A[i,j]*B[j,k] -> C[i,k]
    #     A[i,j]*B[j+1,k+1] -> this is NOT a valid term! B[j+1,k+1] means row j+1, col k+1 of B.
    #     But A[i,j]*B[j+1,k+1] would contribute to C[i, k+1] only if j == j+1, which is false.
    #     
    # I'm confusing myself. Let me use the standard Strassen formula properly.
    # For 2x2 matrices A, B, C:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # 
    # C = A*B means:
    # c11 = a11*b11 + a12*b21
    # c12 = a11*b12 + a12*b22
    # c21 = a21*b11 + a22*b21
    # c22 = a21*b12 + a22*b22
    #
    # Strassen:
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b21 - b22)
    # M4 = a22 * (b11 - b12)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    #
    # c11 = M1 - M4 + M6
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M3 + M7
    
    # In our indexing with 0-based:
    # a11 = A[i_off, j_off], a12 = A[i_off, j_off+1]
    # a21 = A[i_off+1, j_off], a22 = A[i_off+1, j_off+1]
    # b11 = B[j_off, k_off], b12 = B[j_off, k_off+1]
    # b21 = B[j_off+1, k_off], b22 = B[j_off+1, k_off+1]
    # c11 = C[i_off, k_off], c12 = C[i_off, k_off+1]
    # c21 = C[i_off+1, k_off], c22 = C[i_off+1, k_off+1]
    
    def make_col(U_dict, V_dict, W_dict):
        U_col = np.zeros(n * m)
        V_col = np.zeros(m * p)
        W_col = np.zeros(p * n)
        for (i, j) in U_dict:
            U_col[i * m + j] = U_dict[(i, j)]
        for (j, k) in V_dict:
            V_col[j * p + k] = V_dict[(j, k)]
        for (i, k) in W_dict:
            W_col[k * n + i] = W_dict[(i, k)]
        return (U_col, V_col, W_col)
    
    i0, i1 = i_off, i_off + 1
    j0, j1 = j_off, j_off + 1
    k0, k1 = k_off, k_off + 1
    
    # M1 = (a11 + a22) * (b11 + b22)
    U1 = {(i0, j0): 1.0, (i1, j1): 1.0}
    V1 = {(j0, k0): 1.0, (j1, k1): 1.0}
    # c11 += M1, c22 += M1
    W1 = {(i0, k0): 1.0, (i1, k1): 1.0}
    cols.append(make_col(U1, V1, W1))
    
    # M2 = (a21 + a22) * b11
    U2 = {(i1, j0): 1.0, (i1, j1): 1.0}
    V2 = {(j0, k0): 1.0}
    # c21 += M2
    W2 = {(i1, k0): 1.0}
    cols.append(make_col(U2, V2, W2))
    
    # M3 = a11 * (b21 - b22)
    U3 = {(i0, j0): 1.0}
    V3 = {(j1, k0): 1.0, (j1, k1): -1.0}
    # c12 += M3... wait.
    # a11 * b21 -> C[i0, k0]? No. a11 is A[i0,j0], b21 is B[j1,k0].
    # A[i0,j0]*B[j1,k0] contributes to C[i0,k0] only if j0==j1, which is false.
    # So a11*b21 does NOT directly contribute to any C entry in the standard product!
    # 
    # I think I'm making an error. Let me reconsider.
    # In the product C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
    # So A[i0,j0]*B[j1,k0] contributes to C[i0,k0] only if j0 = j1, which is false.
    # Therefore, cross terms like a11*b21 don't appear in the product.
    #
    # But Strassen works by introducing these cross terms and then combining them
    # so that the cross terms cancel out. Let me re-derive carefully.
    #
    # M1 = (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # M2 = (a21+a22)*b11 = a21*b11 + a22*b11
    # M3 = a11*(b21-b22) = a11*b21 - a11*b22
    # M4 = a22*(b11-b12) = a22*b11 - a22*b12
    # M5 = (a11+a12)*b22 = a11*b22 + a12*b22
    # M6 = (a21-a11)*(b11+b12) = a21*b11 + a21*b12 - a11*b11 - a11*b12
    # M7 = (a12-a22)*(b21+b22) = a12*b21 + a12*b22 - a22*b21 - a22*b22
    #
    # c11 = a11*b11 + a12*b21
    # M1 - M4 + M6 = (a11*b11 + a11*b22 + a22*b11 + a22*b22) 
    #              - (a22*b11 - a22*b12) 
    #              + (a21*b11 + a21*b12 - a11*b11 - a11*b12)
    # = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a22*b11 + a22*b12 + a21*b11 + a21*b12 - a11*b11 - a11*b12
    # = a11*b22 + a22*b22 + a22*b12 + a21*b11 + a21*b12 - a11*b12
    # That doesn't look right. Let me just trust the standard formula and verify numerically.
    #
    # Actually, the standard Strassen formulas are correct. Let me just implement them
    # and verify. The key is that each M_c is a product of a linear functional of A
    # and a linear functional of B, and the W coefficients tell us how each M_c
    # contributes to each C entry.
    
    # Let me restart with the correct W assignments based on the standard formulas:
    # c11 = M1 - M4 + M6
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M3 + M7
    
    # So for each M_c, the W coefficients are:
    # M1: c11 += 1, c22 += 1
    # M2: c21 += 1
    # M3: c12 += 1, c22 += -1
    # M4: c11 += -1, c21 += 1
    # M5: c12 += 1
    # M6: c11 += 1
    # M7: c22 += 1
    
    # Now the U and V for each M:
    # M1: U = a11 + a22, V = b11 + b22
    # M2: U = a21 + a22, V = b11
    # M3: U = a11, V = b21 - b22
    # M4: U = a22, V = b11 - b12
    # M5: U = a11 + a12, V = b22
    # M6: U = a21 - a11, V = b11 + b12
    # M7: U = a12 - a22, V = b21 + b22
    
    # Let me rebuild the columns correctly:
    cols = []
    
    # M1
    U1 = {(i0, j0): 1.0, (i1, j1): 1.0}
    V1 = {(j0, k0): 1.0, (j1, k1): 1.0}
    W1 = {(i0, k0): 1.0, (i1, k1): 1.0}
    cols.append(make_col(U1, V1, W1))
    
    # M2
    U2 = {(i1, j0): 1.0, (i1, j1): 1.0}
    V2 = {(j0, k0): 1.0}
    W2 = {(i1, k0): 1.0}
    cols.append(make_col(U2, V2, W2))
    
    # M3
    U3 = {(i0, j0): 1.0}
    V3 = {(j1, k0): 1.0, (j1, k1): -1.0}
    W3 = {(i0, k1): 1.0, (i1, k1): -1.0}
    cols.append(make_col(U3, V3, W3))
    
    # M4
    U4 = {(i1, j1): 1.0}
    V4 = {(j0, k0): 1.0, (j0, k1): -1.0}
    W4 = {(i0, k0): -1.0, (i1, k0): 1.0}
    cols.append(make_col(U4, V4, W4))
    
    # M5
    U5 = {(i0, j0): 1.0, (i0, j1): 1.0}
    V5 = {(j1, k1): 1.0}
    W5 = {(i0, k1): 1.0}
    cols.append(make_col(U5, V5, W5))
    
    # M6
    U6 = {(i1, j0): 1.0, (i0, j0): -1.0}
    V6 = {(j0, k0): 1.0, (j0, k1): 1.0}
    W6 = {(i0, k0): 1.0}
    cols.append(make_col(U6, V6, W6))
    
    # M7
    U7 = {(i0, j1): 1.0, (i1, j1): -1.0}
    V7 = {(j1, k0): 1.0, (j1, k1): 1.0}
    W7 = {(i1, k1): 1.0}
    cols.append(make_col(U7, V7, W7))
    
    return cols

def _schoolbook_221_block(i_off, j_off, k_idx, n, m, p):
    """
    Schoolbook for 2x2x1 block.
    A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_idx] -> C[i_off:i_off+2, k_idx]
    C[i, k_idx] = A[i, j_off]*B[j_off, k_idx] + A[i, j_off+1]*B[j_off+1, k_idx]
    """
    cols = []
    i0, i1 = i_off, i_off + 1
    j0, j1 = j_off, j_off + 1
    k = k_idx
    
    def make_col(U_dict, V_dict, W_dict):
        U_col = np.zeros(n * m)
        V_col = np.zeros(m * p)
        W_col = np.zeros(p * n)
        for (i, j) in U_dict:
            U_col[i * m + j] = U_dict[(i, j)]
        for (j, k) in V_dict:
            V_col[j * p + k] = V_dict[(j, k)]
        for (i, k) in W_dict:
            W_col[k * n + i] = W_dict[(i, k)]
        return (U_col, V_col, W_col)
    
    # Term 1: A[i0, j0] * B[j0, k] -> C[i0, k]
    cols.append(make_col({(i0, j0): 1.0}, {(j0, k): 1.0}, {(i0, k): 1.0}))
    # Term 2: A[i0, j1] * B[j1, k] -> C[i0, k]
    cols.append(make_col({(i0, j1): 1.0}, {(j1, k): 1.0}, {(i0, k): 1.0}))
    # Term 3: A[i1, j0] * B[j0, k] -> C[i1, k]
    cols.append(make_col({(i1, j0): 1.0}, {(j0, k): 1.0}, {(i1, k): 1.0}))
    # Term 4: A[i1, j1] * B[j1, k] -> C[i1, k]
    cols.append(make_col({(i1, j1): 1.0}, {(j1, k): 1.0}, {(i1, k): 1.0}))
    
    return cols

def _schoolbook_1x4x5_block(i_idx, n, m, p):
    """
    Schoolbook for 1x4x5 block.
    A[i_idx, 0:4] * B[0:4, 0:5] -> C[i_idx, 0:5]
    C[i_idx, k] = sum_{j=0}^{3} A[i_idx, j] * B[j, k]
    """
    cols = []
    i = i_idx
    
    def make_col(U_dict, V_dict, W_dict):
        U_col = np.zeros(n * m)
        V_col = np.zeros(m * p)
        W_col = np.zeros(p * n)
        for (i, j) in U_dict:
            U_col[i * m + j] = U_dict[(i, j)]
        for (j, k) in V_dict:
            V_col[j * p + k] = V_dict[(j, k)]
        for (i, k) in W_dict:
            W_col[k * n + i] = W_dict[(i, k)]
        return (U_col, V_col, W_col)
    
    for j in range(m):
        for k in range(p):
            cols.append(make_col({(i, j): 1.0}, {(j, k): 1.0}, {(i, k): 1.0}))
    
    return cols

def _verify_decomposition(U, V, W, n, m, p, seed=42):
    """Verify the decomposition with a random test."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    
    # Compute C using the decomposition
    # C[i,k] = sum_c W[k*n+i, c] * fa_c(A) * fb_c(B)
    # where fa_c(A) = sum_{ii,jj} U[ii*m+jj, c] * A[ii,jj]
    # and fb_c(B) = sum_{jj,kk} V[jj*p+kk, c] * B[jj,kk]
    
    rank = U.shape[1]
    # Vectorized computation
    # fa = U^T @ A_flat, where A_flat is A reshaped to (n*m,)
    # fb = V^T @ B_flat, where B_flat is B reshaped to (m*p,)
    A_flat = A.reshape(n * m)
    B_flat = B.reshape(m * p)
    
    fa = U.T @ A_flat  # shape (rank,)
    fb = V.T @ B_flat  # shape (rank,)
    
    # C[i,k] = sum_c W[k*n+i, c] * fa[c] * fb[c]
    # W_flat = W.reshape(p*n), so W_flat[k*n+i] is the coefficient
    W_flat = W.reshape(p * n)
    
    # C_flat[i*p+k] = sum_c W_flat[k*n+i] * fa[c] * fb[c]
    # This is a bit tricky to vectorize directly. Let's use a loop over k and i.
    C_actual = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            C_actual[i, k] = np.dot(W[k * n + i, :], fa * fb)
    
    diff = np.max(np.abs(C_actual - C_expected))
    return diff < 1e-9, diff

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5:
        U, V, W = _schoolbook_rank(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    columns = []
    
    # Block 1: A[0:2, 0:4] * B[0:4, 0:4] -> C[0:2, 0:4]
    # 4 Strassen 2x2x2 blocks:
    columns.extend(_strassen_222_block(0, 0, 0, n, m, p))
    columns.extend(_strassen_222_block(0, 0, 2, n, m, p))
    columns.extend(_strassen_222_block(0, 2, 0, n, m, p))
    columns.extend(_strassen_222_block(0, 2, 2, n, m, p))
    
    # Block 2: A[0:2, 0:4] * B[0:4, 4:5] -> C[0:2, 4:5]
    # 2 schoolbook 2x2x1 blocks:
    columns.extend(_schoolbook_221_block(0, 0, 4, n, m, p))
    columns.extend(_schoolbook_221_block(0, 2, 4, n, m, p))
    
    # Block 3: A[2:3, 0:4] * B[0:4, 0:5] -> C[2:3, 0:5]
    # 1 schoolbook 1x4x5 block:
    columns.extend(_schoolbook_1x4x5_block(2, n, m, p))
    
    total_rank = len(columns)
    elapsed = time.time() - start_time
    print(f"Constructed rank {total_rank} in {elapsed:.2f}s", file=sys.stderr)
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        U, V, W = _schoolbook_rank(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Stack columns
    U_final = np.column_stack([c[0] for c in columns])
    V_final = np.column_stack([c[1] for c in columns])
    W_final = np.column_stack([c[2] for c in columns])
    
    # Verify with multiple random tests
    is_valid = True
    for seed in [42, 123, 999]:
        ok, diff = _verify_decomposition(U_final, V_final, W_final, n, m, p, seed=seed)
        if not ok:
            is_valid = False
            print(f"Verification failed (seed={seed}): max_diff={diff:.2e}", file=sys.stderr)
            break
    print(f"Verification: {is_valid}", file=sys.stderr)
    
    if not is_valid:
        # Fall back to schoolbook
        U, V, W = _schoolbook_rank(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
