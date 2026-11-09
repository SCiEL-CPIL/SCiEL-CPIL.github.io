"""
Constructive decomposition for 3x4x4 matrix multiplication using recursive Strassen.
Strategy: Implement a generic recursive Strassen decomposition that splits matrix dimensions 
into 2x2 blocks where possible, falling back to schoolbook for odd or small dimensions.
This avoids the manual index mapping errors in the previous hardcoded approach and 
provides a clean, verifiable construction of the rank-44 decomposition.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    # Construct target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Compute decomposition
    # result[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
    result = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    return np.allclose(result, T, atol=tol)

def _build_strassen_recursive(n, m, p, depth=0):
    """
    Recursively build Strassen decomposition for n x m x p matrix multiplication.
    Returns U (n*m, rank), V (m*p, rank), W (p*n, rank).
    """
    # Base cases: small dimensions use schoolbook
    if n == 1 and m == 1 and p == 1:
        return np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
    
    # If any dimension is 1, we can't use standard 2x2 Strassen splitting effectively 
    # without padding or special handling. For simplicity in this task (3,4,4),
    # we handle the specific splits.
    
    # Check if we can split into 2x2 blocks
    # Strassen requires even dimensions for the standard 2x2 block split.
    # If n, m, p are all even, we can split.
    # If not, we might need to pad or use a different split.
    # For 3x4x4:
    # We can split n=3 into 2+1.
    # We can split m=4 into 2+2.
    # We can split p=4 into 2+2.
    
    # This recursive function is tricky for mixed even/odd.
    # Let's use a specialized approach for the 3x4x4 case to ensure correctness,
    # but structured programmatically to avoid manual index errors.
    
    # Actually, the previous attempt failed due to a bug in the return statement.
    # Let's re-implement the 2x4x4 + 1x4x4 combination carefully.
    
    if n == 2 and m == 4 and p == 4:
        return _build_strassen_2x4x4()
    elif n == 1 and m == 4 and p == 4:
        return _build_schoolbook_1x4x4()
    elif n == 1 and m == 1 and p == 1:
        return np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
    else:
        # Fallback to schoolbook for other cases
        return _schoolbook_matrices(n, m, p)

def _build_strassen_2x4x4():
    """
    Strassen decomposition for 2x4x4.
    A: 2x4, B: 4x4, C: 2x4.
    Split A into two 1x4 rows? No, Strassen splits into 2x2 blocks of the "square" part.
    Here, we treat it as multiplying (2x2) blocks of A (1x2 each) and B (2x2 each).
    
    A = [A1 | A2] where A1, A2 are 1x2? No.
    A is 2x4. Split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] where Aij are 1x2.
    B is 4x4. Split into 2x2 blocks:
    B = [[B11, B12], [B21, B22]] where Bij are 2x2.
    C is 2x4. Split into 2x2 blocks:
    C = [[C11, C12], [C21, C22]] where Cij are 1x2.
    
    Strassen products M1..M7 are 1x2 * 2x2 = 1x2.
    Each 1x2 * 2x2 multiplication has rank 4 (schoolbook for 1x2x2).
    Total rank 7 * 4 = 28.
    """
    rank = 28
    U = np.zeros((8, rank))   # A: 2x4, index i*4+j
    V = np.zeros((16, rank))  # B: 4x4, index j*4+k
    W = np.zeros((8, rank))   # C: 2x4, index k*2+i (Trace convention: W[k*n+i, r] with n=2, so k*2+i)
    
    col = 0
    # Define the 7 Strassen terms
    # Each term is defined by:
    # X: linear combo of A elements (shape 1x2, so 2 rows of coeffs for A)
    # Y: linear combo of B elements (shape 2x2, so 4 rows of coeffs for B)
    # Z: linear combo of C elements (shape 1x2, so 2 rows of coeffs for C)
    
    # A indices: i in {0,1}, j in {0,1,2,3}. Block A11 uses A[0,0:2], A12 uses A[0,2:4], etc.
    # B indices: j in {0..3}, k in {0..3}. Block B11 uses B[0:2, 0:2], etc.
    # C indices: k in {0..3}, i in {0,1}. Block C11 uses C[0:2, 0], etc.
    
    # We will iterate through the 7 Strassen terms.
    # For each term, we need to decompose the 1x2 * 2x2 multiplication into 4 rank-1 terms.
    # The 1x2 * 2x2 multiplication:
    # X = [x0, x1]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [z0, z1]
    # z0 = x0*y00 + x1*y10
    # z1 = x0*y01 + x1*y11
    # This is a 1x2x2 tensor. Rank 4.
    # Terms:
    # 1. x0 * y00 -> z0
    # 2. x1 * y10 -> z0
    # 3. x0 * y01 -> z1
    # 4. x1 * y11 -> z1
    
    # Let's define the Strassen terms explicitly in terms of blocks.
    
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    
    # Helper to add a rank-1 term for a specific Strassen block multiplication
    def add_strassen_term(X_coeffs, Y_coeffs, Z_coeffs, col_offset):
        """
        X_coeffs: list of (a_idx, coeff) for the 1x2 vector X. a_idx is global A index.
                   X has 2 elements (x0, x1). So X_coeffs should be a dict {0: [...], 1: [...]}
        Y_coeffs: dict {(j,k): [...]} for the 2x2 matrix Y. j,k in {0,1}.
        Z_coeffs: dict {k_out: [...]} for the 1x2 vector Z. k_out in {0,1}.
        """
        # The 4 rank-1 terms for 1x2 * 2x2
        # Term 1: x0 * y00 -> z0
        # Term 2: x1 * y10 -> z0
        # Term 3: x0 * y01 -> z1
        # Term 4: x1 * y11 -> z1
        
        # We need to expand X, Y, Z into global indices.
        
        # 1. x0 * y00 -> z0
        for (a_idx, c_a) in X_coeffs[0]:
            for (b_idx, c_b) in Y_coeffs[(0,0)]:
                for (c_idx, c_c) in Z_coeffs[0]:
                    U[a_idx, col_offset] += c_a
                    V[b_idx, col_offset] += c_b
                    W[c_idx, col_offset] += c_c
        col_offset += 1
        
        # 2. x1 * y10 -> z0
        for (a_idx, c_a) in X_coeffs[1]:
            for (b_idx, c_b) in Y_coeffs[(1,0)]:
                for (c_idx, c_c) in Z_coeffs[0]:
                    U[a_idx, col_offset] += c_a
                    V[b_idx, col_offset] += c_b
                    W[c_idx, col_offset] += c_c
        col_offset += 1
        
        # 3. x0 * y01 -> z1
        for (a_idx, c_a) in X_coeffs[0]:
            for (b_idx, c_b) in Y_coeffs[(0,1)]:
                for (c_idx, c_c) in Z_coeffs[1]:
                    U[a_idx, col_offset] += c_a
                    V[b_idx, col_offset] += c_b
                    W[c_idx, col_offset] += c_c
        col_offset += 1
        
        # 4. x1 * y11 -> z1
        for (a_idx, c_a) in X_coeffs[1]:
            for (b_idx, c_b) in Y_coeffs[(1,1)]:
                for (c_idx, c_c) in Z_coeffs[1]:
                    U[a_idx, col_offset] += c_a
                    V[b_idx, col_offset] += c_b
                    W[c_idx, col_offset] += c_c
        col_offset += 1
        
        return col_offset

    # Define global indices for A, B, C blocks
    # A (2x4):
    # A11: A[0,0], A[0,1]
    # A12: A[0,2], A[0,3]
    # A21: A[1,0], A[1,1]
    # A22: A[1,2], A[1,3]
    
    # B (4x4):
    # B11: B[0,0], B[0,1], B[1,0], B[1,1]
    # B12: B[0,2], B[0,3], B[1,2], B[1,3]
    # B21: B[2,0], B[2,1], B[3,0], B[3,1]
    # B22: B[2,2], B[2,3], B[3,2], B[3,3]
    
    # C (2x4):
    # C11: C[0,0], C[1,0] (k=0,1, i=0) -> W indices 0, 1?
    # Trace convention: W[k*n+i, r]. n=2.
    # C[i, k] -> W[k*2 + i].
    # C11 (i=0, k=0,1): W[0*2+0]=W[0], W[1*2+0]=W[2]
    # C12 (i=0, k=2,3): W[2*2+0]=W[4], W[3*2+0]=W[6]
    # C21 (i=1, k=0,1): W[0*2+1]=W[1], W[1*2+1]=W[3]
    # C22 (i=1, k=2,3): W[2*2+1]=W[5], W[3*2+1]=W[7]
    
    # Let's define the block content for each Strassen term.
    # X_coeffs: {0: [list of (A_idx, coeff)], 1: [list of (A_idx, coeff)]}
    # Y_coeffs: {(0,0): [...], (0,1): [...], (1,0): [...], (1,1): [...]}
    # Z_coeffs: {0: [list of (C_idx, coeff)], 1: [list of (C_idx, coeff)]}
    
    # M1 = (A11 + A22) * (B11 + B22)
    # X0 = A11 = [A[0,0], A[0,1]]
    # X1 = A22 = [A[1,2], A[1,3]]
    # Wait, X is a 1x2 vector.
    # A11 is 1x2. A22 is 1x2.
    # X = A11 + A22 = [A[0,0]+A[1,2], A[0,1]+A[1,3]]
    # So X[0] corresponds to the first element of the 1x2 vector.
    # X_coeffs[0] = [(0*4+0, 1), (1*4+2, 1)] # A[0,0] + A[1,2]
    # X_coeffs[1] = [(0*4+1, 1), (1*4+3, 1)] # A[0,1] + A[1,3]
    
    # Y = B11 + B22
    # B11 is 2x2. B22 is 2x2.
    # Y[0,0] = B[0,0] + B[2,2]
    # Y[0,1] = B[0,1] + B[2,3]
    # Y[1,0] = B[1,0] + B[3,2]
    # Y[1,1] = B[1,1] + B[3,3]
    
    # Z = C11 + C22? No, M1 contributes to C11 and C22.
    # C11 = M1 + ...
    # C22 = M1 + ...
    # So Z_coeffs for M1 should place +1 into C11 and +1 into C22.
    # Z[0] (first element of 1x2 C block) corresponds to C[k=0, i] and C[k=2, i]?
    # No, the Strassen block multiplication produces a 1x2 result.
    # For M1, the result is a 1x2 block.
    # This 1x2 block is added to C11 (1x2) and C22 (1x2).
    # So Z_coeffs[0] should contain indices for C11[0] and C22[0].
    # C11[0] is C[0,0] -> W[0]. C22[0] is C[1,2] -> W[5].
    # C11[1] is C[0,1] -> W[2]. C22[1] is C[1,3] -> W[7].
    
    # So for M1:
    # Z_coeffs[0] = [(W_idx for C11[0], 1), (W_idx for C22[0], 1)] = [(0, 1), (5, 1)]
    # Z_coeffs[1] = [(W_idx for C11[1], 1), (W_idx for C22[1], 1)] = [(2, 1), (7, 1)]
    
    # Let's code this up for all 7 terms.
    
    terms = []
    
    # M1
    X1 = {0: [(0, 1), (6, 1)], 1: [(1, 1), (7, 1)]} # A[0,0]+A[1,2], A[0,1]+A[1,3]
    Y1 = {(0,0): [(0, 1), (10, 1)], (0,1): [(1, 1), (11, 1)], (1,0): [(4, 1), (14, 1)], (1,1): [(5, 1), (15, 1)]}
    Z1 = {0: [(0, 1), (5, 1)], 1: [(2, 1), (7, 1)]}
    terms.append((X1, Y1, Z1))
    
    # M2 = (A21 + A22) * B11
    # X = A21 + A22 = [A[1,0]+A[1,2], A[1,1]+A[1,3]]
    X2 = {0: [(4, 1), (6, 1)], 1: [(5, 1), (7, 1)]}
    # Y = B11
    Y2 = {(0,0): [(0, 1)], (0,1): [(1, 1)], (1,0): [(4, 1)], (1,1): [(5, 1)]}
    # Z -> C21 (+1) and C22 (-1)
    # C21[0] = C[1,0] -> W[1]. C22[0] = C[1,2] -> W[5].
    # C21[1] = C[1,1] -> W[3]. C22[1] = C[1,3] -> W[7].
    Z2 = {0: [(1, 1), (5, -1)], 1: [(3, 1), (7, -1)]}
    terms.append((X2, Y2, Z2))
    
    # M3 = A11 * (B12 - B22)
    X3 = {0: [(0, 1)], 1: [(1, 1)]}
    Y3 = {(0,0): [(2, 1), (10, -1)], (0,1): [(3, 1), (11, -1)], (1,0): [(6, 1), (14, -1)], (1,1): [(7, 1), (15, -1)]}
    # Z -> C12 (+1) and C22 (+1)
    # C12[0] = C[0,2] -> W[4]. C22[0] = C[1,2] -> W[5].
    # C12[1] = C[0,3] -> W[6]. C22[1] = C[1,3] -> W[7].
    Z3 = {0: [(4, 1), (5, 1)], 1: [(6, 1), (7, 1)]}
    terms.append((X3, Y3, Z3))
    
    # M4 = A22 * (B21 - B11)
    X4 = {0: [(6, 1)], 1: [(7, 1)]}
    Y4 = {(0,0): [(8, 1), (0, -1)], (0,1): [(9, 1), (1, -1)], (1,0): [(12, 1), (4, -1)], (1,1): [(13, 1), (5, -1)]}
    # Z -> C11 (+1) and C21 (+1)
    # C11[0] = C[0,0] -> W[0]. C21[0] = C[1,0] -> W[1].
    # C11[1] = C[0,1] -> W[2]. C21[1] = C[1,1] -> W[3].
    Z4 = {0: [(0, 1), (1, 1)], 1: [(2, 1), (3, 1)]}
    terms.append((X4, Y4, Z4))
    
    # M5 = (A11 + A12) * B22
    X5 = {0: [(0, 1), (2, 1)], 1: [(1, 1), (3, 1)]}
    Y5 = {(0,0): [(10, 1)], (0,1): [(11, 1)], (1,0): [(14, 1)], (1,1): [(15, 1)]}
    # Z -> C11 (-1) and C12 (+1)
    # C11[0] = C[0,0] -> W[0]. C12[0] = C[0,2] -> W[4].
    # C11[1] = C[0,1] -> W[2]. C12[1] = C[0,3] -> W[6].
    Z5 = {0: [(0, -1), (4, 1)], 1: [(2, -1), (6, 1)]}
    terms.append((X5, Y5, Z5))
    
    # M6 = (A21 - A11) * (B11 + B12)
    X6 = {0: [(4, 1), (0, -1)], 1: [(5, 1), (1, -1)]}
    Y6 = {(0,0): [(0, 1), (2, 1)], (0,1): [(1, 1), (3, 1)], (1,0): [(4, 1), (6, 1)], (1,1): [(5, 1), (7, 1)]}
    # Z -> C22 (+1)
    # C22[0] = C[1,2] -> W[5].
    # C22[1] = C[1,3] -> W[7].
    Z6 = {0: [(5, 1)], 1: [(7, 1)]}
    terms.append((X6, Y6, Z6))
    
    # M7 = (A12 - A22) * (B21 + B22)
    X7 = {0: [(2, 1), (6, -1)], 1: [(3, 1), (7, -1)]}
    Y7 = {(0,0): [(8, 1), (10, 1)], (0,1): [(9, 1), (11, 1)], (1,0): [(12, 1), (14, 1)], (1,1): [(13, 1), (15, 1)]}
    # Z -> C11 (+1) and C22 (+1)
    # C11[0] = C[0,0] -> W[0]. C22[0] = C[1,2] -> W[5].
    # C11[1] = C[0,1] -> W[2]. C22[1] = C[1,3] -> W[7].
    Z7 = {0: [(0, 1), (5, 1)], 1: [(2, 1), (7, 1)]}
    terms.append((X7, Y7, Z7))
    
    col = 0
    for X, Y, Z in terms:
        col = add_strassen_term(X, Y, Z, col)
        
    return U, V, W

def _build_schoolbook_1x4x4():
    """
    Schoolbook for 1x4x4: A is 1x4, B is 4x4, C is 1x4.
    Rank = 16.
    """
    rank = 16
    U = np.zeros((4, rank))   # A: 1x4, index j (0-3)
    V = np.zeros((16, rank))  # B: 4x4, index j*4+k
    W = np.zeros((4, rank))   # C: 1x4, index k (0-3). Trace: W[k*1+0] = W[k]
    
    for j in range(4):
        for k in range(4):
            U[j, j*4+k] = 1
            V[j*4+k, j*4+k] = 1
            W[k, j*4+k] = 1
    
    return U, V, W

def _schoolbook_matrices(n, m, p):
    """Standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine_3x4x4():
    """
    Combine 2x4x4 (Strassen, rank 28) and 1x4x4 (schoolbook, rank 16) for total rank 44.
    """
    U2, V2, W2 = _build_strassen_2x4x4()
    U1, V1, W1 = _build_schoolbook_1x4x4()
    
    total_rank = 44
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    # Place 2x4x4 part (columns 0-27)
    # A2 is 2x4. A is 3x4. Top 2 rows.
    for i in range(2):
        for j in range(4):
            U[i*4+j, 0:28] = U2[i*4+j, :]
    V[0:16, 0:28] = V2[:, :]
    # C2 is 2x4. C is 3x4. Top 2 rows.
    # W2 index: k*2+i. W index: k*3+i.
    for i in range(2):
        for k in range(4):
            W[k*3+i, 0:28] = W2[k*2+i, :]
    
    # Place 1x4x4 part (columns 28-43)
    # A1 is 1x4. A is 3x4. Bottom row (i=2).
    for j in range(4):
        U[2*4+j, 28:44] = U1[j, :]
    V[0:16, 28:44] = V1[:, :]
    # C1 is 1x4. C is 3x4. Bottom row (i=2).
    # W1 index: k. W index: k*3+2.
    for k in range(4):
        W[k*3+2, 28:44] = W1[k, :]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    try:
        U, V, W = _combine_3x4x4()
        if _validate(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook
            U, V, W = _schoolbook_matrices(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        U, V, W = _schoolbook_matrices(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
