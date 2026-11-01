import numpy as np
import time

def _strassen_2x2_U(A_idx, is_left):
    """
    Returns the linear coefficients for U for a 2x2 Strassen multiplication.
    A_idx: tuple (i, j) for the element in the 2x2 matrix A.
    is_left: True if this is A_L (cols 0,1 of the 4-col A), False if A_R (cols 2,3).
    Actually, we just need the coefficients for the 2x2 submatrix.
    """
    # Strassen P1..P7
    # P1: (A11+A22)(B11+B22)
    # P2: (A21+A22)B11
    # P3: A11(B12-B22)
    # P4: A22(B21-B11)
    # P5: (A11+A12)B22
    # P6: (A21-A11)(B11+B12)
    # P7: (A12-A22)(B21+B22)
    
    # We return a list of 7 coefficients for the given A element.
    # The A element is at (r, c) in the 2x2 submatrix.
    r, c = A_idx
    coeffs = np.zeros(7)
    
    if r == 0 and c == 0: # A11
        coeffs[0] = 1 # P1
        coeffs[2] = 1 # P3
        coeffs[4] = 1 # P5
        coeffs[5] = -1 # P6
    elif r == 1 and c == 1: # A22
        coeffs[0] = 1 # P1
        coeffs[1] = 1 # P2
        coeffs[3] = 1 # P4
        coeffs[6] = -1 # P7
    elif r == 1 and c == 0: # A21
        coeffs[1] = 1 # P2
        coeffs[5] = 1 # P6
    elif r == 0 and c == 1: # A12
        coeffs[4] = 1 # P5
        coeffs[6] = 1 # P7
        
    return coeffs

def _strassen_2x2_V(B_idx, is_top):
    """
    Returns the linear coefficients for V for a 2x2 Strassen multiplication.
    B_idx: tuple (r, c) for the element in the 2x2 matrix B.
    is_top: True if this is B_T (rows 0,1 of the 4-row B), False if B_B (rows 2,3).
    """
    r, c = B_idx
    coeffs = np.zeros(7)
    
    if r == 0 and c == 0: # B11
        coeffs[0] = 1 # P1
        coeffs[1] = 1 # P2
        coeffs[3] = -1 # P4
        coeffs[5] = 1 # P6
    elif r == 1 and c == 1: # B22
        coeffs[0] = 1 # P1
        coeffs[2] = -1 # P3
        coeffs[4] = 1 # P5
        coeffs[6] = -1 # P7
    elif r == 0 and c == 1: # B12
        coeffs[2] = 1 # P3
        coeffs[5] = 1 # P6
    elif r == 1 and c == 0: # B21
        coeffs[3] = 1 # P4
        coeffs[6] = 1 # P7
        
    return coeffs

def _strassen_2x2_W(C_idx):
    """
    Returns the linear coefficients for W for a 2x2 Strassen multiplication.
    C_idx: tuple (r, c) for the element in the 2x2 matrix C.
    """
    r, c = C_idx
    coeffs = np.zeros(7)
    
    if r == 0 and c == 0: # C11
        coeffs[0] = 1 # P1
        coeffs[3] = 1 # P4
        coeffs[4] = -1 # P5
        coeffs[6] = 1 # P7
    elif r == 0 and c == 1: # C12
        coeffs[2] = 1 # P3
        coeffs[4] = 1 # P5
    elif r == 1 and c == 0: # C21
        coeffs[1] = 1 # P2
        coeffs[3] = 1 # P4
    elif r == 1 and c == 1: # C22
        coeffs[0] = 1 # P1
        coeffs[1] = -1 # P2
        coeffs[2] = 1 # P3
        coeffs[5] = 1 # P6
        
    return coeffs

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=10, tol=1e-9):
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('lk,lkc->kc', B, V_reshaped)
        C_computed = np.einsum('kic,ic,kc->ik', W_reshaped, factor_A, factor_B)
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 8
    # We are constructing a rank 56 decomposition.
    # 4 blocks of 2x2 output, each requiring rank 14 (two 2x2 Strassen multiplications).
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Loop over the 4 output blocks (t = 0..3)
    # Each block t corresponds to columns 2t and 2t+1 of C (and B).
    for t in range(4):
        col_offset = t * 14
        
        # For this block, we have two Strassen multiplications:
        # 1. A_L * B_T
        # 2. A_R * B_B
        # A_L is A[:, 0:2], A_R is A[:, 2:4]
        # B_T is B[0:2, 2t:2t+2], B_B is B[2:4, 2t:2t+2]
        
        # Sub-block 1: A_L * B_T
        # U indices for A_L: i in 0..1, j in 0..1
        # V indices for B_T: l in 0..1, k in 2t..2t+1
        # W indices for C_t: i in 0..1, k in 2t..2t+1
        
        for i in range(2):
            for j_sub in range(2):
                j = j_sub # 0, 1
                u_coeffs = _strassen_2x2_U((i, j_sub), True)
                U[i * m + j, col_offset:col_offset+7] += u_coeffs
        
        for l_sub in range(2):
            l = l_sub # 0, 1
            for k_sub in range(2):
                k = 2 * t + k_sub
                v_coeffs = _strassen_2x2_V((l_sub, k_sub), True)
                V[l * p + k, col_offset:col_offset+7] += v_coeffs
        
        for i in range(2):
            for k_sub in range(2):
                k = 2 * t + k_sub
                w_coeffs = _strassen_2x2_W((i, k_sub))
                W[k * n + i, col_offset:col_offset+7] += w_coeffs
        
        # Sub-block 2: A_R * B_B
        # U indices for A_R: i in 0..1, j in 2..3
        # V indices for B_B: l in 2..3, k in 2t..2t+1
        # W indices for C_t: i in 0..1, k in 2t..2t+1
        
        for i in range(2):
            for j_sub in range(2):
                j = 2 + j_sub # 2, 3
                u_coeffs = _strassen_2x2_U((i, j_sub), False)
                U[i * m + j, col_offset+7:col_offset+14] += u_coeffs
        
        for l_sub in range(2):
            l = 2 + l_sub # 2, 3
            for k_sub in range(2):
                k = 2 * t + k_sub
                v_coeffs = _strassen_2x2_V((l_sub, k_sub), False)
                V[l * p + k, col_offset+7:col_offset+14] += v_coeffs
        
        for i in range(2):
            for k_sub in range(2):
                k = 2 * t + k_sub
                w_coeffs = _strassen_2x2_W((i, k_sub))
                W[k * n + i, col_offset+7:col_offset+14] += w_coeffs

    # Verify
    if not _verify_fast(U, V, W, n, m, p, trials=100):
        # Fallback to schoolbook if verification fails (should not happen)
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
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
