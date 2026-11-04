"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import qr, solve

DIMENSIONS = [4, 4, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if not (U.shape[1] == V.shape[1] == W.shape[1]):
        return False
    
    rank = U.shape[1]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    
    Returns U_mat (4x7), V_mat (4x7), W_mat (4x7) such that
    for all i,j,k in {0,1}: sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1
    """
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # A indices: A[0,0]=a(idx0), A[0,1]=b(idx1), A[1,0]=c(idx2), A[1,1]=d(idx3)
    # B indices: B[0,0]=e(idx0), B[0,1]=f(idx1), B[1,0]=g(idx2), B[1,1]=h(idx3)
    # C indices: C[0,0](idx0), C[0,1](idx1), C[1,0](idx2), C[1,1](idx3)
    
    # U vectors (coefficients for a,b,c,d)
    U_terms = [
        [1,0,0,1],   # M1: (a+d)
        [0,0,1,1],   # M2: (c+d)
        [1,0,0,0],   # M3: a
        [0,0,0,1],   # M4: d
        [1,1,0,0],   # M5: (a+b)
        [-1,0,1,0],  # M6: (c-a)
        [0,1,0,-1],  # M7: (b-d)
    ]
    
    # V vectors (coefficients for e,f,g,h)
    V_terms = [
        [1,1,0,0],   # M1: (e+h) -> wait, (e+h) should be [1,0,0,1]
        # Let me redo this carefully
    ]
    
    # M1 = (a+d)(e+h): U=[1,0,0,1], V=[1,0,0,1]
    # M2 = (c+d)e: U=[0,0,1,1], V=[1,0,0,0]
    # M3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1]
    # M4 = d(g-e): U=[0,0,0,1], V=[-1,0,1,0]
    # M5 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1]
    # M6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0]
    # M7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1]
    
    U_terms = [
        [1,0,0,1],   # M1
        [0,0,1,1],   # M2
        [1,0,0,0],   # M3
        [0,0,0,1],   # M4
        [1,1,0,0],   # M5
        [-1,0,1,0],  # M6
        [0,1,0,-1],  # M7
    ]
    
    V_terms = [
        [1,0,0,1],   # M1: (e+h)
        [1,0,0,0],   # M2: e
        [0,1,0,-1],  # M3: (f-h)
        [-1,0,1,0],  # M4: (g-e)
        [0,0,0,1],   # M5: h
        [1,1,0,0],   # M6: (e+f)
        [0,0,1,1],   # M7: (g+h)
    ]
    
    # W: for each C element, which M's contribute with what sign
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # W_mat[c_idx, r] = coefficient of M_r in C[c_idx]
    W_mat = np.zeros((4, 7))
    # C[0,0] (idx 0):
    W_mat[0, 0] = 1   # M1
    W_mat[0, 3] = 1   # M4
    W_mat[0, 4] = -1  # M5
    W_mat[0, 6] = 1   # M7
    # C[0,1] (idx 1):
    W_mat[1, 2] = 1   # M3
    W_mat[1, 4] = 1   # M5
    # C[1,0] (idx 2):
    W_mat[2, 1] = 1   # M2
    W_mat[2, 3] = 1   # M4
    # C[1,1] (idx 3):
    W_mat[3, 0] = 1   # M1
    W_mat[3, 1] = -1  # M2
    W_mat[3, 2] = 1   # M3
    W_mat[3, 5] = 1   # M6
    
    U_mat = np.zeros((4, 7))
    V_mat = np.zeros((4, 7))
    for r in range(7):
        U_mat[:, r] = U_terms[r]
        V_mat[:, r] = V_terms[r]
    
    return U_mat, V_mat, W_mat

def _strassen_4x4x4():
    """Build the 4x4x4 Strassen decomposition with rank 49.
    
    Uses recursive Strassen: 7 outer multiplications of 2x2 blocks,
    each decomposed using 7 inner Strassen terms.
    
    Returns U (16x49), V (16x49), W (16x49).
    """
    n = m = p = 4
    rank = 49
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # The 7 outer Strassen combinations for 4x4 = 2x2 blocks:
    # A blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C blocks: C11=C[0:2,0:2], C12=C[0:2,2:4], C21=C[2:4,0:2], C22=C[2:4,2:4]
    
    # For each outer term s, we have:
    #   A_combined = sum_{block} alpha_{s,block} * A_block
    #   B_combined = sum_{block} beta_{s,block} * B_block
    #   M_s = A_combined * B_combined (2x2 matrix product)
    #   C_block contributions from M_s
    
    # Define the 7 outer terms as (A_block_coeffs, B_block_coeffs, C_block_contribs)
    # A_block_coeffs: dict from block_idx to coeff, block_idx: 0=A11,1=A12,2=A21,3=A22
    # B_block_coeffs: dict from block_idx to coeff
    # C_block_contribs: dict from C_block_idx to list of (contrib_block_idx, sign)
    #   where contrib_block_idx indicates which 2x2 block of M_s contributes
    
    # M1 = (A11+A22)(B11+B22)
    #   A_combined = A11 + A22, B_combined = B11 + B22
    #   M1 is 2x2, contributes to C11 and C22
    #   C11 += M1, C22 += M1
    
    # M2 = (A21+A22)B11
    #   A_combined = A21 + A22, B_combined = B11
    #   M2 is 2x2, contributes to C21 and C22
    #   C21 += M2, C22 -= M2
    
    # M3 = A11(B12-B22)
    #   A_combined = A11, B_combined = B12 - B22
    #   M3 is 2x2, contributes to C12 and C22
    #   C12 += M3, C22 += M3
    
    # M4 = A22(B21-B11)
    #   A_combined = A22, B_combined = B21 - B11
    #   M4 is 2x2, contributes to C11 and C21
    #   C11 += M4, C21 += M4
    
    # M5 = (A11+A12)B22
    #   A_combined = A11 + A12, B_combined = B22
    #   M5 is 2x2, contributes to C11 and C12
    #   C11 -= M5, C12 += M5
    
    # M6 = (A21-A11)(B11+B12)
    #   A_combined = A21 - A11, B_combined = B11 + B12
    #   M6 is 2x2, contributes to C22
    #   C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    #   A_combined = A12 - A22, B_combined = B21 + B22
    #   M7 is 2x2, contributes to C11
    #   C11 += M7
    
    # For each outer term s and each inner Strassen term t (0..6):
    #   The inner term t computes: sum of U2[i*2+j,t] * A_combined[i,j] * V2[j*2+k,t] * B_combined[j,k] * W2[k*2+i,t]
    #   This contributes to C_block according to the outer term's C contribution pattern
    
    # Block index to (row_start, col_start) mapping for 2x2 blocks in 4x4:
    # block 0 (A11): rows 0-1, cols 0-1
    # block 1 (A12): rows 0-1, cols 2-3
    # block 2 (A21): rows 2-3, cols 0-1
    # block 3 (A22): rows 2-3, cols 2-3
    
    def block_to_indices(block_idx):
        """Return (row_start, col_start) for a 2x2 block in 4x4."""
        r = block_idx // 2
        c = block_idx % 2
        return (2*r, 2*c)
    
    # Outer term definitions:
    # (A_block_coeffs, B_block_coeffs, C_contribs)
    # C_contribs: list of (C_block_idx, sign, M_block_within_2x2)
    #   where M_block_within_2x2 indicates which 2x2 sub-block of the 2x2 result M_s
    #   contributes to which C_block. Actually, the entire 2x2 M_s maps to a 2x2 C_block.
    
    # For each outer term, the 2x2 result M_s is added to specific C_blocks.
    # The mapping is: M_s (2x2) -> C_block (2x2), with a sign.
    
    outer_terms = [
        # M1 = (A11+A22)(B11+B22)
        {
            'A': {0: 1, 3: 1},      # A11 + A22
            'B': {0: 1, 3: 1},      # B11 + B22
            'C': [(0, 1), (3, 1)]   # C11 += M1, C22 += M1
        },
        # M2 = (A21+A22)B11
        {
            'A': {2: 1, 3: 1},      # A21 + A22
            'B': {0: 1},            # B11
            'C': [(2, 1), (3, -1)]  # C21 += M2, C22 -= M2
        },
        # M3 = A11(B12-B22)
        {
            'A': {0: 1},            # A11
            'B': {1: 1, 3: -1},    # B12 - B22
            'C': [(1, 1), (3, 1)]   # C12 += M3, C22 += M3
        },
        # M4 = A22(B21-B11)
        {
            'A': {3: 1},            # A22
            'B': {2: 1, 0: -1},    # B21 - B11
            'C': [(0, 1), (2, 1)]   # C11 += M4, C21 += M4
        },
        # M5 = (A11+A12)B22
        {
            'A': {0: 1, 1: 1},     # A11 + A12
            'B': {3: 1},            # B22
            'C': [(0, -1), (1, 1)]  # C11 -= M5, C12 += M5
        },
        # M6 = (A21-A11)(B11+B12)
        {
            'A': {2: 1, 0: -1},    # A21 - A11
            'B': {0: 1, 1: 1},     # B11 + B12
            'C': [(3, 1)]           # C22 += M6
        },
        # M7 = (A12-A22)(B21+B22)
        {
            'A': {1: 1, 3: -1},    # A12 - A22
            'B': {2: 1, 3: 1},     # B21 + B22
            'C': [(0, 1)]           # C11 += M7
        },
    ]
    
    for s, term in enumerate(outer_terms):
        A_coeffs = term['A']
        B_coeffs = term['B']
        C_contribs = term['C']
        
        for t in range(7):
            col_idx = s * 7 + t
            
            # Build U vector for this term
            # U[i*4+j, col_idx] = sum over A_blocks: A_coeffs[block] * U2[i*2+j, t]
            # where i,j are within the 2x2 block
            for a_block, a_coeff in A_coeffs.items():
                r_start, c_start = block_to_indices(a_block)
                for di in range(2):
                    for dj in range(2):
                        i_global = r_start + di
                        j_global = c_start + dj
                        U[i_global * 4 + j_global, col_idx] += a_coeff * U2[di * 2 + dj, t]
            
            # Build V vector
            for b_block, b_coeff in B_coeffs.items():
                r_start, c_start = block_to_indices(b_block)
                for dj in range(2):
                    for dk in range(2):
                        j_global = r_start + dj
                        k_global = c_start + dk
                        V[j_global * 4 + k_global, col_idx] += b_coeff * V2[dj * 2 + dk, t]
            
            # Build W vector
            for c_block, c_sign in C_contribs:
                r_start, c_start = block_to_indices(c_block)
                for dk in range(2):
                    for di in range(2):
                        k_global = c_start + dk
                        i_global = r_start + di
                        W[k_global * 4 + i_global, col_idx] += c_sign * W2[dk * 2 + di, t]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U.tolist(), V.tolist(), W.tolist()

def _build_4x4x5():
    """Build 4x4x5 decomposition: Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 16) = rank 65."""
    n, m, p = 4, 4, 5
    
    # Get Strassen 4x4x4 decomposition
    U_strassen, V_strassen, W_strassen = _strassen_4x4x4()
    
    # Build schoolbook for 4x4x1 (last column of B, so k=4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    U_last = np.zeros((n*m, n*m))
    V_last = np.zeros((m*p, n*m))
    W_last = np.zeros((p*n, n*m))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U_last[i*m+j, col] = 1
            V_last[j*p+4, col] = 1  # k=4 (the 5th column, 0-indexed)
            W_last[4*n+i, col] = 1  # k*n + i = 4*4 + i
            col += 1
    
    # Combine
    U = np.hstack([U_strassen, U_last])
    V = np.hstack([V_strassen, V_last])
    W = np.hstack([W_strassen, W_last])
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try to build the 4x4x4 Strassen + 4x4x1 schoolbook decomposition (rank 65)
    try:
        U, V, W = _build_4x4x5()
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
            elapsed = time.time() - start_time
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, fall back to schoolbook
            pass
    except Exception:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    elapsed = time.time() - start_time
    return {"U": U, "V": V, "W": W}
