import time
import numpy as np

DIMENSIONS = [4, 4, 7]
SEED = 42

def _build_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = -1; W[2, 0] = -1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 0; V[2, 1] = 0; V[3, 1] = 0 # B11 is (0,0) and (0,1)? No.
    # B11 is top-left 1x1? No, B is 2x2.
    # B11 is element (0,0). B12 is (0,1). B21 is (1,0). B22 is (1,1).
    # In Strassen, we treat A and B as 2x2 blocks of 1x1 matrices.
    # So A11 is A[0,0], A12 is A[0,1], etc.
    # V[j*2+k, r].
    # For M2 = (A21+A22)B11.
    # A21 is A[1,0] (index 2). A22 is A[1,1] (index 3).
    # B11 is B[0,0] (index 0).
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21 = M2 + ...
    
    # M3 = A11(B12-B22)
    # A11 is A[0,0] (index 0).
    # B12 is B[0,1] (index 1). B22 is B[1,1] (index 3).
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12 = M3 + ...
    
    # M4 = A22(B21-B11)
    # A22 is A[1,1] (index 3).
    # B21 is B[1,0] (index 2). B11 is B[0,0] (index 0).
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1 # C21 = ... + M4
    
    # M5 = (A11+A12)B22
    # A11 (0), A12 (1). B22 (3).
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[3, 4] = 1 # C22 = M5 + ...
    
    # M6 = (A21-A11)(B11+B12)
    # A21 (2), A11 (0). B11 (0), B12 (1).
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1 # C11 = ... + M6
    
    # M7 = (A12-A22)(B21+B22)
    # A12 (1), A22 (3). B21 (2), B22 (3).
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1 # C22 = ... + M7
    
    return U, V, W

def _build_strassen_recursive(n, m, p, base_U, base_V, base_W):
    """
    Builds decomposition for n x m x p using Strassen-like recursion.
    Assumes n=m=p=2^k for simplicity, or handles rectangular by padding?
    Here we specifically need 4x4x4.
    """
    if n == 2 and m == 2 and p == 2:
        return base_U, base_V, base_W
    
    # Recurse for 2x2 blocks
    # This function is complex for general rectangular.
    # For 4x4x4, we can just manually apply the 2x2 Strassen to the 4x4 blocks.
    pass

def _build_4x4x4_strassen():
    """
    Constructs rank-49 decomposition for 4x4x4.
    Uses the 2x2x2 Strassen decomposition recursively.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    # We need to lift this to 4x4x4.
    # A 4x4 matrix is 4 2x2 blocks.
    # The recursion doubles the size.
    # Rank becomes 7 * 7 = 49.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # The indices for 4x4 are i in 0..3, j in 0..3.
    # U[i*4+j, r]
    # The base U2[i*2+j, r2]
    
    # We iterate over the 7 terms of the outer Strassen and 7 terms of the inner.
    # This is effectively the Kronecker product of the decompositions.
    
    for r_outer in range(7):
        for r_inner in range(7):
            r = r_outer * 7 + r_inner
            
            # U contribution
            # U_outer[i_outer, r_outer] * U_inner[i_inner, r_inner]
            # i = i_outer * 2 + i_inner
            for i_outer in range(2):
                u_out = U2[i_outer * 2 + 0, r_outer] # Wait, U2 indices are i*2+j.
                # U2 is (4,7). Index i*2+j.
                # We need to map the block structure.
                # This manual mapping is error-prone.
                # Let's use a known property: The decomposition of M_{2n, 2n, 2n} 
                # can be formed from M_{n,n,n} by Kronecker products of the factors.
                
                pass
    
    # Simpler: Just use schoolbook for 4x4x4 if recursion is too complex?
    # Schoolbook rank is 64.
    # 64 + 48 (for 3 cols) = 112. No improvement.
    
    # Let's try a different structural approach:
    # Use the 7-rank Strassen for 2x2x2.
    # For 4x4x4, we can use the fact that it's 4 independent 2x2x2 multiplications? No.
    
    # Let's return a schoolbook decomposition for now to ensure correctness,
    # but try to implement the 4x4x4 Strassen properly.
    
    # Correct recursive construction:
    # M_{2n, 2n, 2n} is built from 7 copies of M_{n,n,n}.
    # U_{2n} = [U_n \otimes I, ...]
    
    # Let's just implement the 4x4x4 Strassen explicitly.
    # It's a standard exercise.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Base case U2, V2, W2 are 4x7.
    # We need to expand to 16x49.
    
    # The indices for 4x4 are (i, j) where i,j in 0..3.
    # i = i1*2 + i0, j = j1*2 + j0.
    
    # The decomposition is a sum of 49 terms.
    # Term (r1, r2) where r1, r2 in 0..6.
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # U factor
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            u_val = U2[i1 * 2 + j1, r1] * U2[i0 * 2 + j0, r2]
                            if u_val != 0:
                                U[i * 4 + j, r] += u_val
            
            # V factor
            # V indices (j, k). j = j1*2+j0, k = k1*2+k0.
            for j1 in range(2):
                for j0 in range(2):
                    for k1 in range(2):
                        for k0 in range(2):
                            j = j1 * 2 + j0
                            k = k1 * 2 + k0
                            v_val = V2[j1 * 2 + k1, r1] * V2[j0 * 2 + k0, r2]
                            if v_val != 0:
                                V[j * 4 + k, r] += v_val
            
            # W factor
            # W indices (k, i). k = k1*2+k0, i = i1*2+i0.
            for k1 in range(2):
                for k0 in range(2):
                    for i1 in range(2):
                        for i0 in range(2):
                            k = k1 * 2 + k0
                            i = i1 * 2 + i0
                            w_val = W2[k1 * 2 + i1, r1] * W2[k0 * 2 + i0, r2]
                            if w_val != 0:
                                W[k * 4 + i, r] += w_val
    
    return U, V, W

def _build_schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape != (n * m, W_mat.shape[1]): return False
    if V_mat.shape != (m * p, W_mat.shape[1]): return False
    if W_mat.shape != (p * n, U_mat.shape[1]): return False
    if U_mat.shape[1] != V_mat.shape[1] or V_mat.shape[1] != W_mat.shape[1]: return False
    
    rank = U_mat.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build 4x4x4 Strassen (Rank 49)
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Build 4x4x3 Schoolbook (Rank 48)
    U3, V3, W3 = _build_schoolbook(4, 4, 3)
    
    # Concatenate
    U = np.hstack([U4, U3])
    V = np.hstack([V4, V3])
    # W for 4x4x4 is (16, 49). W for 4x4x3 is (12, 48).
    # We need to align the k indices.
    # W4[k*4+i, r] for k in 0..3.
    # W3[k*4+i, r] for k in 0..2 (corresponding to cols 4,5,6).
    # We need to shift W3's k index by 4.
    W3_shifted = np.zeros((28, 48)) # p*n = 7*4 = 28
    for k in range(3):
        for i in range(4):
            for r in range(48):
                W3_shifted[(k+4)*4+i, r] = W3[k*4+i, r]
                
    W = np.hstack([W4, W3_shifted])
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U_s, V_s, W_s = _build_schoolbook(n, m, p)
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
