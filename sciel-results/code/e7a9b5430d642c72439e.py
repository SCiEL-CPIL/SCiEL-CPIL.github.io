# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2_decomposition():
    """
    Returns the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns:
        U: 4x7 matrix (flattened A)
        V: 4x7 matrix (flattened B)
        W: 4x7 matrix (flattened C)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M2 = (c+d)(e+f)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    # M3 = (a+b)(e+g)
    U[0, 2] = 1; U[1, 2] = 1
    V[0, 2] = 1; V[2, 2] = 1
    # M4 = (c+d)h
    U[2, 3] = 1; U[3, 3] = 1
    V[3, 3] = 1
    # M5 = (a+b)f
    U[0, 4] = 1; U[1, 4] = 1
    V[1, 4] = 1
    # M6 = (d-c)(g+h)
    U[2, 5] = -1; U[3, 5] = 1
    V[2, 5] = 1; V[3, 5] = 1
    # M7 = (a-b)(e+f)
    U[0, 6] = 1; U[1, 6] = -1
    V[0, 6] = 1; V[1, 6] = 1
    
    # W (Output C)
    # Indexing: W[k*n + i, r] for n=2.
    # c11 (i=0, k=0) -> idx 0
    # c21 (i=1, k=0) -> idx 1
    # c12 (i=0, k=1) -> idx 2
    # c22 (i=1, k=1) -> idx 3
    
    # c11 = m1 + m4 - m5 + m7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # c21 = m2 - m4 + m6 - m7
    # Previously this was swapped with c12 in the buggy version
    W[1, 1] = 1; W[1, 3] = -1; W[1, 5] = 1; W[1, 6] = -1
    
    # c12 = m3 - m1 + m5 + m6
    # Previously this was swapped with c21 in the buggy version
    W[2, 0] = -1; W[2, 2] = 1; W[2, 4] = 1; W[2, 5] = 1
    
    # c22 = m1 - m2 + m3 + m4
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 3] = 1
    
    return U, V, W

def _verify_strassen():
    U, V, W = _strassen_2x2_decomposition()
    # Check if U @ diag(M) @ V^T ... no, the tensor is T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Here i is A index, j is B index, k is C index.
    # The decomposition is C = sum_r U[:,r] * M_r * V[:,r]^T ? No.
    # The standard tensor decomposition is T = sum_r u_r v_r w_r^T.
    # For matrix multiplication, T_{i,j,k} = delta_{i,k}? No.
    # The condition is that the bilinear map defined by U,V,W computes the matrix product.
    # i.e., C[i,k] = sum_r W[k*n + i, r] * (sum_{i,j} U[i*m+j, r] A[i,j]) * (sum_{j,k} V[j*p+k, r] B[j,k])
    # Note: The indices in U, V, W in the 2x2 case are local.
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    
    # We verify by constructing the full tensor and comparing to the identity map.
    # Or just by random matrix multiplication.
    for _ in range(5):
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        M = np.zeros(7)
        for r in range(7):
            L_A = np.dot(U[:, r], A_flat)
            L_B = np.dot(V[:, r], B_flat)
            M[r] = L_A * L_B
        
        C_computed = np.zeros((2, 2))
        for i in range(2):
            for k in range(2):
                idx = k * 2 + i
                C_computed[i, k] = np.dot(W[idx, :], M)
        
        if not np.allclose(C_computed, C_expected):
            return False
    return True

def construct():
    n, m, p = 2, 4, 8
    deadline = time.time() + 110
    
    # Try Strassen
    try:
        U_2, V_2, W_2 = _strassen_2x2_decomposition()
        
        # Verify Strassen on random inputs
        valid = _verify_strassen()
        
        if valid:
            # Construct 2x4x8 using 8 Strassen blocks
            rank = 8 * 7
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Mapping:
            # A is 2x4. Split into A0 (cols 0,1) and A1 (cols 2,3).
            # B is 4x8. Split into B0 (rows 0,1) and B1 (rows 2,3).
            # C = A0 B0 + A1 B1.
            # A0 is 2x2. B0 is 2x8.
            # Split B0 into 4 blocks of 2x2: B0_0, B0_1, B0_2, B0_3.
            # A0 B0 = A0 B0_0 + A0 B0_1 + A0 B0_2 + A0 B0_3.
            # Each is 2x2x2.
            
            for block_a in range(2): # 0 for A0, 1 for A1
                for block_b in range(4): # 0,1 for B0_0, B0_1; 2,3 for B1_0, B1_1
                    # Determine which Strassen block this is
                    block_idx = block_a * 4 + block_b
                    offset = block_idx * 7
                    
                    # Fill U
                    for i in range(2):
                        for j_local in range(2):
                            j_global = block_a * 2 + j_local
                            u_idx = i * 4 + j_global
                            for r in range(7):
                                U[u_idx, offset + r] = U_2[i * 2 + j_local, r]
                    
                    # Fill V
                    for j_local in range(2):
                        j_global = block_a * 2 + j_local
                        for k_local in range(2):
                            k_global = block_b * 2 + k_local
                            v_idx = j_global * 8 + k_global
                            for r in range(7):
                                V[v_idx, offset + r] = V_2[j_local * 2 + k_local, r]
                    
                    # Fill W
                    for i in range(2):
                        for k_local in range(2):
                            k_global = block_b * 2 + k_local
                            w_idx = k_global * 2 + i
                            for r in range(7):
                                W[w_idx, offset + r] = W_2[k_local * 2 + i, r]
                    
            # Convert to lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    except Exception:
        pass
    
    # Fallback to Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
