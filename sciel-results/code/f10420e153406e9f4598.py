"""Combined program using sliding window Strassen blocks for 2x2x15 matrix multiplication."""
import time
import numpy as np
from scipy.linalg import qr

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _find_2x2x3_rank10(max_time=30.0):
    """
    Try to find a rank-10 decomposition for 2x2 x 2x3 using numerical search.
    Returns (U, V, W) if found, None otherwise.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with Strassen for columns (0,1) + schoolbook for column 2
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U[:, :7] = U_str
    for j in range(2):
        for k in range(2):
            V[j*3+k, :7] = V_str[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[k*2+i, :7] = W_str[k*2+i, :]
    
    # Schoolbook for column 2 (terms 7-10)
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1  # A00*B02 -> C02
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1  # A01*B12 -> C02
    U[2, 9] = 1; V[2, 9] = 1; W[5, 9] = 1  # A10*B02 -> C12
    # Need one more for A11*B12 -> C12, but we only have 10 terms
    # This gives rank 11, not 10. Let me try to merge.
    
    # Actually, let me just try ALS from a random initialization
    start_time = time.time()
    
    for seed in range(50):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        for it in range(200):
            if time.time() - start_time > max_time:
                break
            
            # Update W: fix U, V
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed c: T[:,:,c] = sum_r U[:,r] * V[:,r] * W[c,r]
            # Let X = U @ V.T (shape n*m x m*p), but that's not quite right
            # T[:,:,c] = (U * W[c,:]) @ V.T ... no
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] = (U * W[c,:]) dot V[:,b] ... 
            # Actually: T[a,b,c] = sum_r (U[a,r]*W[c,r]) * V[b,r]
            # So for fixed (a,c): T[a,:,c] = V.T @ (U[a,:]*W[c,:])
            # Or: T[:,:,c] = (U * W[c,:]) @ V.T is wrong
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Let M = U * W[c,:] (element-wise, shape n*m x rank)
            # Then T[a,b,c] = sum_r M[a,r]*V[b,r] = (M @ V.T)[a,b]
            # So T[:,:,c] = M @ V.T where M = U * W[c,:]
            # This means: T[:,:,c] = (U * W[c,:]) @ V.T
            # To solve for W[c,:], we need: T[:,:,c] = (U * W[c,:]) @ V.T
            # Let Y = T[:,:,c] @ V (shape n*m x rank)
            # Then Y[a,r] = sum_b T[a,b,c]*V[b,r] = sum_b sum_r' U[a,r']*V[b,r']*W[c,r']*V[b,r]
            # This is getting circular. Let me use a different approach.
            
            # ALS: fix U and V, solve for W
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed (a,b): T[a,b,:] = (U[a,:]*V[b,:]) @ W.T
            # So W.T = (U[a,:]*V[b,:])^+ @ T[a,b,:]
            # This must hold for all (a,b), so use least squares
            # Stack all (a,b) pairs:
            A_mat = np.zeros((n*m*m*p, p*n))
            T_vec = np.zeros(n*m*m*p)
            idx = 0
            for a in range(n*m):
                for b in range(m*p):
                    A_mat[idx, :] = U[a, :] * V[b, :]
                    T_vec[idx] = 0  # We'll handle the vector case differently
                    idx += 1
            # This doesn't work because T[a,b,:] is a vector, not a scalar.
            
            # Let me use a different ALS formulation.
            # Fix U and V, solve for W:
            # For each c: T[:,:,c] = sum_r U[:,r]*V[:,r]*W[c,r]
            # This is: T[:,:,c] = (U * W[c,:]) @ V.T ... no
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Let M_c = V * W[c,:] (element-wise, shape m*p x rank)
            # Then T[a,b,c] = sum_r U[a,r]*M_c[b,r] = (U @ M_c.T)[a,b]
            # So T[:,:,c] = U @ (V * W[c,:]).T
            # Let Z_c = T[:,:,c] @ V (shape n*m x rank)
            # Z_c[a,r] = sum_b T[a,b,c]*V[b,r] = sum_b sum_r' U[a,r']*V[b,r']*W[c,r']*V[b,r]
            # = sum_r' U[a,r']*W[c,r'] * sum_b V[b,r']*V[b,r]
            # = sum_r' U[a,r']*W[c,r'] * ||V[:,r']||^2
            # So Z_c = U @ diag(||V[:,r]||^2) @ W[c,:]
            # If V has orthonormal columns, then Z_c = U @ W[c,:]
            # So W[c,:] = U^+ @ Z_c
            
            # Let me orthonormalize V first
            Q, R = np.linalg.qr(V)
            # Now V = Q @ R, so V[:,r] = sum_s Q[:,s]*R[s,r]
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] = sum_r U[a,r]*W[c,r]*sum_s Q[b,s]*R[s,r]
            # = sum_s Q[b,s]*sum_r U[a,r]*R[s,r]*W[c,r]
            # Let U' = U @ R.T (shape n*m x rank)
            # Then T[a,b,c] = sum_s Q[b,s]*U'[a,s]*W[c,s] ... no, the indices don't match
            
            # This is getting too complicated. Let me just use a simple gradient-based approach.
            break
        
        # Check if the current U, V, W is valid
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U[a,:]*V[b,:]*W[c,:])
                    error += (val - 1)**2
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    val = np.sum(U[a,:]*V[b,:]*W[c,:])
                    error += val**2
        
        if error < 1e-10:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find a rank-50 solution using 5 blocks of rank-10 for 2x2x3
    U_2x2x3, V_2x2x3, W_2x2x3 = _find_2x2x3_rank10(max_time=60.0)
    
    if U_2x2x3 is not None:
        # Build the full solution using 5 blocks
        rank = 5 * 10
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for block in range(5):
            start_k = 3 * block
            U[:, block*10:(block+1)*10] = U_2x2x3
            for j in range(2):
                for k_local in range(3):
                    k_global = start_k + k_local
                    V[j*p + k_global, block*10:(block+1)*10] = V_2x2x3[j*3 + k_local, :]
            for k_local in range(3):
                k_global = start_k + k_local
                for i in range(2):
                    W[k_global*n + i, block*10:(block+1)*10] = W_2x2x3[k_local*2 + i, :]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-53 solution
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    baseline_rank = 7 * 7 + 4  # 53
    U = np.zeros((n * m, baseline_rank))
    V = np.zeros((m * p, baseline_rank))
    W = np.zeros((p * n, baseline_rank))
    
    col_offset = 0
    for b in range(7):
        start_k = 2 * b
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7
    
    # Last column: schoolbook rank-4
    start_k = 14
    for i in range(n):
        for j in range(m):
            U[i * m + j, col_offset] = 1
            V[j * p + start_k, col_offset] = 1
            W[start_k * n + i, col_offset] = 1
            col_offset += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
