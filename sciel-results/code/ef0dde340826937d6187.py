import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _strassen_2x2():
    """Return U, V, W for Strassen 2x2x2, shapes (4,7), (4,7), (4,7)"""
    # Indexing: A[i,k] -> index i*2+k, B[k,j] -> index k*2+j, C[i,j] -> index i*2+j
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3 = A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # M4 = A11(B01-B00)
    U[3,3] = 1
    V[0,3] = -1; V[1,3] = 1
    W[0,3] = -1; W[3,3] = 1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1; W[1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[0,5] = -1; U[2,5] = 1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[1,6] = -1
    
    return U, V, W

def _schoolbook(a, b, c):
    """Return U, V, W for schoolbook a×b × b×c, shapes (a*b, a*b*c), (b*c, a*b*c), (a*c, a*b*c)"""
    rank = a * b * c
    U = np.zeros((a * b, rank))
    V = np.zeros((b * c, rank))
    W = np.zeros((a * c, rank))
    col = 0
    for i in range(a):
        for k in range(b):
            for j in range(c):
                U[i * b + k, col] = 1.0
                V[k * c + j, col] = 1.0
                W[i * c + j, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Partition: [2, 2, 1]
    splits = [2, 2, 1]
    row_starts = [0, 2, 4, 5]
    mid_starts = [0, 2, 4, 5]
    col_starts = [0, 2, 4, 5]
    
    # Collect all (U, V, W) blocks
    all_U = []
    all_V = []
    all_W = []
    
    for i in range(3):
        for k in range(3):
            for j in range(3):
                a = splits[i]
                b = splits[k]
                c = splits[j]
                
                if a == 2 and b == 2 and c == 2:
                    U_blk, V_blk, W_blk = _strassen_2x2()
                else:
                    U_blk, V_blk, W_blk = _schoolbook(a, b, c)
                
                # Offset indices
                row_off = row_starts[i]
                mid_off = mid_starts[k]
                col_off = col_starts[j]
                
                # U has shape (a*b, rank), indices are (row_off + i_local, mid_off + k_local)
                # We need to place this into the full (n*m, rank) matrix
                # U_local[i_local * b + k_local, r] -> U_full[(row_off + i_local) * m + (mid_off + k_local), r]
                
                rank_blk = U_blk.shape[1]
                U_full = np.zeros((n * m, rank_blk))
                V_full = np.zeros((m * p, rank_blk))
                W_full = np.zeros((p * n, rank_blk))
                
                for i_local in range(a):
                    for k_local in range(b):
                        U_full[(row_off + i_local) * m + (mid_off + k_local), :] = U_blk[i_local * b + k_local, :]
                
                for k_local in range(b):
                    for j_local in range(c):
                        V_full[(mid_off + k_local) * p + (col_off + j_local), :] = V_blk[k_local * c + j_local, :]
                
                for i_local in range(a):
                    for j_local in range(c):
                        W_full[(col_off + j_local) * n + (row_off + i_local), :] = W_blk[i_local * c + j_local, :]
                
                all_U.append(U_full)
                all_V.append(V_full)
                all_W.append(W_full)
    
    # Concatenate all blocks
    U = np.hstack(all_U)
    V = np.hstack(all_V)
    W = np.hstack(all_W)
    
    rank = U.shape[1]
    
    # Verify the decomposition
    # T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Should equal 1 if i,k,j are valid indices, 0 otherwise
    # Actually, the tensor for matrix multiplication is:
    # T[i,k,j] = 1 if the product A[i,k]*B[k,j] contributes to C[i,j]
    # In the standard formulation, the tensor is T[i,k,j] = delta_{appropriate}
    
    # The bilinear algorithm computes: C[i,j] = sum_k A[i,k] B[k,j] = sum_r (sum_k U[i*m+k,r] A[i,k]) * (sum_k V[k*p+j,r] B[k,j]) * W[j*n+i,r]
    # Wait, the standard formulation is:
    # C = sum_r (U^T A_flat) * (V^T B_flat) * W
    # where A_flat is A reshaped to (n*m,), B_flat is B reshaped to (m*p,), C is reshaped to (p*n,)
    # C_flat[j*n+i] = sum_r U[i*m+k... 
    
    # Actually, the correct verification is:
    # For the matrix multiplication tensor, T[i, k, j] = 1 for all valid i, k, j
    # (since C[i,j] = sum_k A[i,k] * B[k,j])
    # The decomposition gives: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # Let's verify with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    # Compute using the decomposition
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        A_proj = np.zeros((n, m))
        B_proj = np.zeros((m, p))
        for i in range(n):
            for k in range(m):
                A_proj[i, k] = U[i * m + k, r]
        for k in range(m):
            for j in range(p):
                B_proj[k, j] = V[k * p + j, r]
        C_proj = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                C_proj[i, j] = W[j * n + i, r]
        C_decomp += (A_proj * A_test) @ (B_proj * B_test) * C_proj
    
    # Hmm, this isn't quite right. Let me think about the correct verification.
    # The bilinear algorithm computes:
    # C[i,j] = sum_r M_r * W[j*n+i, r]
    # where M_r = (sum_{i,k} U[i*m+k,r] * A[i,k]) * (sum_{k,j} V[k*p+j,r] * B[k,j])
    
    C_decomp2 = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r]  # shape (n*m,)
        v_vec = V[:, r]  # shape (m*p,)
        w_vec = W[:, r]  # shape (p*n,)
        
        # u_vec reshaped to (n,m), v_vec reshaped to (m,p)
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        
        # M_r = sum_{i,k} u_mat[i,k] * A[i,k] * sum_{k,j} v_mat[k,j] * B[k,j]
        # Wait, that's not right either.
        
        # The correct formula: M_r = (u_vec dot A_flat) * (v_vec dot B_flat)
        # where A_flat = A.reshape(n*m) and B_flat = B.reshape(m*p)
        # Then C_flat[j*n+i] += M_r * W[j*n+i, r]
        
        A_flat = A_test.reshape(n * m)
        B_flat = B_test.reshape(m * p)
        M_r = np.dot(u_vec, A_flat) * np.dot(v_vec, B_flat)
        C_decomp2 += M_r * w_vec.reshape(p, n).T  # w_vec is indexed as [j*n+i]
    
    # Wait, W is indexed as W[k*n+i] per the problem statement: "W[k*n+i,r]"
    # So W[j*n+i, r] is the coefficient for C[i,j]
    # C[i,j] = sum_r M_r * W[j*n+i, r]
    # C_flat[i*p+j] or C_flat[j*n+i]?
    
    # The problem says W[k*n+i,r] where k is the column index and i is the row index.
    # So W has shape (p*n, rank) and W[j*n+i, r] corresponds to C[i,j].
    # C[i,j] = sum_r M_r * W[j*n+i, r]
    
    C_decomp3 = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r]
        v_vec = V[:, r]
        w_vec = W[:, r]
        M_r = np.dot(u_vec, A_test.reshape(n*m)) * np.dot(v_vec, B_test.reshape(m*p))
        w_mat = w_vec.reshape(p, n)  # w_mat[j, i] = W[j*n+i, r]
        C_decomp3 += M_r * w_mat.T  # C_decomp3[i,j] += M_r * w_mat[j,i]
    
    if not np.allclose(C_decomp3, C_test, atol=1e-10):
        # Decomposition is invalid, fall back to schoolbook
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1.0
                    V[k * p + j, col] = 1.0
                    W[j * n + i, col] = 1.0
                    col += 1
        rank = rank_schoolbook
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
