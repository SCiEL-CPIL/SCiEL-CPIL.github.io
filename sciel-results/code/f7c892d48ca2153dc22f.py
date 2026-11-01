"""
Program to find a rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
Strategy: Construct a rank-14 decomposition using two independent Strassen 2x2x2 blocks.
Verify the decomposition. If valid, return rank 14. Otherwise, fallback to schoolbook rank 16.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=4):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct tensor T[i, j, k]
    # T[i,j,k] should be 1
    # Using einsum for efficiency:
    # U_idx: i*m+j, V_idx: j*p+k, W_idx: k*n+i
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's just do it explicitly for small sizes to be safe and clear
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        # u_r is vector of length n*m
        # v_r is vector of length m*p
        # w_r is vector of length p*n
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        # Reshape u_r to (n, m), v_r to (m, p), w_r to (p, n)
        # Note: W index is k*n + i, so W reshaped to (p, n) has W[k, i] = W_flat[k*n+i]
        # U index is i*m + j, so U reshaped to (n, m) has U[i, j] = U_flat[i*m+j]
        # V index is j*p + k, so V reshaped to (m, p) has V[j, k] = V_flat[j*p+k]
        
        u_mat = u_r.reshape(n, m)
        v_mat = v_r.reshape(m, p)
        w_mat = w_r.reshape(p, n)
        
        # The product term for rank r contributes to T[i,j,k] as:
        # U[i,j] * V[j,k] * W[k,i]
        # This is the trace of U[:,j] * V[j,:] * W[:,i] ... no, it's a scalar for each i,j,k
        # We can compute this by iterating or using einsum on the reshaped arrays
        # Term_{i,j,k} = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        # Efficient way:
        # Sum over r of (u_mat_r * v_mat_r * w_mat_r_transposed?)
        # Actually, let's just sum up the contributions
        # Since n, m, p are small, explicit loops are fine and less error-prone than getting einsum indices right for this specific cyclic pattern.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]

    # Check if T_recon is all ones
    if not np.allclose(T_recon, 1.0, atol=1e-9):
        return False
    return True

def _strassen_rank14():
    """
    Construct the rank-14 decomposition using two Strassen blocks.
    Left block handles B columns 0,1 -> C columns 0,1
    Right block handles B columns 2,3 -> C columns 2,3
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))  # 4 x 14
    V = np.zeros((m*p, rank))  # 8 x 14
    W = np.zeros((p*n, rank))  # 8 x 14
    
    # Indices:
    # U: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # V: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # W: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
    
    # Strassen Products for 2x2 * 2x2:
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B10 - B11)
    # m4 = A11 * (B01 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B01)
    # m7 = (A01 - A11) * (B10 + B11)
    
    # Outputs:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Left Block (Columns 0, 1)
    # B00=V[0], B01=V[1], B10=V[4], B11=V[5]
    # C00=W[0], C01=W[1], C10=W[4], C11=W[5]
    
    # m1_L
    r = 0
    U[0, r] = 1; U[3, r] = 1
    V[0, r] = 1; V[5, r] = 1
    W[0, r] = 1; W[5, r] = 1
    
    # m2_L
    r = 1
    U[2, r] = 1; U[3, r] = 1
    V[0, r] = 1
    W[4, r] = 1; W[5, r] = -1
    
    # m3_L
    r = 2
    U[0, r] = 1
    V[4, r] = 1; V[5, r] = -1
    W[1, r] = 1; W[5, r] = 1
    
    # m4_L
    r = 3
    U[3, r] = 1
    V[1, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[4, r] = 1
    
    # m5_L
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[5, r] = 1
    W[1, r] = 1; W[0, r] = -1
    
    # m6_L
    r = 5
    U[2, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[5, r] = 1
    
    # m7_L
    r = 6
    U[1, r] = 1; U[3, r] = -1
    V[4, r] = 1; V[5, r] = 1
    W[0, r] = 1
    
    # Right Block (Columns 2, 3)
    # B02=V[2], B03=V[3], B12=V[6], B13=V[7]
    # C02=W[2], C03=W[3], C12=W[6], C13=W[7]
    
    # m1_R
    r = 7
    U[0, r] = 1; U[3, r] = 1
    V[2, r] = 1; V[7, r] = 1
    W[2, r] = 1; W[7, r] = 1
    
    # m2_R
    r = 8
    U[2, r] = 1; U[3, r] = 1
    V[2, r] = 1
    W[6, r] = 1; W[7, r] = -1
    
    # m3_R
    r = 9
    U[0, r] = 1
    V[6, r] = 1; V[7, r] = -1
    W[3, r] = 1; W[7, r] = 1
    
    # m4_R
    r = 10
    U[3, r] = 1
    V[3, r] = 1; V[2, r] = -1
    W[2, r] = 1; W[6, r] = 1
    
    # m5_R
    r = 11
    U[0, r] = 1; U[1, r] = 1
    V[7, r] = 1
    W[3, r] = 1; W[2, r] = -1
    
    # m6_R
    r = 12
    U[2, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[7, r] = 1
    
    # m7_R
    r = 13
    U[1, r] = 1; U[3, r] = -1
    V[6, r] = 1; V[7, r] = 1
    W[2, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try rank 14
    result_14 = _strassen_rank14()
    
    if _verify(result_14["U"], result_14["V"], result_14["W"], n, m, p):
        return result_14
    
    # Fallback to schoolbook rank 16
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
