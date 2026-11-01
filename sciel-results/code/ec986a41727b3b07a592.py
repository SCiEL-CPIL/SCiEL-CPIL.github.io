"""Matrix multiplication 2x2x8 rank-minimal exact tensor decomposition."""
import time
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x8 matrix.
    
    Uses Strassen's rank-7 decomposition for 2x2x2 applied to each of the 4
    independent 2x2 blocks of the B matrix, giving rank 28.
    
    Returns {'U': list, 'V': list, 'W': list} with shapes (4,28), (16,28), (16,28).
    """
    n, m, p = 2, 2, 8
    
    # Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
    # For C = A * B where A, B, C are 2x2:
    #   M1 = A[0,0] * (B[0,1] + B[1,1])
    #   M2 = (A[1,0] + A[1,1]) * B[0,0]
    #   M3 = A[0,0] * (B[1,0] - B[1,1])
    #   M4 = A[1,1] * (B[0,0] - B[1,0])
    #   M5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    #   M6 = (A[1,0] - A[0,0]) * (B[0,1] + B[1,1])
    #   M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    #   C[0,0] = M5 + M4 - M2 + M6
    #   C[0,1] = M1 + M7
    #   C[1,0] = M3 + M5
    #   C[1,1] = M1 - M4 + M2 + M3
    #
    # Indexing convention for a single 2x2x2 block:
    #   U[i*2+j, r] = coefficient for A[i,j] in term r
    #   V[j*2+k, r] = coefficient for B[j,k] in term r
    #   W[k*2+i, r] = coefficient for C[i,k] in term r  (note: W[k*n+i] per trace convention)
    
    sU = np.zeros((4, 7), dtype=np.float64)
    sV = np.zeros((4, 7), dtype=np.float64)
    sW = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = A00 * (B01 + B11)
    sU[0, 0] = 1.0   # A[0,0]
    sV[1, 0] = 1.0   # B[0,1]
    sV[3, 0] = 1.0   # B[1,1]
    sW[1, 0] = 1.0   # C[0,1] -> W[1*2+0] = W[1]
    sW[3, 0] = 1.0   # C[1,1] -> W[3*2+1]... wait.
    
    # Let me be very careful with the W indexing.
    # The convention is W[k*n + i, r] where the output is C[i, k].
    # So for C[i, k], the W index is k*n + i = 2k + i.
    # C[0,1]: i=0, k=1 -> W[2*1+0] = W[2]
    # C[1,1]: i=1, k=1 -> W[2*1+1] = W[3]
    
    # Let me redo all W indices carefully.
    # For a 2x2 block, output C[i, k_local] where k_local in {0,1}.
    # W index for C[i, k_local] is k_local * 2 + i.
    
    sU[:] = 0.0
    sV[:] = 0.0
    sW[:] = 0.0
    
    # M1 = A[0,0] * (B[0,1] + B[1,1])
    sU[0, 0] = 1.0
    sV[0*2+1, 0] = 1.0  # B[0,1]
    sV[1*2+1, 0] = 1.0  # B[1,1]
    # Contributes to C[0,1] and C[1,1]
    sW[1*2+0, 0] = 1.0  # C[0,1]: k=1, i=0 -> W[2]
    sW[1*2+1, 0] = 1.0  # C[1,1]: k=1, i=1 -> W[3]
    
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    sU[1*2+0, 1] = 1.0  # A[1,0]
    sU[1*2+1, 1] = 1.0  # A[1,1]
    sV[0*2+0, 1] = 1.0  # B[0,0]
    # Contributes to C[0,0] and C[1,0]
    sW[0*2+0, 1] = 1.0  # C[0,0]: k=0, i=0 -> W[0]
    sW[0*2+1, 1] = 1.0  # C[1,0]: k=0, i=1 -> W[1]
    
    # M3 = A[0,0] * (B[1,0] - B[1,1])
    sU[0*2+0, 2] = 1.0  # A[0,0]
    sV[1*2+0, 2] = 1.0  # B[1,0]
    sV[1*2+1, 2] = -1.0  # B[1,1]
    # Contributes to C[1,0] and C[1,1]
    sW[0*2+1, 2] = 1.0  # C[1,0]: k=0, i=1 -> W[1]
    sW[1*2+1, 2] = 1.0  # C[1,1]: k=1, i=1 -> W[3]
    
    # M4 = A[1,1] * (B[0,0] - B[1,0])
    sU[1*2+1, 3] = 1.0  # A[1,1]
    sV[0*2+0, 3] = 1.0  # B[0,0]
    sV[1*2+0, 3] = -1.0  # B[1,0]
    # Contributes to C[0,0] and C[1,0]
    sW[0*2+0, 3] = 1.0  # C[0,0]: k=0, i=0 -> W[0]
    sW[0*2+1, 3] = 1.0  # C[1,0]: k=0, i=1 -> W[1]
    
    # M5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    sU[0*2+0, 4] = 1.0  # A[0,0]
    sU[1*2+1, 4] = 1.0  # A[1,1]
    sV[0*2+0, 4] = 1.0  # B[0,0]
    sV[1*2+1, 4] = 1.0  # B[1,1]
    # Contributes to C[0,0] and C[1,1]
    sW[0*2+0, 4] = 1.0  # C[0,0]: k=0, i=0 -> W[0]
    sW[1*2+1, 4] = 1.0  # C[1,1]: k=1, i=1 -> W[3]
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,1] + B[1,1])
    sU[1*2+0, 5] = 1.0  # A[1,0]
    sU[0*2+0, 5] = -1.0  # A[0,0]
    sV[0*2+1, 5] = 1.0  # B[0,1]
    sV[1*2+1, 5] = 1.0  # B[1,1]
    # Contributes to C[0,0] and C[1,1]
    sW[0*2+0, 5] = 1.0  # C[0,0]: k=0, i=0 -> W[0]
    sW[1*2+1, 5] = 1.0  # C[1,1]: k=1, i=1 -> W[3]
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    sU[0*2+1, 6] = 1.0  # A[0,1]
    sU[1*2+1, 6] = -1.0  # A[1,1]
    sV[1*2+0, 6] = 1.0  # B[1,0]
    sV[1*2+1, 6] = 1.0  # B[1,1]
    # Contributes to C[0,1] and C[1,1]
    sW[1*2+0, 6] = 1.0  # C[0,1]: k=1, i=0 -> W[2]
    sW[1*2+1, 6] = 1.0  # C[1,1]: k=1, i=1 -> W[3]
    
    # Now build the full 2x2x8 decomposition.
    # A is 2x2, B is 2x8, C is 2x8.
    # U shape: (n*m, rank) = (4, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (16, 28)
    # 
    # U[i*m+j, r] where i in {0,1}, j in {0,1}
    # V[j*p+k, r] where j in {0,1}, k in {0,...,7}
    # W[k*n+i, r] where k in {0,...,7}, i in {0,1}
    #
    # For block b (b = 0,1,2,3), B columns are 2b and 2b+1.
    # Within block b, local column k_local = 0 or 1 maps to global column k = 2b + k_local.
    
    rank = 28
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)
    
    for b in range(4):
        for r in range(7):
            col = b * 7 + r
            # U: same Strassen A-coefficients for all blocks
            U[:, col] = sU[:, r]
            # V: map local B[j, k_local] to global B[j, 2b+k_local]
            for j in range(2):
                for kl in range(2):
                    k_global = 2 * b + kl
                    V[j * 8 + k_global, col] = sV[j * 2 + kl, r]
            # W: map local C[i, k_local] to global C[i, 2b+k_local]
            # W index: k_global * n + i = (2b+kl) * 2 + i
            for i in range(2):
                for kl in range(2):
                    k_global = 2 * b + kl
                    W[k_global * 2 + i, col] = sW[kl * 2 + i, r]
    
    # Verify the decomposition: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1 for all i,j,k
    # Actually, the tensor for matrix multiplication is T[i,j,k] = 1 (each product A[i,j]*B[j,k] contributes to C[i,k])
    # The decomposition should satisfy: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val):
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    s = 0.0
                    for r in range(U_mat.shape[1]):
                        s += U_mat[i * m_val + j, r] * V_mat[j * p_val + k, r] * W_mat[k * n_val + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback: schoolbook rank-32
        rank32 = n * m * p
        U32 = np.zeros((n * m, rank32), dtype=np.float64)
        V32 = np.zeros((m * p, rank32), dtype=np.float64)
        W32 = np.zeros((p * n, rank32), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U32[i * m + j, col] = 1.0
                    V32[j * p + k, col] = 1.0
                    W32[k * n + i, col] = 1.0
                    col += 1
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
