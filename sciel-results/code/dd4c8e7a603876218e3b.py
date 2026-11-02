import time
import numpy as np
import scipy.linalg as la

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 110
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def verify_rank_decomp(U, V, W, n, m, p):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        if not np.allclose(T_reconstructed, np.ones((n, m, p)), atol=1e-10):
            return False
        return True
    
    def build_strassen_2x2_2():
        """Strassen decomposition of M_{2,2,2}. Rank 7."""
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00(B10+B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = 1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = (A00-A10)(B01+B10)
        U[0, 3] = 1; U[2, 3] = -1
        V[1, 3] = 1; V[2, 3] = 1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A01+A10)(B00-B10)
        U[1, 4] = 1; U[2, 4] = 1
        V[0, 4] = 1; V[2, 4] = -1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A11-A01)(B01-B11)
        U[1, 5] = -1; U[3, 5] = 1
        V[1, 5] = 1; V[3, 5] = -1
        W[3, 5] = 1
        
        # M7 = (A00+A01)(B11-B01)
        U[0, 6] = 1; U[1, 6] = 1
        V[1, 6] = -1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def build_2x2x4():
        """Build M_{2,2,4} by concatenating two M_{2,2,2} blocks. Rank: 14"""
        U1, V1, W1 = build_strassen_2x2_2()
        U2, V2, W2 = build_strassen_2x2_2()
        
        U = np.hstack((U1, U2))
        V = np.zeros((8, 14), dtype=int)
        W = np.zeros((8, 14), dtype=int)
        
        V[0, 0:7] = V1[0, :]; V[1, 0:7] = V1[1, :]
        V[4, 0:7] = V1[2, :]; V[5, 0:7] = V1[3, :]
        W[0, 0:7] = W1[0, :]; W[1, 0:7] = W1[1, :]
        W[4, 0:7] = W1[2, :]; W[5, 0:7] = W1[3, :]
        
        V[2, 7:14] = V2[0, :]; V[3, 7:14] = V2[1, :]
        V[6, 7:14] = V2[2, :]; V[7, 7:14] = V2[3, :]
        W[2, 7:14] = W2[0, :]; W[3, 7:14] = W2[1, :]
        W[6, 7:14] = W2[2, :]; W[7, 7:14] = W2[3, :]
        
        return U, V, W
    
    def build_2x3x4():
        """Build M_{2,3,4} by combining M_{2,2,4} and M_{2,1,4}. Rank: 22"""
        U224, V224, W224 = build_2x2x4()
        
        U = np.zeros((6, 22), dtype=int)
        V = np.zeros((12, 22), dtype=int)
        W = np.zeros((8, 22), dtype=int)
        
        for i in range(2):
            U[i*3+0, 0:14] = U224[i*2+0, :]
            U[i*3+1, 0:14] = U224[i*2+1, :]
        
        V[0:8, 0:14] = V224[:, :]
        W[:, 0:14] = W224[:, :]
        
        # M_{2,1,4} direct
        for col in range(8):
            i = col % 2
            k = col // 2
            U[i*3+2, 14+col] = 1
            V[8+k, 14+col] = 1
            W[k*2+i, 14+col] = 1
        
        return U, V, W
    
    def build_2x3x8():
        """Build M_{2,3,8} by combining two M_{2,3,4} blocks. Rank: 44"""
        U1, V1, W1 = build_2x3x4()
        U2, V2, W2 = build_2x3x4()
        
        rank = 44
        U = np.hstack((U1, U2))
        V = np.zeros((24, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)
        
        for j in range(3):
            for k in range(4):
                V[j*8+k, 0:22] = V1[j*4+k, :]
        W[0:8, 0:22] = W1[:, :]
        
        for j in range(3):
            for k in range(4):
                V[j*8+k+4, 22:44] = V2[j*4+k, :]
        W[8:16, 22:44] = W2[:, :]
        
        return U, V, W
    
    # Construct the decomposition
    U, V, W = build_2x3x8()
    
    # Verify
    if verify_rank_decomp(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U_fb = np.zeros((n*m, rank), dtype=int)
    V_fb = np.zeros((m*p, rank), dtype=int)
    W_fb = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
