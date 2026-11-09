"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B11 + B10)
    
    # U factors (A matrix)
    # Row 0: A00, Row 1: A01, Row 2: A10, Row 3: A11
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 += M1, C11 += M1
    W[1, 0] = -1; W[2, 0] = -1 # C01 -= M1, C10 -= M1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1 # C01 += M2, C11 += M2
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[2, 2] = 1 # C00 += M3, C10 += M3
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = 1 # C01 += M4, C11 += M4
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1 # C00 += M5, C10 += M5
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1 # C00 += M6, C10 += M6
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1 # C01 += M7, C11 += M7
    
    return U, V, W

def _build_2x2x4():
    """
    Builds decomposition for 2x2x4 matrix multiplication.
    Rank 14 (two 2x2x2 Strassen).
    U: (4, 14), V: (8, 14), W: (8, 14)
    U[i*2+j, c]
    V[j*4+k, c]
    W[k*2+i, c]
    """
    U1, V1, W1 = _strassen_2x2x2()
    
    U = np.zeros((4, 14))
    V = np.zeros((8, 14))
    W = np.zeros((8, 14))
    
    # First 2x2x2 for k=0,1
    U[:, 0:7] = U1
    V[0:4, 0:7] = V1 # j=0,1 -> k=0,1
    W[0:4, 0:7] = W1 # k=0,1 -> i=0,1
    
    # Second 2x2x2 for k=2,3
    U[:, 7:14] = U1
    V[4:8, 7:14] = V1 # j=0,1 -> k=2,3 (mapped to local 0,1)
    W[4:8, 7:14] = W1 # k=2,3 -> i=0,1 (mapped to local 0,1)
    
    return U, V, W

def _build_2x4x4():
    """
    Builds decomposition for 2x4x4 matrix multiplication.
    Rank 28 (two 2x2x4).
    U: (8, 28), V: (16, 28), W: (16, 28)
    U[i*4+j, c]
    V[j*4+k, c]
    W[k*2+i, c]
    """
    U1, V1, W1 = _build_2x2x4()
    
    U = np.zeros((8, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))
    
    # First 2x2x4 for j=0,1
    U[0:4, 0:14] = U1
    V[0:8, 0:14] = V1
    W[:, 0:14] = W1
    
    # Second 2x2x4 for j=2,3
    U[4:8, 14:28] = U1
    V[8:16, 14:28] = V1
    W[:, 14:28] = W1
    
    return U, V, W

def _build_scalar_mults(n_u, n_v, n_w, rank):
    """
    Builds decomposition for scalar multiplications (dot products).
    Used for 1xm x px1 or similar simple cases.
    """
    U = np.zeros((n_u, rank))
    V = np.zeros((n_v, rank))
    W = np.zeros((n_w, rank))
    
    for c in range(rank):
        # This assumes we are decomposing a specific tensor structure
        # For simplicity, we will construct the full matrices manually in the main function
        pass
        
    return U, V, W

def construct():
    deadline = time.time() + 110
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    
    # Block 1: 2x4x4 (i=0,1; j=0..3; k=0..3)
    U1, V1, W1 = _build_2x4x4()
    
    # Block 2: 2x4x2 (i=0,1; j=0..3; k=4,5)
    # This is 4 dot products of length 4.
    # U: (8, 4), V: (16, 4), W: (4, 4)
    U2 = np.zeros((8, 4))
    V2 = np.zeros((16, 4))
    W2 = np.zeros((4, 4))
    
    # C_{0,4} = sum A_{0,j} B_{j,4}
    # C_{0,5} = sum A_{0,j} B_{j,5}
    # C_{1,4} = sum A_{1,j} B_{j,4}
    # C_{1,5} = sum A_{1,j} B_{j,5}
    
    # Term 0: C_{0,4}
    U2[0:4, 0] = 1 # A_{0,0..3}
    V2[0:4, 0] = 1 # B_{0..3, 4}
    W2[0, 0] = 1   # C_{0, 0} (local k=0)
    
    # Term 1: C_{0,5}
    U2[0:4, 1] = 1
    V2[4:8, 1] = 1 # B_{0..3, 5}
    W2[1, 1] = 1   # C_{0, 1} (local k=1)
    
    # Term 2: C_{1,4}
    U2[4:8, 2] = 1 # A_{1,0..3}
    V2[0:4, 2] = 1
    W2[2, 2] = 1   # C_{1, 0}
    
    # Term 3: C_{1,5}
    U2[4:8, 3] = 1
    V2[4:8, 3] = 1
    W2[3, 3] = 1   # C_{1, 1}
    
    # Block 3: 1x4x4 (i=2; j=0..3; k=0..3)
    # 4 dot products.
    U3 = np.zeros((4, 4))
    V3 = np.zeros((16, 4))
    W3 = np.zeros((8, 4))
    
    for k in range(4):
        U3[:, k] = 1 # A_{2, 0..3}
        V3[k*4:(k+1)*4, k] = 1 # B_{0..3, k}
        W3[k, k] = 1 # C_{2, k}
        
    # Block 4: 1x4x2 (i=2; j=0..3; k=4,5)
    # 2 dot products.
    U4 = np.zeros((4, 2))
    V4 = np.zeros((16, 2))
    W4 = np.zeros((4, 2))
    
    for k in range(2):
        U4[:, k] = 1
        V4[(4+k)*4:(4+k+1)*4, k] = 1 # B_{0..3, 4+k}
        W4[k, k] = 1 # C_{2, 4+k}
        
    # Concatenate
    U = np.hstack([U1, U2, U3, U4])
    V = np.hstack([V1, V2, V3, V4])
    W = np.hstack([W1, W2, W3, W4])
    
    # Verify
    # U shape: (12, 38)
    # V shape: (24, 38)
    # W shape: (18, 38)
    
    Ur = U.reshape(3, 4, U.shape[1])
    Vr = V.reshape(4, 6, V.shape[1])
    Wr = W.reshape(6, 3, W.shape[1])
    
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((3, 4, 6))
    
    if not np.allclose(T, target, atol=1e-6):
        # Fallback to schoolbook if verification fails
        U_sb = np.zeros((12, 72))
        V_sb = np.zeros((24, 72))
        W_sb = np.zeros((18, 72))
        col = 0
        for i in range(3):
            for j in range(4):
                for k in range(6):
                    U_sb[i*4+j, col] = 1
                    V_sb[j*6+k, col] = 1
                    W_sb[k*3+i, col] = 1
                    col += 1
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    construct()
