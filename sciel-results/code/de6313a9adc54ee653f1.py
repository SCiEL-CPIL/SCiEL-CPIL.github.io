"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 130.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_2x2x4_tensors():
    """
    Build 2x2x4 decomposition, rank 14.
    U: (4, 14)
    V: (8, 14)
    W: (8, 14)
    """
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((4, 14), dtype=np.int16)
    V4 = np.zeros((8, 14), dtype=np.int16)
    W4 = np.zeros((8, 14), dtype=np.int16)
    
    # First 7 columns: for B_L (cols 0,1)
    U4[:, 0:7] = U2
    # V indices for B_L: j*2 + k where j,k in 0,1. Range 0..3
    for j in range(2):
        for k in range(2):
            V4[j*2+k, 0:7] = V2[j*2+k, :]
    # W indices for C_L: k*2 + i where k,i in 0,1. Range 0..3
    for k in range(2):
        for i in range(2):
            W4[k*2+i, 0:7] = W2[k*2+i, :]
            
    # Last 7 columns: for B_R (cols 2,3)
    U4[:, 7:14] = U2
    # V indices for B_R: j*2 + (k+2). Range 4..7
    for j in range(2):
        for k in range(2):
            V4[j*2+k+2, 7:14] = V2[j*2+k, :]
    # W indices for C_R: k*2 + i + 2. Range 4..7
    for k in range(2):
        for i in range(2):
            W4[k*2+i+2, 7:14] = W2[k*2+i, :]
            
    return U4, V4, W4

def _build_4x4x8_tensors():
    """
    Build 4x4x8 decomposition using Kronecker product of 2x2x2 (rank 7) and 2x2x4 (rank 14).
    Total rank 98.
    U: (16, 98)
    V: (32, 98)
    W: (32, 98)
    """
    U2, V2, W2 = _strassen_2x2x2()
    U24, V24, W24 = _build_2x2x4_tensors()
    
    rank = 98
    U = np.zeros((16, rank), dtype=np.int16)
    V = np.zeros((32, rank), dtype=np.int16)
    W = np.zeros((32, rank), dtype=np.int16)
    
    # Kronecker product structure:
    # c = c0 * 14 + c1
    # i = i0 * 2 + i1
    # j = j0 * 2 + j1
    # k = k0 * 2 + k1 (for 2x2x4, k is 0..7, so k0 is 0..3? No.
    # In 2x2x4, k is 0..7.
    # In 4x4x8, k is 0..7.
    # The Kronecker product of T[2,2,2] and T[2,2,4] gives T[4,4,8].
    # U[i0*2+i1, j0*2+j1, c0*14+c1] = U2[i0*2+i1? No.
    
    # Let's do it explicitly.
    for c0 in range(7):
        for c1 in range(14):
            c = c0 * 14 + c1
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U[i*4+j, c] = U2[i0*2+j0, c0] * U24[i1*2+j1, c1]
            
            for j in range(4):
                for k in range(8):
                    j0, j1 = j // 2, j % 2
                    # k in 0..7.
                    # V2 is (4,7). V24 is (8,14).
                    # Decomposition:
                    # j = j0 * 2 + j1
                    # k = k0 * 4 + k1
                    k0, k1 = k // 4, k % 4
                    V[j*8+k, c] = V2[j0*2+j1, k0] * V24[j1, k1]
                    # The 2x2x4 tensor handles j in 0,1 and k in 0..7.
                    # The 2x2x2 tensor handles j in 0,1 and k in 0,1.
                    # So k in 4x4x8 is split into k_high (0,1) and k_low (0..7)?
                    # No.
                    # T[4,4,8] = T[2,2,2] (on i,j,k_high) x T[2,2,4] (on i1,j1,k_low)?
                    # This doesn't match dimensions.
                    
                    # Let's use the block structure directly.
                    # We already have _build_4x4x8_shared which works.
                    # I will just use that logic but cleaner.
                    pass
            
            # Revert to the working block logic from current program
            # It is simpler and verified.
            pass
            
    # Fallback to the working implementation
    U4, V4, W4 = _build_4x4x4_tensors()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
                
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    return U4, V4, W4

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    U, V, W = _build_4x4x8_tensors()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
