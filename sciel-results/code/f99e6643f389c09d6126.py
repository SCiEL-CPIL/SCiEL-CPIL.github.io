"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p).
    Returns U (n*m x nmp), V (m*p x nmp), W (p*n x nmp)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    
    Convention: T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    
    Using standard Strassen:
      M1 = (a00+a11)(b00+b11)
      M2 = (a10+a11)b00
      M3 = a00(b01-b11)
      M4 = a11(b10-b00)
      M5 = (a00+a10)b11
      M6 = (a11-a10)(b00+b10)
      M7 = (a01-a11)(b01+b11)
      
      c00 = M1 + M4 - M5 + M7
      c01 = M3 + M5
      c10 = M2 + M4
      c11 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # r=0: M1 = (a00+a11)(b00+b11)
    # U: a00=(i=0,j=0), a11=(i=1,j=1)
    U[0, 0] = 1  # i=0, j=0
    U[3, 0] = 1  # i=1, j=1
    # V: b00=(j=0,k=0), b11=(j=1,k=1)
    V[0, 0] = 1  # j=0, k=0
    V[3, 0] = 1  # j=1, k=1
    # W: c00(k=0,i=0)=+1, c11(k=1,i=1)=+1
    W[0, 0] = 1   # k=0, i=0
    W[3, 0] = 1   # k=1, i=1
    
    # r=1: M2 = (a10+a11)b00
    # U: a10=(i=1,j=0), a11=(i=1,j=1)
    U[2, 1] = 1  # i=1, j=0
    U[3, 1] = 1  # i=1, j=1
    # V: b00=(j=0,k=0)
    V[0, 1] = 1  # j=0, k=0
    # W: c10(k=0,i=1)=+1, c11(k=1,i=1)=-1
    W[1, 1] = 1   # k=0, i=1
    W[3, 1] = -1  # k=1, i=1
    
    # r=2: M3 = a00(b01-b11)
    # U: a00=(i=0,j=0)
    U[0, 2] = 1  # i=0, j=0
    # V: b01=(j=0,k=1), b11=(j=1,k=1)
    V[1, 2] = 1   # j=0, k=1
    V[3, 2] = -1  # j=1, k=1
    # W: c01(k=1,i=0)=+1, c11(k=1,i=1)=+1
    W[2, 2] = 1   # k=1, i=0
    W[3, 2] = 1   # k=1, i=1
    
    # r=3: M4 = a11(b10-b00)
    # U: a11=(i=1,j=1)
    U[3, 3] = 1  # i=1, j=1
    # V: b10=(j=1,k=0), b00=(j=0,k=0)
    V[2, 3] = 1   # j=1, k=0
    V[0, 3] = -1  # j=0, k=0
    # W: c00(k=0,i=0)=+1, c10(k=0,i=1)=+1
    W[0, 3] = 1   # k=0, i=0
    W[1, 3] = 1   # k=0, i=1
    
    # r=4: M5 = (a00+a10)b11
    # U: a00=(i=0,j=0), a10=(i=1,j=0)
    U[0, 4] = 1  # i=0, j=0
    U[2, 4] = 1  # i=1, j=0
    # V: b11=(j=1,k=1)
    V[3, 4] = 1  # j=1, k=1
    # W: c00(k=0,i=0)=-1, c01(k=1,i=0)=+1
    W[0, 4] = -1  # k=0, i=0
    W[2, 4] = 1   # k=1, i=0
    
    # r=5: M6 = (a11-a10)(b00+b10)
    # U: a11=(i=1,j=1), a10=(i=1,j=0)
    U[3, 5] = 1   # i=1, j=1
    U[2, 5] = -1  # i=1, j=0
    # V: b00=(j=0,k=0), b10=(j=1,k=0)
    V[0, 5] = 1  # j=0, k=0
    V[2, 5] = 1  # j=1, k=0
    # W: c11(k=1,i=1)=+1
    W[3, 5] = 1   # k=1, i=1
    
    # r=6: M7 = (a01-a11)(b01+b11)
    # U: a01=(i=0,j=1), a11=(i=1,j=1)
    U[1, 6] = 1   # i=0, j=1
    U[3, 6] = -1  # i=1, j=1
    # V: b01=(j=0,k=1), b11=(j=1,k=1)
    V[1, 6] = 1  # j=0, k=1
    V[3, 6] = 1  # j=1, k=1
    # W: c00(k=0,i=0)=+1
    W[0, 6] = 1   # k=0, i=0
    
    return U, V, W


def _verify_strassen():
    """Verify the Strassen 2x2x2 decomposition."""
    U_s, V_s, W_s = _strassen_2x2x2()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r] for r in range(7))
                if abs(s - 1) > 1e-10:
                    return False
    return True


def _build_2x4x2():
    """Build rank-14 decomposition for 2x4x2 using two 2x2x2 Strassen blocks.
    
    Split j into {0,1} and {2,3}. Each block is a 2x2x2 multiplication.
    U shape: (8, 14), V shape: (8, 14), W shape: (4, 14)
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U1 = np.zeros((8, 14), dtype=float)
    V1 = np.zeros((8, 14), dtype=float)
    W1 = np.zeros((4, 14), dtype=float)
    
    # Block 1: j in {0,1}, columns 0-6
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U1[i*4 + jl, r] = U_s[i*2 + jl, r]
        for jl in range(2):
            for k in range(2):
                V1[jl*2 + k, r] = V_s[jl*2 + k, r]
        for k in range(2):
            for i in range(2):
                W1[k*2 + i, r] = W_s[k*2 + i, r]
    
    # Block 2: j in {2,3}, columns 7-13
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U1[i*4 + jl + 2, r + 7] = U_s[i*2 + jl, r]
        for jl in range(2):
            for k in range(2):
                V1[(jl+2)*2 + k, r + 7] = V_s[jl*2 + k, r]
        for k in range(2):
            for i in range(2):
                W1[k*2 + i, r + 7] = W_s[k*2 + i, r]
    
    return U1, V1, W1


def _build_2x4x4():
    """Build rank-28 decomposition for 2x4x4 using two 2x4x2 blocks.
    
    Split k into {0,1} and {2,3}. Each block is a 2x4x2 multiplication.
    U shape: (8, 28), V shape: (16, 28), W shape: (8, 28)
    """
    U1, V1, W1 = _build_2x4x2()
    U2 = np.zeros((8, 28), dtype=float)
    V2 = np.zeros((16, 28), dtype=float)
    W2 = np.zeros((8, 28), dtype=float)
    
    # Block 1: k in {0,1}, columns 0-13
    for r in range(14):
        U2[:, r] = U1[:, r]
        for j in range(4):
            for k in range(2):
                V2[j*4 + k, r] = V1[j*2 + k, r]
        for k in range(2):
            for i in range(2):
                W2[k*2 + i, r] = W1[k*2 + i, r]
    
    # Block 2: k in {2,3}, columns 14-27
    for r in range(14):
        U2[:, r + 14] = U1[:, r]
        for j in range(4):
            for k in range(2):
                V2[j*4 + k + 2, r + 14] = V1[j*2 + k, r]
        for k in range(2):
            for i in range(2):
                W2[k*2 + i + 4, r + 14] = W1[k*2 + i, r]
    
    return U2, V2, W2


def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Verify Strassen base case
    if not _verify_strassen():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 2x4x4 (rank 28) and 2x4x2 (rank 14)
    U2, V2, W2 = _build_2x4x4()
    U1, V1, W1 = _build_2x4x2()
    
    # Combine: 2x4x6 = 2x4x4 (k=0..3) + 2x4x2 (k=4..5)
    total_rank = 28 + 14  # 42
    U_total = np.zeros((n * m, total_rank), dtype=float)
    V_total = np.zeros((m * p, total_rank), dtype=float)
    W_total = np.zeros((p * n, total_rank), dtype=float)
    
    # 2x4x4 part: k in {0,1,2,3}
    U_total[:, :28] = U2
    for j in range(4):
        for k in range(4):
            V_total[j * 6 + k, :28] = V2[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W_total[k * 2 + i, :28] = W2[k * 2 + i, :]
    
    # 2x4x2 part: k in {4,5}
    U_total[:, 28:42] = U1
    for j in range(4):
        for k in range(2):
            V_total[j * 6 + k + 4, 28:42] = V1[j * 2 + k, :]
    for k in range(2):
        for i in range(2):
            W_total[k * 2 + i + 8, 28:42] = W1[k * 2 + i, :]
    
    # Verify full decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_total[i*m+j, r] * V_total[j*p+k, r] * W_total[k*n+i, r] for r in range(total_rank))
                if abs(s - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
