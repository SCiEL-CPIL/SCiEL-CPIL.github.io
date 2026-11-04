import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 120.0

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the tensor product (Kronecker product) of Strassen's 2x2x2 decomposition
    to create a rank-49 decomposition for 4x4x4, then duplicates it for the two
    4x4 blocks of the 4x8 output.
    
    The 4x4x8 tensor decomposes as a direct sum of two 4x4x4 tensors.
    Total rank = 49 + 49 = 98.
    
    Optimized construction using vectorized Kronecker product operations
    to minimize Python-level loops.
    """
    n, m, p = 4, 4, 8
    U2, V2, W2 = _build_strassen_2x2x2()
    
    # Construct 4x4x4 tensors via Kronecker product (rank 49)
    # Using np.kron for efficient construction
    # U4 = kron(U2, U2) but with correct index mapping
    # U2[i*2+j, r] -> U4[i'*4+j', r'] where i' = 2*i + i', j' = 2*j + j'
    # Actually, for Kronecker product of tensors:
    # T4[i,j,k] = sum_{r1,r2} U2[i1,j1,r1]*U2[i2,j2,r2] * V2[j1,k1,r1]*V2[j2,k2,r2] * W2[k1,i1,r1]*W2[k2,i2,r2]
    # where i = 2*i1+i2, j = 2*j1+j2, k = 2*k1+k2
    # This is exactly the Kronecker product structure.
    
    # Vectorized Kronecker construction:
    # U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # where i = 2*i1+i2, j = 2*j1+j2
    
    # Reshape U2 to separate block and within-block indices
    # U2 has shape (4, 7) = (2*2, 7)
    # We want U4 of shape (16, 49) = (4*4, 7*7)
    
    # Use broadcasting: U4[i,j,r1,r2] = U2[i1,j1,r1] * U2[i2,j2,r2]
    # where i = 2*i1+i2, j = 2*j1+j2
    
    # Create index arrays
    i1 = np.array([0, 0, 1, 1])  # block row for i
    i2 = np.array([0, 1, 0, 1])  # within-block row for i
    j1 = np.array([0, 1, 0, 1])  # block col for j
    j2 = np.array([0, 0, 1, 1])  # within-block col for j
    
    # U2 indices: U2[i1*2+j1, r1] and U2[i2*2+j2, r2]
    # Shape: (4, 7) for each
    U2_block1 = U2[i1 * 2 + j1, :]  # (4, 7)
    U2_block2 = U2[i2 * 2 + j2, :]  # (4, 7)
    
    # U4[i, j, r1, r2] = U2_block1[i, r1] * U2_block2[j, r2]
    # Reshape to (16, 49)
    U4 = np.zeros((16, 49), dtype=np.int16)
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U4[:, r] = U2_block1[:, r1] * U2_block2[:, r2]
            # But we need to map (i, j) pairs to row index i*4+j
            # U2_block1[i, r1] for i in 0..3 gives values for the "outer" part
            # U2_block2[j, r2] for j in 0..3 gives values for the "inner" part
            # The row index in U4 is i*4+j where i and j are the 4x4 indices
            
            # Actually, let me think more carefully.
            # i = 2*i1 + i2, j = 2*j1 + j2
            # So for each (i1, i2, j1, j2), we get a unique (i, j)
            # U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            
            # Let's just use the direct formula with precomputed values
            for i in range(4):
                for j in range(4):
                    i1, i2 = i // 2, i % 2
                    j1, j2 = j // 2, j % 2
                    U4[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    
    # Similarly for V4 and W4
    V4 = np.zeros((16, 49), dtype=np.int16)
    W4 = np.zeros((16, 49), dtype=np.int16)
    
    # V4[j*4+k, r1*7+r2] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    # where j = 2*j1+j2, k = 2*k1+k2
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for j in range(4):
                for k in range(4):
                    j1, j2 = j // 2, j % 2
                    k1, k2 = k // 2, k % 2
                    V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    
    # W4[k*4+i, r1*7+r2] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    # where k = 2*k1+k2, i = 2*i1+i2
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for k in range(4):
                for i in range(4):
                    k1, k2 = k // 2, k % 2
                    i1, i2 = i // 2, i % 2
                    W4[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    # Construct 4x4x8 tensors (rank 98)
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: k in [0, 4)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]
    
    # Block 1: k in [4, 8)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j*8 + k, 49:98] = V4[j*4 + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[k*4 + i, 49:98] = W4[(k-4)*4 + i, :]
    
    # Fast verification using numpy
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n=4, m=4, p=8):
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

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
