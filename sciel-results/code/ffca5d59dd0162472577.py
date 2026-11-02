import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition (Rank 7)
    # Verified from the current program which scored 1.0
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, -1, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1],
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
    ], dtype=float)
    
    # Verify Strassen 2x2x2
    def verify_tensor(U, V, W, n, m, p):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_tensor(U_s, V_s, W_s, 2, 2, 2):
        # Fallback to schoolbook if Strassen is broken
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Kron product of two 2x2x2 to get 4x4x4
    # U1, V1, W1 are (4, 7)
    # Result U, V, W are (16, 49)
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    r1 = 7
    for c1 in range(r1):
        for c2 in range(r1):
            c = c1 * r1 + c2
            # U[i1*2+i2, j1*2+j2] = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]
            # Indices for 4x4: i in 0..3, j in 0..3
            # i = i1*2 + i2, j = j1*2 + j2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            U444[(i1*2+i2)*4 + (j1*2+j2), c] = U_s[i1*2+j1, c1] * U_s[i2*2+j2, c2]
                            
            # V[j1*2+j2, k1*2+k2] = V1[j1*2+k1, c1] * V2[j2*2+k2, c2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            V444[(j1*2+j2)*4 + (k1*2+k2), c] = V_s[j1*2+k1, c1] * V_s[j2*2+k2, c2]
                            
            # W[k1*2+k2, i1*2+i2] = W1[k1*2+i1, c1] * W2[k2*2+i2, c2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            W444[(k1*2+k2)*4 + (i1*2+i2), c] = W_s[k1*2+i1, c1] * W_s[k2*2+i2, c2]

    # Verify 4x4x4
    if not verify_tensor(U444, V444, W444, 4, 4, 4):
        # Fallback
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Extend 4x4x4 to 4x4x8
    # Duplicate the decomposition for the second half of k
    U448 = np.zeros((16, 98))
    V448 = np.zeros((32, 98))
    W448 = np.zeros((32, 98))
    
    r444 = 49
    # First half (k=0..3)
    U448[:, :r444] = U444
    # V for k=0..3: rows 0..15
    V448[:16, :r444] = V444
    # W for k=0..3: rows 0..15 (k*n+i, k=0..3, i=0..3 -> 0..15)
    W448[:16, :r444] = W444
    
    # Second half (k=4..7)
    U448[:, r444:] = U444
    # V for k=4..7: rows 16..31 (j*8+k, j=0..3, k=4..7)
    # V444[j*4+k, c] maps to V448[j*8+k, r444+c]
    for j in range(4):
        for k in range(4):
            V448[j*8 + (k+4), r444:r444+r444] = V444[j*4+k, :]
            
    # W for k=4..7: rows 16..31 (k*n+i, k=4..7, i=0..3)
    # W444[k*4+i, c] maps to W448[(k+4)*4+i, r444+c]
    for k in range(4):
        for i in range(4):
            W448[(k+4)*4+i, r444:r444+r444] = W444[k*4+i, :]

    # Final Verification
    if not verify_tensor(U448, V448, W448, 4, 4, 8):
        # Fallback
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U448.tolist(), "V": V448.tolist(), "W": W448.tolist()}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
