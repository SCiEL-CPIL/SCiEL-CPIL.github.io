"""Matrix multiplication 5x5x5: hybrid Strassen-border decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[3,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[3,6]=-1
    return U, V, W

def _construct_4x4_strassen():
    U4, V4, W4 = np.zeros((16, 49)), np.zeros((16, 49)), np.zeros((16, 49))
    U2, V2, W2 = _strassen_2x2_terms()
    
    def add_block(U4, V4, W4, a_blks, b_blks, c_blks, off):
        for t in range(7):
            for (r,c,s) in a_blks:
                for i in range(2):
                    for j in range(2):
                        if U2[i*2+j, t] != 0:
                            U4[r+i, c+j, off+t] += s * U2[i*2+j, t]
            for (r,c,s) in b_blks:
                for i in range(2):
                    for j in range(2):
                        if V2[i*2+j, t] != 0:
                            V4[r+i, c+j, off+t] += s * V2[i*2+j, t]
            for (r,c,s) in c_blks:
                for i in range(2):
                    for j in range(2):
                        if W2[i*2+j, t] != 0:
                            W4[r+i, c+j, off+t] += s * W2[i*2+j, t]

    add_block(U4, V4, W4, [(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], 0)
    add_block(U4, V4, W4, [(2,0,1),(2,2,1)], [(0,0,1)], [(2,0,1)], 7)
    add_block(U4, V4, W4, [(0,0,1)], [(2,0,1),(2,2,-1)], [(0,2,1),(2,2,1)], 14)
    add_block(U4, V4, W4, [(2,2,1)], [(0,2,1),(0,0,-1)], [(0,0,1),(2,0,1)], 21)
    add_block(U4, V4, W4, [(0,0,1),(0,2,1)], [(2,2,1)], [(0,0,-1),(0,2,1)], 28)
    add_block(U4, V4, W4, [(2,0,1),(0,0,-1)], [(0,0,1),(2,2,1)], [(2,2,1)], 35)
    add_block(U4, V4, W4, [(0,2,1),(2,2,-1)], [(2,0,1),(2,2,1)], [(0,0,1),(2,2,-1)], 42)
    return U4, V4, W4

def _construct_5x5_hybrid():
    if time.time() - START_TIME > TIME_LIMIT:
        return None
    U4, V4, W4 = _construct_4x4_strassen()
    U5, V5, W5 = np.zeros((25, 49)), np.zeros((25, 49)), np.zeros((25, 49))
    U5[:16,:], V5[:16,:], W5[:16,:] = U4, V4, W4
    
    col = 49
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i == 4 or j == 4 or k == 4:
                    U5[i*5+j, col] = 1
                    V5[j*5+k, col] = 1
                    W5[k*5+i, col] = 1
                    col += 1
                    if time.time() - START_TIME > TIME_LIMIT:
                        return None
    return {"U": U5[:,:col], "V": V5[:,:col], "W": W5[:,:col]}

def _verify(U, V, W):
    if time.time() - START_TIME > TIME_LIMIT:
        return False
    n, m, p = 5, 5, 5
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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
        hybrid = _construct_5x5_hybrid()
        if hybrid and _verify(hybrid["U"], hybrid["V"], hybrid["W"]):
            return {
                "U": hybrid["U"].tolist(),
                "V": hybrid["V"].tolist(),
                "W": hybrid["W"].tolist()
            }
    except Exception:
        pass
    return _schoolbook()
# EVOLVE-BLOCK-END
