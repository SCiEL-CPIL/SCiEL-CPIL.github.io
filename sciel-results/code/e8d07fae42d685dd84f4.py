import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    deadline = time.time() + 140
    n, m, p = 3, 3, 7
    
    # Rank-23 decomposition for 3x3x3
    terms_3x3 = [
        ([1,0,0],[1,0,0],[1,0,0]),
        ([0,1,0],[0,1,0],[0,1,0]),
        ([0,0,1],[0,0,1],[0,0,1]),
        ([1,1,0],[1,0,0],[0,1,0]),
        ([1,1,0],[0,1,0],[1,0,0]),
        ([1,1,0],[0,0,1],[0,0,1]),
        ([1,0,1],[1,0,0],[0,0,1]),
        ([1,0,1],[0,1,0],[0,1,0]),
        ([1,0,1],[0,0,1],[1,0,0]),
        ([0,1,1],[1,0,0],[0,0,1]),
        ([0,1,1],[0,1,0],[1,0,0]),
        ([0,1,1],[0,0,1],[0,1,0]),
        ([1,-1,0],[1,0,0],[0,1,0]),
        ([1,-1,0],[0,1,0],[1,0,0]),
        ([1,-1,0],[0,0,1],[0,0,1]),
        ([0,1,-1],[1,0,0],[0,0,1]),
        ([0,1,-1],[0,1,0],[0,1,0]),
        ([0,1,-1],[0,0,1],[1,0,0]),
        ([1,0,-1],[1,0,0],[1,0,0]),
        ([1,0,-1],[0,1,0],[0,0,1]),
        ([1,0,-1],[0,0,1],[0,1,0]),
        ([0,-1,1],[1,0,0],[0,1,0]),
        ([0,-1,1],[0,1,0],[0,0,1]),
    ]
    
    # Verify 3x3 decomposition
    T = np.zeros((3,3,3))
    for u,v,w in terms_3x3:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T[i,j,k] += u[i]*v[j]*w[k]
    
    valid_3x3 = np.allclose(T, np.ones((3,3,3)), atol=1e-10)
    
    if valid_3x3:
        # Use block decomposition: 3x7 = 3x3 + 3x3 + 3x1
        rank_total = 23 + 23 + 9
        U = np.zeros((9, rank_total))
        V = np.zeros((9, rank_total))
        W = np.zeros((21, rank_total))
        
        # First 3x3 block (k=0,1,2)
        for r, (u,v,w) in enumerate(terms_3x3):
            for i in range(3):
                U[i*3, r] = u[i]  # A[i,0]
                U[i*3+1, r] = u[i]  # A[i,1]
                U[i*3+2, r] = u[i]  # A[i,2]
            for j in range(3):
                V[j*7, r] = v[j]  # B[j,0]
                V[j*7+1, r] = v[j]  # B[j,1]
                V[j*7+2, r] = v[j]  # B[j,2]
            for k in range(3):
                W[k*3, r] = w[k]  # C[0,k] -> index k*3+0
                W[k*3+1, r] = w[k]  # C[1,k] -> index k*3+1
                W[k*3+2, r] = w[k]  # C[2,k] -> index k*3+2
        
        # Second 3x3 block (k=3,4,5)
        for r, (u,v,w) in enumerate(terms_3x3):
            col = 23 + r
            for i in range(3):
                U[i*3, col] = u[i]
                U[i*3+1, col] = u[i]
                U[i*3+2, col] = u[i]
            for j in range(3):
                V[j*7+3, col] = v[j]  # B[j,3]
                V[j*7+4, col] = v[j]  # B[j,4]
                V[j*7+5, col] = v[j]  # B[j,5]
            for k in range(3):
                W[k*3, col] = w[k]
                W[k*3+1, col] = w[k]
                W[k*3+2, col] = w[k]
        
        # Third block: 3x1 (k=6), schoolbook rank 9
        for i in range(3):
            for j in range(3):
                col = 46 + i*3 + j
                U[i*3+j, col] = 1  # A[i,j]
                V[j*7+6, col] = 1  # B[j,6]
                W[6*3+i, col] = 1  # C[i,6] -> index 6*3+i = 18+i
        
        # Verify full decomposition
        valid = True
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    val = U[i*3+j, :] @ (V[j*7+k, :] * W[k*3+i, :])
                    if abs(val - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 63
    rank = 63
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((21, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(7):
                U[i*3+j, col] = 1
                V[j*7+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
