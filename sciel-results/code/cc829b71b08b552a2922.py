import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Term 1
            U[0*m + j0, col] = 1
            U[1*m + j0+1, col] = 1
            V[j0*p + k0, col] = 1
            V[(j0+1)*p + k0+1, col] = 1
            W[k0*n + 0, col] = 1
            W[(k0+1)*n + 1, col] = 1
            col += 1
            
            # Term 2
            U[1*m + j0, col] = 1
            U[1*m + j0+1, col] = 1
            V[j0*p + k0, col] = 1
            W[k0*n + 1, col] = 1
            W[(k0+1)*n + 1, col] = -1
            col += 1
            
            # Term 3
            U[0*m + j0, col] = 1
            V[j0*p + k0+1, col] = 1
            V[(j0+1)*p + k0+1, col] = -1
            W[(k0+1)*n + 0, col] = 1
            W[(k0+1)*n + 1, col] = 1
            col += 1
            
            # Term 4
            U[1*m + j0+1, col] = 1
            V[(j0+1)*p + k0, col] = 1
            V[j0*p + k0, col] = -1
            W[k0*n + 0, col] = 1
            W[k0*n + 1, col] = 1
            col += 1
            
            # Term 5
            U[0*m + j0, col] = 1
            U[0*m + j0+1, col] = 1
            V[(j0+1)*p + k0+1, col] = 1
            W[k0*n + 0, col] = -1
            W[(k0+1)*n + 0, col] = 1
            col += 1
            
            # Term 6
            U[1*m + j0, col] = 1
            U[0*m + j0, col] = -1
            V[j0*p + k0, col] = 1
            V[j0*p + k0+1, col] = 1
            W[(k0+1)*n + 1, col] = 1
            col += 1
            
            # Term 7
            U[0*m + j0+1, col] = 1
            U[1*m + j0+1, col] = -1
            V[(j0+1)*p + k0, col] = 1
            V[(j0+1)*p + k0+1, col] = 1
            W[k0*n + 0, col] = 1
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
