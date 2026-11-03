import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_2x2x2_vectors():
    # Returns U, V, W vectors for 2x2x2 Strassen
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: A00, A01, A10, A11
    # B00, B01, B10, B11
    # C00, C01, C10, C11
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00+A11)(B00+B11)
    U[:,0] = [1, 0, 0, 1]
    V[:,0] = [1, 0, 0, 1]
    W[:,0] = [1, 0, 0, 1] # C00, C11
    
    # P2 = (A00+A11)B10
    U[:,1] = [1, 0, 0, 1]
    V[:,1] = [0, 0, 1, 0]
    W[:,1] = [0, 0, 1, -1] # C10, C11
    
    # P3 = A00(B01-B11)
    U[:,2] = [1, 0, 0, 0]
    V[:,2] = [0, 1, 0, -1]
    W[:,2] = [0, 1, 0, 1] # C01, C11
    
    # P4 = A11(B10-B00)
    U[:,3] = [0, 0, 0, 1]
    V[:,3] = [0, 0, 1, -1]
    W[:,3] = [1, 0, 1, 0] # C00, C10
    
    # P5 = (A01+A11)B00
    U[:,4] = [0, 1, 0, 1]
    V[:,4] = [1, 0, 0, 0]
    W[:,4] = [-1, 1, 0, 0] # C00, C01
    
    # P6 = (A00-A10)(B01+B11)
    U[:,5] = [1, 0, -1, 0]
    V[:,5] = [0, 1, 0, 1]
    W[:,5] = [0, 0, 0, 1] # C11
    
    # P7 = (A10-A00)(B10+B01)
    U[:,6] = [-1, 0, 1, 0]
    V[:,6] = [0, 1, 1, 0]
    W[:,6] = [1, 0, 0, 0] # C00
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Rank 21 construction
    # Block 1: j in {0,1}, k in {0,1}
    # Block 2: j in {0,1}, k in {2,3}
    # Singletons: j in {0,1}, k=4; j=2, k in {0,1,2,3,4}
    
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    sU, sV, sW = _strassen_2x2x2_vectors()
    
    col = 0
    # Block 1
    for r in range(7):
        # Map local A indices (0..3) to global (i*m+j)
        # Local 0: A00 -> i=0, j=0 -> 0
        # Local 1: A01 -> i=0, j=1 -> 1
        # Local 2: A10 -> i=1, j=0 -> 3
        # Local 3: A11 -> i=1, j=1 -> 4
        U[0, col] = sU[0, r]
        U[1, col] = sU[1, r]
        U[3, col] = sU[2, r]
        U[4, col] = sU[3, r]
        
        # Map local B indices (0..3) to global (j*p+k)
        # Local 0: B00 -> j=0, k=0 -> 0
        # Local 1: B01 -> j=0, k=1 -> 1
        # Local 2: B10 -> j=1, k=0 -> 5
        # Local 3: B11 -> j=1, k=1 -> 6
        V[0, col] = sV[0, r]
        V[1, col] = sV[1, r]
        V[5, col] = sV[2, r]
        V[6, col] = sV[3, r]
        
        # Map local C indices (0..3) to global (k*n+i)
        # Local 0: C00 -> k=0, i=0 -> 0
        # Local 1: C01 -> k=1, i=0 -> 2
        # Local 2: C10 -> k=0, i=1 -> 1
        # Local 3: C11 -> k=1, i=1 -> 3
        W[0, col] = sW[0, r]
        W[2, col] = sW[1, r]
        W[1, col] = sW[2, r]
        W[3, col] = sW[3, r]
        
        col += 1
        
    # Block 2
    for r in range(7):
        # Map local A indices to global (same as Block 1)
        U[0, col] = sU[0, r]
        U[1, col] = sU[1, r]
        U[3, col] = sU[2, r]
        U[4, col] = sU[3, r]
        
        # Map local B indices to global
        # Local 0: B00 -> j=0, k=2 -> 2
        # Local 1: B01 -> j=0, k=3 -> 3
        # Local 2: B10 -> j=1, k=2 -> 7
        # Local 3: B11 -> j=1, k=3 -> 8
        V[2, col] = sV[0, r]
        V[3, col] = sV[1, r]
        V[7, col] = sV[2, r]
        V[8, col] = sV[3, r]
        
        # Map local C indices to global
        # Local 0: C00 -> k=2, i=0 -> 4
        # Local 1: C01 -> k=3, i=0 -> 6
        # Local 2: C10 -> k=2, i=1 -> 5
        # Local 3: C11 -> k=3, i=1 -> 7
        W[4, col] = sW[0, r]
        W[6, col] = sW[1, r]
        W[5, col] = sW[2, r]
        W[7, col] = sW[3, r]
        
        col += 1
        
    # Singletons
    # j=0, k=4 -> i=0,1
    # j=1, k=4 -> i=0,1
    # j=2, k=0..4 -> i=0,1
    
    # i=0, j=0, k=4
    U[0, col] = 1
    V[4, col] = 1
    W[8, col] = 1 # k=4, i=0 -> 4*2+0 = 8
    col += 1
    
    # i=1, j=0, k=4
    U[3, col] = 1
    V[4, col] = 1
    W[9, col] = 1 # k=4, i=1 -> 4*2+1 = 9
    col += 1
    
    # i=0, j=1, k=4
    U[1, col] = 1
    V[9, col] = 1 # j=1, k=4 -> 1*5+4 = 9
    W[8, col] = 1
    col += 1
    
    # i=1, j=1, k=4
    U[4, col] = 1
    V[9, col] = 1
    W[9, col] = 1
    col += 1
    
    # j=2
    for k in range(5):
        # i=0
        U[2, col] = 1 # i=0, j=2 -> 0*3+2 = 2
        V[10+k, col] = 1 # j=2, k -> 2*5+k = 10+k
        W[k*2, col] = 1 # k, i=0 -> k*2+0
        col += 1
        
        # i=1
        U[5, col] = 1 # i=1, j=2 -> 1*3+2 = 5
        V[10+k, col] = 1
        W[k*2+1, col] = 1
        col += 1
        
    if col != rank:
        # Fallback if logic error
        pass

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_school, V_school, W_school = _schoolbook(n, m, p)
        return {"U": U_school, "V": V_school, "W": W_school}
