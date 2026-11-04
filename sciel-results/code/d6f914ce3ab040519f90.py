import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def verify_numpy(U_arr, V_arr, W_arr, tol=1e-9):
        rank = U_arr.shape[1]
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=tol)

    def check_solution(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        return verify_numpy(U_arr, V_arr, W_arr)

    def strassen_decomp():
        """
        Construct rank-92 decomposition using Strassen on 8 disjoint 2x2x2 blocks.
        
        i-pairs: (0,1), (2,3)
        j-pairs: (0,1), (2,3), leftover j=4
        k-pairs: (0,1), (2,3), leftover k=4
        
        8 Strassen blocks: for each (ii, jj, kk) in pairs x pairs x pairs
        36 schoolbook terms: all (i,j,k) with j=4 or k=4 (not in any Strassen block)
        """
        U = np.zeros((n * m, 92))
        V = np.zeros((m * p, 92))
        W = np.zeros((p * n, 92))
        
        i_pairs = [(0, 1), (2, 3)]
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3)]
        
        col = 0
        for (i1, i2) in i_pairs:
            for (j1, j2) in j_pairs:
                for (k1, k2) in k_pairs:
                    # Strassen M1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2])
                    U[i1*m+j1, col] += 1
                    U[i2*m+j2, col] += 1
                    V[j1*p+k1, col] += 1
                    V[j2*p+k2, col] += 1
                    W[k1*n+i1, col] += 1
                    W[k2*n+i1, col] += 1
                    W[k1*n+i2, col] += 1
                    W[k2*n+i2, col] += 1
                    col += 1
                    
                    # Strassen M2 = (A[i2,j1]+A[i2,j2])*B[j1,k1]
                    U[i2*m+j1, col] += 1
                    U[i2*m+j2, col] += 1
                    V[j1*p+k1, col] += 1
                    W[k1*n+i2, col] += 1
                    col += 1
                    
                    # Strassen M3 = A[i1,j1]*(B[j1,k2]-B[j2,k2])
                    U[i1*m+j1, col] += 1
                    V[j1*p+k2, col] += 1
                    V[j2*p+k2, col] += -1
                    W[k2*n+i1, col] += 1
                    col += 1
                    
                    # Strassen M4 = A[i2,j2]*(B[j2,k1]-B[j1,k1])
                    U[i2*m+j2, col] += 1
                    V[j2*p+k1, col] += 1
                    V[j1*p+k1, col] += -1
                    W[k1*n+i2, col] += 1
                    col += 1
                    
                    # Strassen M5 = (A[i1,j1]+A[i1,j2])*B[j2,k2]
                    U[i1*m+j1, col] += 1
                    U[i1*m+j2, col] += 1
                    V[j2*p+k2, col] += 1
                    W[k2*n+i1, col] += 1
                    col += 1
                    
                    # Strassen M6 = (A[i2,j1]-A[i1,j1])*(B[j1,k1]+B[j1,k2])
                    U[i2*m+j1, col] += 1
                    U[i1*m+j1, col] += -1
                    V[j1*p+k1, col] += 1
                    V[j1*p+k2, col] += 1
                    W[k1*n+i2, col] += 1
                    W[k2*n+i2, col] += 1
                    col += 1
                    
                    # Strassen M7 = (A[i1,j2]-A[i2,j2])*(B[j2,k1]+B[j2,k2])
                    U[i1*m+j2, col] += 1
                    U[i2*m+j2, col] += -1
                    V[j2*p+k1, col] += 1
                    V[j2*p+k2, col] += 1
                    W[k1*n+i1, col] += 1
                    W[k2*n+i1, col] += 1
                    col += 1
        
        # Schoolbook for remaining terms (j=4 or k=4, not in Strassen blocks)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip if this (i,j,k) is in a Strassen block
                    in_block = False
                    for (i1, i2) in i_pairs:
                        if i in (i1, i2):
                            for (j1, j2) in j_pairs:
                                if j in (j1, j2):
                                    for (k1, k2) in k_pairs:
                                        if k in (k1, k2):
                                            in_block = True
                    if not in_block:
                        U[i*m+j, col] += 1
                        V[j*p+k, col] += 1
                        W[k*n+i, col] += 1
                        col += 1
        
        assert col == 92, f"Expected rank 92, got {col}"
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if check_solution(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return strassen_decomp()
