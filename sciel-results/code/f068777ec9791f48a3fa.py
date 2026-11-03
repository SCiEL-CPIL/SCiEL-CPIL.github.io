"""Rewritten program for 3x3x7 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, r):
        """Verify the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = sum(U[a][ci] * V[b][ci] * W[c][ci] for ci in range(r))
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check off-diagonal
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    if i_a != i_c or k_b != k_c:
                        # This is an off-diagonal entry that should be 0
                        val = sum(U[a][ci] * V[b][ci] * W[c][ci] for ci in range(r))
                        if abs(val) > 1e-6:
                            return False
        return True
    
    def try_rank_structured(r):
        """Try to find a structured decomposition of given rank."""
        # Use the fact that we can have sparse U, V, W
        # Each rank-1 term can have multiple nonzero entries
        
        # Strategy: for the 3x3x7 case, use the structure
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # For each i (3 values), we have a 1x3 row of A times 3x7 B
        # 
        # Let's try: for each (i,j) pair, create terms that handle all k
        # But that's 9*7 = 63 again
        #
        # Better: use linear combinations
        # For fixed i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:]
        # This is a linear combination of 3 rows of B, weighted by A[i,:]
        #
        # We can use the fact that B is 3x7, so its rows span a 3-dim space
        # 
        # Let's try rank = 21 with the following structure:
        # For each k (7 values) and each "Strassen-like" combination for the 3x3 part
        #
        # Actually, let me try a direct construction:
        # Use 3 terms per (i,k) but share across i
        #
        # For now, let's do a random search with alternating minimization
        
        np.random.seed(123 + r)
        
        def rand_half_int(rows, cols, scale=4):
            return np.random.randint(-scale, scale+1, size=(rows, cols)) * 0.5
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        for attempt in range(30):
            if time.time() > deadline:
                break
                
            U = rand_half_int(n*m, r, scale=3)
            V = rand_half_int(m*p, r, scale=3)
            W = rand_half_int(p*n, r, scale=3)
            
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                T_flat = T.reshape(n*m, m*p*p*n)
                M = np.zeros((r, m*p*p*n))
                for ci in range(r):
                    for b in range(m*p):
                        for c in range(p*n):
                            M[ci, b*p*n+c] = V[b,ci] * W[c,ci]
                try:
                    U = np.linalg.lstsq(M.T, T_flat.T, rcond=None)[0].T
                except:
                    pass
                U = np.round(U*2)/2
                
                # Solve for V
                M2 = np.zeros((r, n*m*p*n))
                for ci in range(r):
                    for a in range(n*m):
                        for c in range(p*n):
                            M2[ci, a*p*n+c] = U[a,ci] * W[c,ci]
                T_flat2 = T.reshape(m*p, n*m*p*n)
                try:
                    V = np.linalg.lstsq(M2.T, T_flat2.T, rcond=None)[0].T
                except:
                    pass
                V = np.round(V*2)/2
                
                # Solve for W
                M3 = np.zeros((r, n*m*m*p))
                for ci in range(r):
                    for a in range(n*m):
                        for b in range(m*p):
                            M3[ci, a*m*p+b] = U[a,ci] * V[b,ci]
                T_flat3 = T.reshape(p*n, n*m*m*p)
                try:
                    W = np.linalg.lstsq(M3.T, T_flat3.T, rcond=None)[0].T
                except:
                    pass
                W = np.round(W*2)/2
            
            # Check
            U_list = [[U[a,ci] for ci in range(r)] for a in range(n*m)]
            V_list = [[V[b,ci] for ci in range(r)] for b in range(m*p)]
            W_list = [[W[c,ci] for ci in range(r)] for c in range(p*n)]
            
            if verify(U_list, V_list, W_list, r):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from lower bound
    for r in range(21, 25):
        if time.time() > deadline:
            break
        result = try_rank_structured(r)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
