import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Rank-5 schoolbook decomposition for 4x5x6 matrix multiplication
    # The tensor T has T[i,j,k] = 1 if we multiply A[i,j] * B[j,k] = C[i,k]
    # Actually, the tensor for matrix multiplication M_{n,m,p} has:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor entry T[i, j, k] = 1 (it's a 0-1 tensor)
    # The rank-1 terms are: for each j, u_j has 1 at (i, j) for all i, v_j has 1 at (j, k) for all k, w_j has 1 at (k, i) for all k, i
    
    # Wait, let me reconsider the tensor structure.
    # The matrix multiplication tensor M_{n,m,p} has dimensions (n*m) x (m*p) x (p*n)
    # An entry T[i,j; j,k; k,i] = 1 if the indices match properly
    # More precisely: T_{(i,j),(j',k),(k',i')} = 1 if j=j' and k=k' and i=i'
    # Wait no. The tensor is: sum over j of A[i,j] B[j,k] = C[i,k]
    # So T[i, j, k] = 1 for each triple (i,j,k)
    # In flattened form: U dimension is n*m (index i*m + j), V dimension is m*p (index j*p + k), W dimension is p*n (index k*n + i)
    # The tensor has 1 at position (i*m + j, j*p + k, k*n + i) for all i,j,k
    
    # Rank-1 term for each j:
    # u has 1 at positions (i, j) for all i -> U[(i*m + j)] = 1 for all i
    # v has 1 at positions (j, k) for all k -> V[(j*p + k)] = 1 for all k
    # w has 1 at positions (k, i) for all k, i -> W[(k*n + i)] = 1 for all k, i
    
    # This gives rank = m = 5
    
    U = []
    V = []
    W = []
    
    for j in range(m):
        u_row = np.zeros(n * m)
        v_row = np.zeros(m * p)
        w_row = np.zeros(p * n)
        
        for i in range(n):
            u_row[i * m + j] = 1
        for k in range(p):
            v_row[j * p + k] = 1
        for k in range(p):
            for i in range(n):
                w_row[k * n + i] = 1
        
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)
    
    return {"U": U, "V": V, "W": W}
