import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Build the target tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a lower rank decomposition
    # Start from schoolbook and try to reduce
    
    # Schoolbook decomposition
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Return schoolbook as fallback
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x) for x in row] for row in mat]
    
    def make_result(U, V, W):
        rank = U.shape[1]
        return {
            "U": [[int(round(U[i,j])) if abs(U[i,j] - round(U[i,j])) < 1e-9 else 2*U[i,j] for j in range(rank)] for i in range(U.shape[0])],
            "V": [[int(round(V[i,j])) if abs(V[i,j] - round(V[i,j])) < 1e-9 else 2*V[i,j] for j in range(rank)] for i in range(V.shape[0])],
            "W": [[int(round(W[i,j])) if abs(W[i,j] - round(W[i,j])) < 1e-9 else 2*W[i,j] for j in range(rank)] for i in range(W.shape[0])]
        }
    
    # Try rank 47
    for target_rank in range(schoolbook_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        
        # Try random initialization
        for trial in range(100):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Check if the tensor matches
            # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # This is expensive for large tensors, but our tensor is small
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            
            if err < 1e-9:
                # Found a valid decomposition!
                # Try to clean up to integer/half-integer
                return make_result(U, V, W)
        
        # Try a more structured approach: start from schoolbook and zero out one column
        # Then adjust the remaining columns
        if target_rank == schoolbook_rank - 1:
            # Try to merge two columns
            for c1 in range(schoolbook_rank):
                for c2 in range(c1+1, schoolbook_rank):
                    if time.time() > deadline:
                        break
                    
                    # Try to see if we can merge columns c1 and c2
                    # by setting U[:,c2] = 0, V[:,c2] = 0, W[:,c2] = 0
                    # and adjusting U[:,c1], V[:,c1], W[:,c1]
                    # This is a complex optimization, so let's skip for now
                    pass
    
    # Return schoolbook
    return make_result(U_sb, V_sb, W_sb)
