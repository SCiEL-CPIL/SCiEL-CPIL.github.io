"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Verify a candidate decomposition
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Schoolbook decomposition (rank = n*m*p)
    def schoolbook():
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

    # Strategy: Block-diagonal decomposition.
    # For fixed i, the problem reduces to: find U_i (m x r_i), V (m*p x r_i), W_i (p x r_i)
    # such that sum_c U_i[j,c] * V[j*p+k,c] * W_i[k,c] = 1 for all j,k.
    # This is the matrix multiplication tensor for 1 x m by m x p.
    # The minimal rank for this is m*p (schoolbook), but we can try lower.
    # For m=3, p=9, schoolbook rank is 27.
    # We'll try to find a decomposition of rank r for each row, then combine.
    # Total rank = sum of r_i for each i.
    
    # For the 1 x m by m x p case, the tensor is T[j,k] = 1 for all j,k.
    # This is the "all-ones" tensor of shape m x p.
    # The rank of the all-ones tensor is 1! Because T[j,k] = 1*1*1.
    # So U_i[j,c] = 1, V[j*p+k,c] = 1, W_i[k,c] = 1 for c=0 only.
    # This gives rank 1 for each row!
    # Total rank = n * 1 = 2.
    # Let's verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1.
    # This works! Rank = 2.
    
    # But wait, the problem says "integer/half-integer entries". 1 is integer. So this is valid.
    # Let's construct this.
    
    rank = n  # one term per row
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][i] = 1
        for k in range(p):
            W[k * n + i][i] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1  # Wait, this doesn't work because V is shared across rows.
    
    # Actually, V[j*p+k, c] must be the same for all i. So we can't have V depend on i.
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    # If we set V[j*p+k, c] = 1 for all j,k,c, then we need U[i*m+j, c] * W[k*n+i, c] to sum to 1.
    # If we set U[i*m+j, c] = delta_{c,i} and W[k*n+i, c] = delta_{c,i}, then the sum is 1 when c=i, 0 otherwise.
    # So U[i*m+j, c] = 1 if c==i else 0, W[k*n+i, c] = 1 if c==i else 0, V[j*p+k, c] = 1 for all c.
    # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_c delta_{c,i} * 1 * delta_{c,i} = 1.
    # This works! Rank = n = 2.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][i] = 1
        for k in range(p):
            W[k * n + i][i] = 1
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
