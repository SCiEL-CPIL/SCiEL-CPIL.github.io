import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Target rank to try
    target_rank = 20
    
    # The tensor T is defined by T[i, j, k] = 1 if j == k else 0
    # Dimensions: i in [0, n), j in [0, m), k in [0, p)
    # Note: The problem statement says W[k*n+i, r].
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We want to find U, V, W such that the above holds.
    # This is equivalent to:
    # For each i, k: sum_j T[i, j, k] * V[j*p+k, c] ... no.
    # Let's flatten T.
    # T_flat[index] where index corresponds to (i, j, k).
    # But the sum is over c.
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = T[i, j, k]
    
    # We can fix U and V and solve for W.
    # The equation is linear in W.
    # Let A be a matrix of size (n*p, p*n*rank)? No.
    # For a fixed (i, k), the equation is:
    # sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = T[i, j, k]
    # This is not linear in W because T[i, j, k] depends on j.
    # Wait, T[i, j, k] is 1 if j==k.
    # So for fixed i, k:
    # If j == k: sum_c U[i*m+k, c] V[k*p+k, c] W[k*n+i, c] = 1
    # If j != k: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 0
    
    # This is a system of equations for W.
    # Let's build the matrix M such that M * W_flat = T_flat.
    # W_flat has size (p*n*rank).
    # For each i, j, k, we have an equation.
    # Coefficients of W[k*n+i, c] are U[i*m+j, c] * V[j*p+k, c].
    # So M[row(i,j,k), col(k*n+i, c)] = U[i*m+j, c] * V[j*p+k, c].
    # T_flat[row(i,j,k)] = 1 if j==k else 0.
    
    # We can solve this least squares problem.
    # If the solution W gives T_flat exactly, we are done.
    
    while time.time() < deadline:
        rank = target_rank
        # Random U, V
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        
        # Build M
        # Number of equations: n * m * p
        num_eq = n * m * p
        num_w = p * n * rank
        M = np.zeros((num_eq, num_w))
        T_flat = np.zeros(num_eq)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_flat[idx] = 1.0 if j == k else 0.0
                    for c in range(rank):
                        # W index is k*n + i, c
                        w_idx = (k * n + i) * rank + c
                        M[idx, w_idx] = U[i * m + j, c] * V[j * p + k, c]
                    idx += 1
        
        # Solve M * W_flat = T_flat
        try:
            W_flat, residuals, rank_M, sv = np.linalg.lstsq(M, T_flat, rcond=None)
            # Check if the solution is exact
            # If residuals is 0 (or very small), it's a solution.
            # But lstsq minimizes error. We need exact.
            # Check if M @ W_flat == T_flat
            if np.allclose(M @ W_flat, T_flat, atol=1e-9):
                # We found a solution!
                W = W_flat.reshape(p * n, rank)
                # Return as lists
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except:
            pass
    
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}
