"""Program to find a rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = n * m * p  # 24

    # Schoolbook solution
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

    # Try to find a lower rank decomposition (e.g., 23)
    # This is a heuristic check. If it fails quickly, we return schoolbook.
    start_time = time.time()
    time_limit = 10.0  # seconds
    
    for r in range(rank - 1, 22, -1):
        if time.time() - start_time > time_limit:
            break
        result = _try_find_decomposition(n, m, p, r)
        if result is not None:
            return result

    return {"U": U, "V": V, "W": W}

def _try_find_decomposition(n, m, p, rank):
    """
    Attempt to find a decomposition of the given rank.
    Returns None if not found.
    """
    # Heuristic: Try random initializations for W, then solve for U and V.
    # The equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    # For fixed W, this is linear in U if V is fixed, or vice versa.
    # We can alternate: fix W, solve for U,V; fix U,V, solve for W.
    
    # To keep it fast, we'll try a few random W matrices with small integer entries.
    np.random.seed(42)
    
    for _ in range(10): # Few attempts
        # Initialize W with random integers in {-1, 0, 1}
        W_rand = np.random.choice([-1, 0, 1], size=(p * n, rank))
        
        # Solve for U and V using alternating least squares (ALS)
        U_curr = np.random.randn(n * m, rank)
        V_curr = np.random.randn(m * p, rank)
        
        converged = False
        for it in range(20):
            # Update U: For each (i,j), we need sum_r U[i,j,r] * (sum_k V[j,k,r] W[k,i,r]) = 1
            # This is a linear system for U[i,j,:] if we treat the term in brackets as a vector.
            # Let A_ij_r = sum_k V[j,k,r] W[k,i,r].
            # We need U[i,j,:] . A_ij = 1.
            # This is underdetermined if rank > 1. We can find the minimum norm solution.
            # However, we need exact integer/half-integer solutions.
            # ALS with rounding is a heuristic.
            
            # Compute effective coefficients for U
            # V is (m*p, rank), W is (p*n, rank)
            # We need to contract V and W over k.
            # V[j*p + k, r] * W[k*n + i, r]
            # Let's construct a tensor A[i, j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            A = np.zeros((n, m, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A[i, j, :] += V_curr[j * p + k, :] * W_rand[k * n + i, :]
            
            # Now we need U[i, m+j, :] such that sum_r U[i, m+j, r] A[i, j, r] = 1
            # Minimum norm solution: U[i, m+j, :] = A[i, j, :] / (A[i, j, :] . A[i, j, :])
            for i in range(n):
                for j in range(m):
                    norm_sq = np.dot(A[i, j, :], A[i, j, :])
                    if norm_sq > 1e-9:
                        U_curr[i * m + j, :] = A[i, j, :] / norm_sq
                    else:
                        U_curr[i * m + j, :] = 0
            
            # Similarly for V
            B = np.zeros((m, p, rank))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        B[j, k, :] += U_curr[i * m + j, :] * W_rand[k * n + i, :]
            
            for j in range(m):
                for k in range(p):
                    norm_sq = np.dot(B[j, k, :], B[j, k, :])
                    if norm_sq > 1e-9:
                        V_curr[j * p + k, :] = B[j, k, :] / norm_sq
                    else:
                        V_curr[j * p + k, :] = 0
            
            # Check residual
            residual = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_curr[i * m + j, :], V_curr[j * p + k, :] * W_rand[k * n + i, :])
                        residual += (val - 1.0) ** 2
            
            if residual < 1e-6:
                # Try to round to integers/half-integers
                U_rounded = np.round(U_curr * 2) / 2.0
                V_rounded = np.round(V_curr * 2) / 2.0
                W_rounded = np.round(W_rand * 2) / 2.0
                
                # Verify exactness
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                exact = False
                                break
                        if not exact: break
                    if not exact: break
                
                if exact:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
    
    return None

# EVOLVE-BLOCK-END
