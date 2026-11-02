import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    schoolbook = get_schoolbook()
    best_rank = n * m * p
    best_solution = schoolbook

    start_time = time.time()
    time_limit = 110  # Leave 10s buffer for 120s limit

    # Precompute the target tensor T_target as a 3D array for fast access
    # T_target[i, j, k] = 1 for all i, j, k
    # But the decomposition is over the "flattened" indices.
    # Let's stick to the loop-based check for clarity, but optimize the inner loop.
    
    # A faster verification function
    def verify(U, V, W, tol=1e-9):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Try to find rank < best_rank
    # We will use a structured search based on the "sparse linear algebra" idea.
    # Instead of random ALS, we will try to solve a system of equations for a specific rank R.
    # The equations are: sum_c U[u,c] V[v,c] W[w,c] = 1 for all valid (u,v,w)
    # This is a system of dim_u * dim_v * dim_w equations? No, only n*m*p = 30 non-zero entries in the tensor?
    # Wait, the tensor T[i,j,k] is 1 for ALL i,j,k.
    # So we have n*m*p = 30 equations.
    # Unknowns: R * (dim_u + dim_v + dim_w) = R * (6 + 15 + 10) = 31 R.
    # For R=29, 31*29 = 899 unknowns, 30 equations. Underdetermined.
    # We can fix some variables to 0 or 1 to reduce the search space (canonical basis).
    
    # Let's try a direct numerical approach for R=29.
    # We can use the fact that we can set the first R-1 columns of U, V, W to have specific structures.
    # Or simply use a random restart with a more robust solver.
    
    # Let's try R from 29 down to 25.
    for R in range(best_rank - 1, 24, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization with small values to help convergence
            U = np.random.randn(dim_u, R) * 0.1
            V = np.random.randn(dim_v, R) * 0.1
            W = np.random.randn(dim_w, R) * 0.1
            
            # ALS iteration
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                for u in range(dim_u):
                    i = u // m
                    j = u % m
                    # T[i,j,k] = 1 for all k
                    # Sum_c U[u,c] V[j*p+k, c] W[k*n+i, c] = 1
                    # This is a linear system for U[u, :]
                    # Let A be a matrix where A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Then A @ U[u, :] = 1 (vector of ones of length p)
                    A = np.zeros((p, R))
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        A[k, :] = V[v_idx, :] * W[w_idx, :]
                    
                    # Solve A @ x = ones
                    try:
                        U[u, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except:
                        pass

                # Update V
                for v in range(dim_v):
                    j = v // p
                    k = v % p
                    # T[i,j,k] = 1 for all i
                    # Sum_c U[i*m+j, c] V[v, c] W[k*n+i, c] = 1
                    # Let A be a matrix where A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # Then A @ V[v, :] = 1 (vector of ones of length n)
                    A = np.zeros((n, R))
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        A[i, :] = U[u_idx, :] * W[w_idx, :]
                    
                    try:
                        V[v, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    except:
                        pass

                # Update W
                for w in range(dim_w):
                    k = w // n
                    i = w % n
                    # T[i,j,k] = 1 for all j
                    # Sum_c U[i*m+j, c] V[j*p+k, c] W[w, c] = 1
                    # Let A be a matrix where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # Then A @ W[w, :] = 1 (vector of ones of length m)
                    A = np.zeros((m, R))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        A[j, :] = U[u_idx, :] * V[v_idx, :]
                    
                    try:
                        W[w, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except:
                        pass
            
            # Check validity
            if verify(U, V, W, tol=1e-6):
                # Check if entries are close to integers or half-integers
                valid = True
                for mat in [U, V, W]:
                    for val in mat:
                        closest_half = round(val * 2) / 2
                        if abs(val - closest_half) > 1e-3:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    U_snapped = np.round(U * 2) / 2
                    V_snapped = np.round(V * 2) / 2
                    W_snapped = np.round(W * 2) / 2
                    best_rank = R
                    best_solution = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                    found = True
                    break
        
        if found:
            break

    return best_solution
# EVOLVE-BLOCK-END
