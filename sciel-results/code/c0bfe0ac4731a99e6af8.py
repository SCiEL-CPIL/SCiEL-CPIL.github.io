"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i, j, k.
    
    # Start with schoolbook (rank = n*m*p = 60)
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    best_error = 0.0
    
    deadline = time.time() + 140.0  # Leave 10s margin for 150s limit
    
    # Try to find lower rank decompositions via random initialization and optimization
    # We will try ranks from rank0-1 downwards, but given time constraints, we focus on
    # a few promising lower ranks.
    
    # Theoretical lower bound for matrix multiplication A(nxm) * B(mxk) is n*m + m*k + n*k - m
    # For 2,3,10: 6 + 30 + 20 - 3 = 53.
    # So we can try ranks between 53 and 59.
    
    # We will use a simple gradient-free or gradient-based approach.
    # Given the small dimensions, we can try to solve for W given U, V or vice versa.
    # However, the tensor is 3-way.
    
    # Let's try a random search / local optimization for a specific rank r.
    # If we fix U and V, W is determined by:
    # W[k*n+i, c] = T[i,j,k] / (U[i*m+j, c] * V[j*p+k, c]) ? No, it's a sum over c.
    # This is not linear in W if we sum over c.
    
    # Actually, for fixed U and V, the equation is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a system of linear equations for W?
    # Let x_c = W[k*n+i, c].
    # The equation is for fixed i, j, k.
    # sum_c (U[i*m+j, c] * V[j*p+k, c]) * x_c = 1.
    # The coefficient depends on i, j, k.
    # So for each (i, j, k), we have a linear equation in W[k*n+i, :].
    # But W[k*n+i, :] appears in equations for all j.
    # Specifically, for fixed i, k, and varying j, we have:
    # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
    # This is a set of m linear equations for the p*n elements of W? No, for fixed i, k,
    # W[k*n+i, :] is a vector of length r.
    # The equations are for j=0..m-1.
    # So we have m equations for r unknowns (W[k*n+i, 0..r-1]).
    # If m < r, we have underdetermined system. We can find a solution.
    # But we must satisfy this for ALL i, j, k.
    
    # Algorithm for fixed U, V and rank r:
    # For each i in [0,n), k in [0,p):
    #   Let A be an m x r matrix where A[j, c] = U[i*m+j, c] * V[j*p+k, c].
    #   We need W[k*n+i, :] such that A @ W[k*n+i, :] = [1, 1, ..., 1] (vector of ones, length m).
    #   If such W exists, we can pick the minimum norm solution:
    #   W[k*n+i, :] = ones(m) @ pinv(A)
    #   If the system is inconsistent, we can minimize the error.
    
    # We can iterate:
    # 1. Randomly initialize U, V with rank r.
    # 2. Compute W optimally for fixed U, V.
    # 3. Compute error.
    # 4. Update U, V (e.g., using gradient descent or random perturbations).
    
    # Let's try this for a few ranks.
    
    def try_rank(r, iterations=1000):
        if time.time() > deadline:
            return None, None, None, float('inf')
            
        # Initialize U, V randomly
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        
        best_local_error = float('inf')
        best_local_U, best_local_V, best_local_W = U, V, None
        
        for it in range(iterations):
            if time.time() > deadline:
                break
                
            # Compute W
            W = np.zeros((p * n, r))
            for i in range(n):
                for k in range(p):
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    # Solve A @ w = ones(m)
                    # w = pinv(A) @ ones
                    try:
                        w = np.linalg.pinv(A) @ np.ones(m)
                        W[k * n + i, :] = w
                    except:
                        w = np.zeros(r)
                        W[k * n + i, :] = w
                        
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        error += (val - 1.0) ** 2
                        
            if error < best_local_error:
                best_local_error = error
                best_local_U = U.copy()
                best_local_V = V.copy()
                best_local_W = W.copy()
                
            if best_local_error < 1e-10:
                break
                
            # Update U and V randomly
            # Perturb U and V
            U += 0.1 * np.random.randn(n * m, r)
            V += 0.1 * np.random.randn(m * p, r)
            
        return best_local_U, best_local_V, best_local_W, best_local_error

    # Try ranks from rank0-1 down to rank0-5
    # Start with lower ranks first to see if we can beat schoolbook
    for r in range(rank0 - 1, max(50, rank0 - 10), -1):
        if time.time() > deadline:
            break
            
        U_r, V_r, W_r, err_r = try_rank(r, iterations=500)
        
        if err_r < best_error:
            best_error = err_r
            best_U = U_r
            best_V = V_r
            best_W = W_r
            best_rank = r
            
            # If we found a very low error, we can try to polish or move to next rank
            if best_error < 1e-10:
                # Exact solution found?
                # We can break or try lower rank
                pass
                
    # If no better rank found, return schoolbook
    if best_rank == rank0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
