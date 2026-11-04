"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook fallback
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify function
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Search for lower rank
    # Strategy: Fix U, V randomly, solve for W linearly, check if W is half-integer.
    # We iterate rank from best_rank-1 down to some limit.
    
    # To speed up, we can try specific ranks known to be good or just sweep.
    # Let's sweep from 71 down to 50.
    
    for target_rank in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few random instances for this rank
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Generate random U and V with half-integer entries in [-2, 2]
            # Using small integers to keep W potentially small
            U = np.random.randint(-4, 5, (n * m, target_rank)) * 0.5
            V = np.random.randint(-4, 5, (m * p, target_rank)) * 0.5
            
            W = np.zeros((p * n, target_rank))
            
            valid_W = True
            for k in range(p):
                # Build the linear system for W slice k
                # Variables: W[k*n + i, c] for i in 0..n-1, c in 0..rank-1
                # Equations: sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all i, j
                
                # Matrix A of size (n*m, n*rank)
                # Row idx: i*m+j
                # Col idx: i*rank + c
                A = np.zeros((n * m, n * target_rank))
                for i in range(n):
                    for j in range(m):
                        for c in range(target_rank):
                            # Coefficient for variable W[k*n + i, c]
                            # The term is W[k*n + i, c] * U[i*m + j, c] * V[j*p + k, c]
                            # So in the equation for (i,j), the variable is W[k*n+i, c]
                            # Wait, the sum is over c.
                            # Equation for (i,j): sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
                            # This couples all c for a fixed i.
                            # So for each i, we have a vector of coefficients for the variables W[k*n+i, 0..rank-1].
                            # The equation is dot(coeff_i, W_slice_i) = 1.
                            # Where coeff_i[c] = U[i*m+j, c] * V[j*p+k, c].
                            # But this must hold for ALL j.
                            # So for a fixed i, we have m equations (one for each j) that must be satisfied by the same W_slice_i.
                            # This means W_slice_i must satisfy:
                            # W_slice_i . coeff_{i,0} = 1
                            # W_slice_i . coeff_{i,1} = 1
                            # W_slice_i . coeff_{i,2} = 1
                            # This is a system of 3 equations for rank variables.
                            # If rank > 3, there are multiple solutions. We want a half-integer solution.
                            
                            # Let's solve for each i independently.
                            # For fixed i, we have a matrix M_i of size (m, rank) where M_i[j, c] = U[i*m+j, c] * V[j*p+k, c].
                            # We want W_slice_i (size rank) such that M_i @ W_slice_i = [1, 1, 1]^T.
                            
                            M_i = np.zeros((m, target_rank))
                            for j in range(m):
                                for c in range(target_rank):
                                    M_i[j, c] = U[i * m + j, c] * V[j * p + k, c]
                            
                            # Solve M_i @ W_slice_i = ones(m)
                            # Use least squares or pseudo-inverse if not square/invertible
                            try:
                                # We want a solution. If rank > m, there are infinite solutions.
                                # We want one that is half-integer.
                                # A simple approach: find one solution, then round?
                                # Or, pick a specific solution by adding null space components?
                                # Let's just find the minimum norm solution and see if it's close to half-integers.
                                W_slice_i, residuals, rank_sol, sv = np.linalg.lstsq(M_i, np.ones(m), rcond=None)
                                
                                # Check if W_slice_i is close to half-integers
                                W_rounded = np.round(W_slice_i * 2) / 2
                                
                                # Verify if rounded solution works
                                check = M_i @ W_rounded
                                if np.allclose(check, np.ones(m), atol=1e-9):
                                    W[k * n + i, :] = W_rounded
                                else:
                                    # Try a different random perturbation or fail this attempt
                                    valid_W = False
                                    break
                            except:
                                valid_W = False
                                break
                        if not valid_W:
                            break
                    if not valid_W:
                        break
                if not valid_W:
                    break
            
            if valid_W:
                # Final verification
                if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break # Found valid for this rank, try lower
    
    return best_result
# EVOLVE-BLOCK-END
