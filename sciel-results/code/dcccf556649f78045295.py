import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 110  # 110s limit, 40s margin for safety
    
    # Fallback to schoolbook
    def get_schoolbook():
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
        
    best_result = get_schoolbook()
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Use a more systematic approach:
    # 1. For each candidate rank R, generate random U, V with entries from a small set
    # 2. For each (i,k) pair, solve the linear system for W
    # 3. Check if W entries are close to half-integers
    # 4. Verify the full decomposition
    
    # Key insight: Use denser random matrices with entries from {-1, 0, 1, 2}
    # and try many more attempts per rank
    
    for R in range(rank_schoolbook - 1, 35, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random instances for this rank
        # Use different entry sets to increase diversity
        entry_sets = [
            [0, 1],
            [-1, 0, 1],
            [-1, 0, 1, 2],
            [0, 1, 2],
            [-2, -1, 0, 1, 2],
        ]
        
        for entry_set in entry_sets:
            if time.time() > deadline:
                    break
            
            # Try several random instances with this entry set
            num_attempts = 30
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break
                    
                # Random U, V with entries from entry_set
                U = np.array([[np.random.choice(entry_set) for _ in range(R)] for _ in range(n * m)])
                V = np.array([[np.random.choice(entry_set) for _ in range(R)] for _ in range(m * p)])
                
                W = np.zeros((p * n, R))
                valid = True
                
                for i in range(n):
                    for k in range(p):
                        # Build M matrix: M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        # Solve M @ W_vec = ones(m)
                        # W_vec is W[k*n+i, :]
                        try:
                            # Check if system is consistent and has a nice solution
                            # First, check rank of M
                            rank_M = np.linalg.matrix_rank(M)
                            
                            if rank_M < m:
                                # Underdetermined or inconsistent - check if ones(m) is in column space
                                # Use least squares and check residual
                                W_vec, residuals, rank_ls, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                                if len(residuals) == 0 or residuals[0] > 1e-10:
                                    valid = False
                                    break
                            else:
                                # Full rank, check if solution is nice
                                W_vec = np.linalg.solve(M, np.ones(m))
                            
                            # Check if W_vec is close to integer/half-integer
                            W_half = np.round(W_vec * 2) / 2
                            
                            # Verify the solution
                            if np.allclose(M @ W_half, np.ones(m), atol=1e-9):
                                W[k * n + i, :] = W_half
                            else:
                                # Try rounding to integers
                                W_int = np.round(W_vec)
                                if np.allclose(M @ W_int, np.ones(m), atol=1e-9):
                                    W[k * n + i, :] = W_int
                                else:
                                    valid = False
                                    break
                        except:
                            valid = False
                            break
                            
                if valid:
                    # Final verification of the entire decomposition
                    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                    all_correct = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for r in range(R):
                                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                                if abs(val - 1.0) > 1e-9:
                                    all_correct = False
                                    break
                            if not all_correct:
                                break
                        if not all_correct:
                            break
                    
                    if all_correct:
                        # Convert to lists
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
