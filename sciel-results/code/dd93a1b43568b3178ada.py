"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84
    
    # Schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def verify(U, V, W, rank):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # T[i,k,a,b,d] = sum_c W[k,i,c] * U[a,b,c] * V[b,d,c]
        T = np.einsum('kic,abc,bdc->ikabd', W_3d, U_3d, V_3d)
        
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and d == k) else 0.0
                            if abs(T[i, k, a, b, d] - expected) > 1e-6:
                                return False
        return True
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition by fixing U and V to random integer matrices
    # and solving for W.
    # We'll try rank 83, 82, ... down to some minimum.
    # For each rank, we try several random initializations of U and V.
    
    # The constraint is:
    # sum_c W[k,i,c] * U[a,b,c] * V[b,d,c] = delta_{a,i} * delta_{d,k}
    
    # This is a linear system for W.
    # Let's vectorize the construction of the linear system.
    
    # We want to solve A * W_flat = b
    # where A has shape (n*p*n*m*p, p*n*rank)
    # and b has shape (n*p*n*m*p)
    
    # But we can be smarter. The equation is:
    # For fixed (i, k, a, b, d):
    # sum_c W[k,i,c] * (U[a,b,c] * V[b,d,c]) = delta_{a,i} * delta_{d,k}
    
    # Let M_{a,b,d,c} = U[a,b,c] * V[b,d,c]
    # Then the equation for (i,k,a,b,d) is:
    # sum_c W[k,i,c] * M_{a,b,d,c} = delta_{a,i} * delta_{d,k}
    
    # We can group equations by (i, k). For fixed (i, k), we have n*m*p equations
    # (one for each a, b, d) and n*rank unknowns (W[k,i,c] for c=0..rank-1).
    # Wait, the unknowns are W[k,i,c], which are n*rank values for fixed k, i.
    # The equations are for all a, b, d. There are n*m*p of them.
    # So for each (i, k), we have a system of n*m*p equations in n*rank unknowns.
    # n*m*p = 3*4*7 = 84.
    # n*rank = 3*83 = 249.
    # This is underdetermined! 84 equations, 249 unknowns.
    # So there should be a solution for almost any random U, V.
    # But we need the solution W to be "nice" (integers or half-integers).
    # Or at least, we need the full system to be consistent.
    # But the system is not independent for different (i, k).
    # Actually, the equations for different (i, k) involve different sets of W variables.
    # W[k,i,c] only appears in equations with that specific (i, k).
    # So the system decomposes into n*p independent subsystems, one for each (i, k).
    # Each subsystem has n*m*p = 84 equations and n*rank = 3*rank unknowns.
    # For rank = 83, we have 249 unknowns and 84 equations.
    # This is always solvable (in fact, underdetermined).
    # But we need the solution to be consistent with the requirement that W is "nice".
    # Or, we can just find ANY solution and check if it's valid.
    # But the problem requires integer/half-integer entries.
    # If we use random integer U and V, the solution W will generally be rational numbers.
    # We can round them and check if the rounded W still satisfies the equations.
    
    # Let's try this approach:
    # 1. For rank r from 83 down to some limit:
    #    a. For several random integer U, V:
    #       i. For each (i, k), solve the linear system for W[k, :, :].
    #       ii. Round W to nearest half-integer.
    #       iii. Verify the full decomposition.
    
    # However, solving 21 linear systems of size 84x249 for each attempt might be slow.
    # Let's estimate: 21 * 84 * 249 * 249 * 21 (for Gaussian elimination?) No, that's too slow.
    # Actually, we can use np.linalg.lstsq for each (i, k).
    # 21 calls to lstsq with 84x249 matrices. Each call is fast.
    
    # Let's try rank 83 first.
    
    max_attempts_per_rank = 20
    
    for target_rank in range(83, 70, -1):  # Try ranks 83 down to 71
        if time.time() > deadline:
            break
        
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline:
                break
            
            # Random integer U and V
            U_rand = np.random.randint(-2, 3, size=(n * m, target_rank))
            V_rand = np.random.randint(-2, 3, size=(m * p, target_rank))
            
            # Reshape
            U_3d = U_rand.reshape(n, m, target_rank)
            V_3d = V_rand.reshape(m, p, target_rank)
            
            # Solve for W
            W_rand = np.zeros((p * n, target_rank))
            
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build the system for W[k, i, :]
                    # Equations: for all a, b, d:
                    # sum_c W[k,i,c] * U[a,b,c] * V[b,d,c] = delta_{a,i} * delta_{d,k}
                    
                    # A_sub has shape (n*m*p, target_rank)
                    # b_sub has shape (n*m*p)
                    
                    A_sub = np.zeros((n * m * p, target_rank))
                    b_sub = np.zeros(n * m * p)
                    
                    eq_idx = 0
                    for a in range(n):
                        for b in range(m):
                            for d in range(p):
                                # Coefficients: U[a,b,c] * V[b,d,c]
                                A_sub[eq_idx, :] = U_3d[a, b, :] * V_3d[b, d, :]
                                b_sub[eq_idx] = 1.0 if (a == i and d == k) else 0.0
                                eq_idx += 1
                    
                    # Solve A_sub * W_sub = b_sub
                    # W_sub has shape (target_rank,)
                    try:
                        W_sub, residuals, rank_sol, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                        
                        # Check residual
                        if len(residuals) > 0 and residuals[0] > 1e-5:
                            valid = False
                            break
                        
                        # Round to nearest half-integer
                        W_sub_rounded = np.round(W_sub * 2.0) / 2.0
                        
                        # Store in W_rand
                        W_rand[k * n + i, :] = W_sub_rounded
                    
                    except Exception:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition
            if verify(U_rand, V_rand, W_rand, target_rank):
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                best_rank = target_rank
                break  # Found a valid decomposition for this rank, try lower
        
        if best_rank < schoolbook_rank:
            break  # Found a better rank, stop
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
