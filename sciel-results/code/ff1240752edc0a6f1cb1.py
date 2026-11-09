"""Tensor decomposition for 4x5 x 5x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Schoolbook fallback: rank = n*m*p = 100
    rank_school = n * m * p
    U_fallback = np.zeros((n*m, rank_school))
    V_fallback = np.zeros((m*p, rank_school))
    W_fallback = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i*m+j, col] = 1
                V_fallback[j*p+k, col] = 1
                W_fallback[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Verify exact decomposition using numpy for speed."""
        # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    if abs(s - 1.0) > 1e-8:
                        errors.append((i,j,k,s))
        return len(errors) == 0, errors
    
    # Try structured decompositions using block methods
    # Strategy: Use the fact that we can combine terms
    
    best_rank = rank_school
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    
    # Approach: Try to find rank R decomposition by solving linear system
    # For a fixed R, we need U,V,W such that for all i,j,k:
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a system of n*m*p = 100 equations in R*(n*m + m*p + p*n) = R*65 unknowns
    # But it's nonlinear (products of variables)
    
    # Better: Fix two matrices and solve for the third linearly
    # If U and V are fixed, then for each (i,j,k):
    #   sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
    # Let M[i,j,k,r] = U[i*m+j,r] * V[j*p+k,r]
    # Then for each (i,k): sum_j sum_r M[i,j,k,r] * W[k*n+i,r] = 1
    # Wait, that's not quite right either.
    
    # Actually, for fixed U,V, the constraint is:
    # For each (i,j,k): sum_r [U[i*m+j,r] * V[j*p+k,r]] * W[k*n+i,r] = 1
    # This is linear in W! For each (i,k), we have m equations (one for each j):
    #   sum_r [U[i*m+j,r] * V[j*p+k,r]] * W[k*n+i,r] = 1, for j=0..m-1
    # So we have m equations in R unknowns (the W[k*n+i,:] vector)
    
    # Let's try: random U,V, then solve for W
    # For this to work, we need the system to be consistent
    
    for target_rank in [25, 26, 27, 28, 29, 30, 32, 35, 40, 45, 50, 60, 70, 80, 90]:
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Random U, V with entries from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            
            # For each (i,k), solve for W[k*n+i,:]
            # System: for j in 0..m-1: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            W = np.zeros((p*n, target_rank))
            consistent = True
            
            for i in range(n):
                for k in range(p):
                    # Build m x target_rank system
                    A_sys = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            A_sys[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    # Solve A_sys * W_vec = ones(m)
                    # Use least squares
                    try:
                        W_vec, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                        # Check if solution is exact
                        if np.allclose(A_sys @ W_vec, np.ones(m), atol=1e-8):
                            W[k*n+i, :] = W_vec
                        else:
                            consistent = False
                            break
                    except:
                        consistent = False
                        break
                
                if not consistent:
                    break
            
            if consistent:
                # Verify
                ok, errors = verify(U, V, W, target_rank)
                if ok:
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
        
        if found:
            break
    
    # If we found something better, return it; else return schoolbook
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
# EVOLVE-BLOCK-END
