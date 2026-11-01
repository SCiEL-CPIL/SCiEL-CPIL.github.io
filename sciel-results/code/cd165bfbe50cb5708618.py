"""Find rank-minimal tensor decomposition for 2x5 x 5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Build the multiplication tensor: T[i,j,k] = 1
    # In flattened form: T[i*m+j, j*p+k, k*n+i] = 1
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 50
    
    # Try to find lower rank decompositions
    # Use a structured approach: try ranks from a lower bound up
    
    # Lower bound: max(nm, mp, pn) = max(10, 25, 10) = 25
    # But this is a weak lower bound. Better lower bounds exist.
    
    # Let's try a randomized search for rank r
    # For each rank r, we have r variables in each of U, V, W
    # Total unknowns: r*(nm + mp + pn) = r*45
    # Constraints: nm*mp*pn / (some symmetry) ... actually the tensor has nm*mp*pn entries
    # but many are zero. The non-zero entries are those where (i,j,k) is valid,
    # i.e., for each (i,j,k), T[i,j,k]=1, and we need the product to match.
    
    # Actually, the constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # And for invalid (i,j,k) combinations (which don't exist in this formulation since
    # we're indexing by the actual indices), we don't have additional constraints.
    
    # Wait, the tensor is only defined for valid (i,j,k). The multiplication tensor
    # has entries T[i,j,k] = 1 for all valid i,j,k, and we need the decomposition
    # to reproduce this.
    
    # Number of constraints: n*m*p = 50 (one for each (i,j,k))
    # Number of unknowns: r*(nm + mp + pn) = 45r
    # For r=2, 45*2=90 > 50, so it's potentially feasible.
    
    # Let's try a Newton-like approach or randomized initialization with refinement.
    
    # Actually, let me try a different structural approach.
    # For 2x5 x 5x5, we can use the fact that this is related to bilinear maps.
    
    # Let me try: use a random initialization and then use gradient descent
    # to minimize the error, hoping to find an exact solution.
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Generate schoolbook as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try lower ranks
    # Theoretical lower bound is at least max(nm, mp, pn) = 25
    # But let's try from a reasonable range
    for r in range(25, schoolbook_rank + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        max_attempts = 5 if r < 35 else 2
        for attempt in range(max_attempts):
            if time.time() > deadline:
                    break
                    
            # Random initialization with small values
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            # Use iterative refinement
            # Objective: minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            # This is a non-convex problem.
            
            # Simple approach: use alternating minimization
            # Fix U and V, solve for W, etc.
            
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                    
                # Update W: for each (k,i), we have W[k*n+i, c]
                # The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # Fix U and V, this is linear in W.
                
                # Build the linear system for W
                # For each (i,j,k): sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
                # Group by (k,i): for fixed (k,i), sum over j and c:
                # sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = n... wait
                
                # Actually, for fixed (k,i), the constraint involves all j:
                # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a system of m equations in r unknowns (W[k*n+i, 0..r-1])
                
                # Let me build it properly
                W_new = W.copy()
                for k in range(p):
                    for i in range(n):
                        # For this (k,i), we have m equations (one for each j)
                        # Equation j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        A = np.zeros((m, r))
                        b = np.ones(m)
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = U[i*m + j, c] * V[j*p + k, c]
                        # Solve A @ W[k*n+i, :] = b
                        try:
                            sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                            W_new[k*n + i, :] = sol
                        except:
                            pass
                W = W_new
                
                # Update U: for each (i,j), fix V and W
                # For fixed (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for each k
                # System of p equations in r unknowns
                U_new = U.copy()
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        b = np.ones(p)
                        for k in range(p):
                            for c in range(r):
                                A[k, c] = V[j*p + k, c] * W[k*n + i, c]
                        try:
                            sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                            U_new[i*m + j, :] = sol
                        except:
                            pass
                U = U_new
                
                # Update V: for each (j,k), fix U and W
                # For fixed (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for each i
                # System of n equations in r unknowns
                V_new = V.copy()
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        b = np.ones(n)
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U[i*m + j, c] * W[k*n + i, c]
                        try:
                            sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                            V_new[j*p + k, :] = sol
                        except:
                            pass
                V = V_new
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                            error += (val - 1.0) ** 2
                
                if error < 1e-10:
                    converged = True
                    break
                
                if error < 1e-6 and iteration > 50:
                    # Try to snap to rational/half-integer
                    # Check if close to a simple solution
                    pass
        
        if converged:
            # Verify and store
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                        error += (val - 1.0) ** 2
            if error < 1e-8:
                best_rank = r
                best_decomp = (U, V, W)
                break  # Found a valid decomposition at this rank, try lower? No, we're going up.
    
    # Convert to list format
    U, V, W = best_decomp
    U_list = [[U[i, c] for c in range(best_rank)] for i in range(nm)]
    V_list = [[V[i, c] for c in range(best_rank)] for i in range(mp)]
    W_list = [[W[i, c] for c in range(best_rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
