"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build the target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We need: T[a,b,c] = sum_c U[a,c] * V[b,c] * W[c_index,c]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    
    # Try to find a low-rank decomposition
    # Start with schoolbook and try to improve
    
    # Schoolbook: rank = n*m*p = 100
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify_decomp(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the multiplication tensor."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Try to find a better decomposition using iterative refinement
    # We'll try ranks from 50 to 100 and see if we can find one
    
    best_result = None
    best_rank = rank_school
    
    # First, verify schoolbook works
    if verify_decomp(U_school, V_school, W_school):
        best_rank = rank_school
        best_result = (U_school.copy(), V_school.copy(), W_school.copy())
    
    # Try to find lower rank decompositions
    # Use a simple iterative method for each candidate rank
    for candidate_rank in range(50, 100):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random values
        np.random.seed(candidate_rank)
        U = np.random.randn(n*m, candidate_rank) * 0.5
        V = np.random.randn(m*p, candidate_rank) * 0.5
        W = np.random.randn(p*n, candidate_rank) * 0.5
        
        # Iterative refinement: alternate between fixing two and solving for the third
        converged = False
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            # Fix U, V, solve for W: for each (i,j,k), we need
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is: for each c_index, W[c_index, :] should satisfy
            # sum_c U[a,c] * V[b,c] * W[c_index,c] = 1 for all (a,b,c_index) with a=i*m+j, b=j*p+k, c_index=k*n+i
            
            # Actually, this is a system of linear equations. Let's set it up properly.
            # For each output index c_idx (0 to p*n-1), we have:
            # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
            # where T[a,b,c_idx] = 1 if a=i*m+j, b=j*p+k, c_idx=k*n+i for some i,j,k
            # with consistent indices, else 0.
            
            # Let's build the linear system for W.
            # For each c_idx, the equation is:
            # sum_c (U[a,c] * V[b,c]) * W[c_idx,c] = T[a,b,c_idx]
            # But a and b are determined by c_idx and the indices.
            
            # Actually, for a given c_idx = k*n + i, the relevant (a,b) pairs are:
            # a = i*m + j for j=0..m-1, b = j*p + k for j=0..m-1
            # So for each j, we have: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c = 1
            
            # This is a separate equation for each (i,k) pair, with the constraint that
            # the sum is over c of (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1 for each j.
            
            # Wait, no. For a fixed (i,k), and for each j, we have:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # So for a fixed (i,k), we have m equations (one for each j), all involving the same W[k*n+i, :].
            
            # Let's collect all equations. For each (i,j,k):
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is a linear equation in W[k*n+i, c] for c=0..rank-1.
            # The coefficient of W[k*n+i, c] is U[i*m+j, c] * V[j*p+k, c].
            
            # So we have a system: M * W_flat = 1, where M is a (n*m*p) x (p*n*rank) matrix.
            # But that's too large. Let's do it row by row for W.
            
            # For each c_idx = k*n + i, we have m equations (for j=0..m-1):
            # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[c_idx, c] = 1
            
            # So for each c_idx, we have a system of m equations in rank unknowns.
            # If m < rank, this is underdetermined. We can find a minimum-norm solution.
            
            # Let's build the system for all c_idx at once.
            # Actually, let's just do it iteratively for each c_idx.
            
            W_new = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    c_idx = k*n + i
                    # Build the m x rank system
                    A_sys = np.zeros((m, candidate_rank))
                    b_sys = np.ones(m)
                    for j in range(m):
                        A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    # Solve A_sys * W[c_idx, :] = b_sys
                    try:
                        W_new[c_idx, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                    except:
                        W_new[c_idx, :] = W[c_idx, :]
            
            W = W_new
            
            # Similarly for U and V
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    a_idx = i*m + j
                    # For each k, we have: sum_c U[a_idx, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    # So: sum_c (V[j*p+k, c] * W[k*n+i, c]) * U[a_idx, c] = 1 for each k
                    A_sys = np.zeros((p, candidate_rank))
                    b_sys = np.ones(p)
                    for k in range(p):
                        A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U_new[a_idx, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                    except:
                        U_new[a_idx, :] = U[a_idx, :]
            
            U = U_new
            
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    b_idx = j*p + k
                    # For each i, we have: sum_c U[i*m+j, c] * V[b_idx, c] * W[k*n+i, c] = 1
                    # So: sum_c (U[i*m+j, c] * W[k*n+i, c]) * V[b_idx, c] = 1 for each i
                    A_sys = np.zeros((n, candidate_rank))
                    b_sys = np.ones(n)
                    for i in range(n):
                        A_sys[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V_new[b_idx, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                    except:
                        V_new[b_idx, :] = V[b_idx, :]
            
            V = V_new
            
            # Check convergence
            if verify_decomp(U, V, W, tol=1e-6):
                converged = True
                break
        
        if converged:
            # Try to round to integers/half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if verify_decomp(U_round, V_round, W_round, tol=1e-9):
                best_rank = candidate_rank
                best_result = (U_round.copy(), V_round.copy(), W_round.copy())
                break  # Found a valid decomposition, stop searching
            
            # If rounding didn't work, try with the unrounded values
            if verify_decomp(U, V, W, tol=1e-9):
                best_rank = candidate_rank
                best_result = (U.copy(), V.copy(), W.copy())
                break
    
    # Return the best result found
    if best_result is not None:
        U_final, V_final, W_final = best_result
        # Convert to lists as required
        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }
    else:
        # Fall back to schoolbook
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
# EVOLVE-BLOCK-END
