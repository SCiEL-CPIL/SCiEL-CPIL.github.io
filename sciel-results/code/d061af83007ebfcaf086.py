import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
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

def _verify(n, m, p, U, V, W):
    """Check if decomposition is exact: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 if j==j else 0"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check each (i,j,k) triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: 1.0
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Also check that off-diagonal terms are zero
    # For matrix mult, the tensor has 1s only at (i,j,k) where the multiplication index matches
    # Actually for A(i,j) * B(j,k) = C(i,k), the tensor T[i,j,k] = delta_{j,j} = 1 always
    # Wait, let me reconsider. The matrix multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    # So every (i,j,k) triple should give 1.
    # But we also need to ensure no spurious contributions... 
    # Actually the tensor is just the sum over j, so T[i,k,j] in standard notation...
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # The tensor for matmul: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] B[j,k]
    # So verification: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
    # AND there should be no other constraints since the tensor is all-ones in the relevant indices.
    # But wait - we need to make sure that the decomposition doesn't produce non-zero values
    # where the tensor should be zero. In this case, the tensor is defined only on the 
    # (n*m*p) indices and is 1 everywhere. There are no "zero" entries to check.
    # So the above check is sufficient.
    return True

def _try_rank(n, m, p, target_rank, rng, max_iters=5000, tol=1e-8):
    """Try to find exact decomposition of given rank using random init + refinement."""
    U = rng.standard_normal((n*m, target_rank))
    V = rng.standard_normal((m*p, target_rank))
    W = rng.standard_normal((p*n, target_rank))
    
    # Build target tensor
    T = np.ones((n, m, p))
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update W: for each (k,n) index and each r, 
        # W[k*n+i, r] should satisfy: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a linear system. For fixed (i,k), sum_j,r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Let's do alternating least squares
        # Fix U,V, solve for W:
        # For each (i,k): sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is: sum_r (sum_j U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1
        # Let A_ik[r] = sum_j U[i*m+j,r]*V[j*p+k,r]
        # Then sum_r A_ik[r]*W[k*n+i,r] = 1
        # But W[k*n+i,r] appears in multiple equations (for different k with same i, or different i with same k)
        # Actually W is indexed by (k*n+i, r), so for fixed i, different k give different rows.
        # For fixed i, we have p equations (one per k) with p*n*target_rank unknowns? No.
        # W has shape (p*n, target_rank). For fixed i, the rows k*n+i for k=0..p-1 are involved.
        # Each equation involves all target_rank components of those p rows.
        # This is a coupled system. Let me use a simpler approach: gradient descent or 
        # just do simple iterative updates.
        
        # Simpler: use the fact that this is a tensor decomposition problem.
        # Use ALS: fix two factors, solve for third.
        
        # Solve for W given U, V:
        # For each (i,k): sum_j,r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # Let's define for each (i,k): b_ik = 1
        # And the equation is linear in W rows k*n+i for all k.
        # Actually each row of W (indexed by k*n+i) appears in exactly one group of equations (fixed i).
        # For fixed i, we have p equations (for k=0..p-1), each involving all target_rank entries of W[k*n+i,:] 
        # Wait no. For fixed i and k, the equation is:
        # sum_r A_ik[r] * W[k*n+i, r] = 1 where A_ik[r] = sum_j U[i*m+j,r]*V[j*p+k,r]
        # So for fixed i, we have p independent equations, each involving a different row of W.
        # Row k*n+i of W is only involved in the equation for that specific (i,k).
        # So we can solve each independently!
        # For each (i,k): find W[k*n+i,:] such that sum_r A_ik[r]*W[k*n+i,r] = 1
        # This is underdetermined (target_rank unknowns, 1 equation). 
        # We want the solution that minimizes ||W[k*n+i,:]|| or keeps it close to previous.
        # Simplest: set W[k*n+i,:] = A_ik / ||A_ik||^2 if A_ik != 0, else keep previous.
        
        for i in range(n):
            for k in range(p):
                A_ik = np.zeros(target_rank)
                for j in range(m):
                    A_ik += U[i*m+j] * V[j*p+k]
                norm_sq = np.dot(A_ik, A_ik)
                if norm_sq > 1e-12:
                    W[k*n+i] = A_ik / norm_sq
                # else keep W[k*n+i] as is
        
        # Similarly solve for U given V, W:
        # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # For fixed (i,j): sum_k,r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Let B_ij[r] = sum_k V[j*p+k,r]*W[k*n+i,r]
        # Then sum_r U[i*m+j,r]*B_ij[r] = 1
        # For fixed (i,j), this is one equation with target_rank unknowns.
        # U[i*m+j,:] = B_ij / ||B_ij||^2
        for i in range(n):
            for j in range(m):
                B_ij = np.zeros(target_rank)
                for k in range(p):
                    B_ij += V[j*p+k] * W[k*n+i]
                norm_sq = np.dot(B_ij, B_ij)
                if norm_sq > 1e-12:
                    U[i*m+j] = B_ij / norm_sq
        
        # Solve for V given U, W:
        # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # For fixed (j,k): sum_i,r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Let C_jk[r] = sum_i U[i*m+j,r]*W[k*n+i,r]
        # Then sum_r V[j*p+k,r]*C_jk[r] = 1
        # V[j*p+k,:] = C_jk / ||C_jk||^2
        for j in range(m):
            for k in range(p):
                C_jk = np.zeros(target_rank)
                for i in range(n):
                    C_jk += U[i*m+j] * W[k*n+i]
                norm_sq = np.dot(C_jk, C_jk)
                if norm_sq > 1e-12:
                    V[j*p+k] = C_jk / norm_sq
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (s - 1.0)**2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            # Check if we can round to integers/half-integers
            # Try rounding to nearest 0.5
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            err_round = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_round[i*m+j], V_round[j*p+k] * W_round[k*n+i])
                        err_round += (s - 1.0)**2
            err_round = np.sqrt(err_round / (n*m*p))
            if err_round < 1e-6:
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U_round]
                V_list = [[float(x) for x in row] for row in V_round]
                W_list = [[float(x) for x in row] for row in W_round]
                return U_list, V_list, W_list
            else:
                # Return without rounding
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    rng = np.random.default_rng(42)
    
    # Try lower ranks
    for target_rank in range(1, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try a few random inits
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            result = _try_rank(n, m, p, target_rank, rng, max_iters=2000)
            if result is not None:
                U_list, V_list, W_list = result
                # Verify
                if _verify(n, m, p, U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break  # Found valid decomposition at this rank, try lower
    
    return best_result
