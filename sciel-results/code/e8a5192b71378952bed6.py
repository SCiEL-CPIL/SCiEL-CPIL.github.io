"""Find rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly computes matrix multiplication.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 if the (i,k) entry is produced by multiplying row i of A by column k of B,
    which it always is, so T[i,j,k] = 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Build the tensor and check all entries equal 1
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We check by computing the full tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Compute via einsum
    # First: A[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] -> (n, m, p)
    # Then: T[i,j,k] = sum_k A[i,j,k] * W3[k,i,k] ... no that's not right
    # Actually T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # = sum_c (U3[i,j,c] * V3[j,k,c]) * W3[k,i,c]
    # Let's compute B[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c]
    B = np.einsum('ijc,jkc->ijk', U3, V3)
    # Then T[i,j,k] = sum_c ... wait, we already summed over c in B
    # T[i,j,k] = B[i,j,k] * ... no
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a 4-way contraction. Let me think differently.
    # 
    # Alternative: for each c, the rank-1 tensor is outer(U_col, V_col, W_col)
    # where U_col has shape (n*m,), V_col has shape (m*p,), W_col has shape (p*n,)
    # Reshape U_col to (n,m), V_col to (m,p), W_col to (p,n)
    # Then the contribution to T[i,j,k] is U_col[i,j] * V_col[j,k] * W_col[k,i]
    #
    # So T = sum_c outer(U3[:, :, c], V3[:, :, c], W3[:, :, c]) with appropriate reshaping
    # T[i,j,k] += U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # Use einsum: T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kin->ijk', U3, V3, W3)
    
    # T should be all ones
    return np.allclose(T, 1.0, atol=1e-9)

def _try_rank_decomposition(n, m, p, rank, max_time=30.0):
    """Try to find a valid rank-rank decomposition using iterative refinement.
    
    Strategy: Start with random initialization, then use a fixed-point/gradient approach
    to minimize the error. If we can get error to ~0, we have a valid decomposition.
    """
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Reshape for convenience
    # U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
    
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    for attempt in range(5):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with small random values
        np.random.seed(attempt * 1000 + 42)
        U = np.random.randn(n, m, rank) * 0.1
        V = np.random.randn(m, p, rank) * 0.1
        W = np.random.randn(p, n, rank) * 0.1
        
        # Iterative refinement: for each c, we can try to optimize
        # Actually, let's use a different approach: 
        # Fix two matrices and solve for the third.
        
        # Strategy: Alternate between solving for U, V, W given the others.
        # Given V and W, solve for U:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        # For fixed i,j: sum_c U[i,j,c] * (sum_k V[j,k,c] * W[k,i,c]) = 1
        # Let A[i,j,c] = sum_k V[j,k,c] * W[k,i,c]  (shape: n, m, rank)
        # Then we need: U[i,j,:] @ A[i,j,:] = 1 for all i,j
        # This is: for each (i,j), U[i,j,:] is a vector such that dot with A[i,j,:] = 1
        # If A[i,j,:] is nonzero, U[i,j,:] = A[i,j,:] / ||A[i,j,:]||^2
        # But we need this to hold simultaneously for all i,j,k, so it's more complex.
        
        # Better approach: Use the fact that this is a linear system in U given V,W.
        # T[i,j,k] = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])
        # Let M[i,j,k,c] = V[j*p+k, c] * W[k*n+i, c]  (shape: n, m, p, rank)
        # Then T[i,j,k] = sum_c U[i*m+j, c] * M[i,j,k,c]
        # For each (i,j,k): sum_c U[i*m+j, c] * M[i,j,k,c] = 1
        # This is a system: for each (i,j), the vector U[i*m+j, :] must satisfy
        # sum_k M[i,j,k,:] @ U[i*m+j, :] ... no
        # 
        # Actually: for fixed (i,j), we have p equations (one for each k):
        # sum_c U[i*m+j, c] * M[i,j,k,c] = 1 for k=0,...,p-1
        # This is a linear system: M_ij @ U_ij = 1_p where M_ij is (p, rank)
        # M_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
        
        # So for each (i,j), we solve M_ij @ u_ij = ones(p) for u_ij in R^rank
        # If rank >= p, this may have solutions.
        
        # Let's try alternating minimization:
        # 1. Given V, W: solve for U
        # 2. Given U, W: solve for V  
        # 3. Given U, V: solve for W
        # Repeat until convergence.
        
        for iteration in range(200):
            if time.time() - start_time > max_time:
                break
            
            # Solve for U given V, W
            U_new = np.zeros((n, m, rank))
            for i in range(n):
                for j in range(m):
                    # M_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M_ij = V[j, :, :] * W[:, i, :]  # (p, rank)
                    # Solve M_ij @ u = ones(p)
                    try:
                        # Least squares solution
                        u, residuals, rank_val, sv = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)
                        U_new[i, j, :] = u
                    except:
                        U_new[i, j, :] = U[i, j, :]
            U = U_new
            
            # Solve for V given U, W
            V_new = np.zeros((m, p, rank))
            for j in range(m):
                for k in range(p):
                    # Need: sum_c V[j*p+k, c] * (sum_i U[i*m+j, c] * W[k*n+i, c]) = 1
                    # Let N[j,k,c] = sum_i U[i*m+j, c] * W[k*n+i, c]  (shape: m, p, rank)
                    N_jk = np.sum(U[:, j, :] * W[k, :, :], axis=0)  # (rank,)
                    # Solve: N_jk @ v = 1 ... but this is one equation in rank unknowns
                    # We need this for all (j,k), so it's underdetermined if rank > 1
                    # Use least squares with the constraint that it works for all (j,k)
                    # Actually, for fixed (j,k), we just need one equation: N_jk @ v_jk = 1
                    # Minimum norm solution: v_jk = N_jk / ||N_jk||^2
                    norm_sq = np.dot(N_jk, N_jk)
                    if norm_sq > 1e-12:
                        V_new[j, k, :] = N_jk / norm_sq
                    else:
                        V_new[j, k, :] = V[j, k, :]
            V = V_new
            
            # Solve for W given U, V
            W_new = np.zeros((p, n, rank))
            for k in range(p):
                for i in range(n):
                    # Need: sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
                    # Let P[k,i,c] = sum_j U[i*m+j, c] * V[j*p+k, c]  (shape: p, n, rank)
                    P_ki = np.sum(U[i, :, :] * V[:, k, :], axis=0)  # (rank,)
                    norm_sq = np.dot(P_ki, P_ki)
                    if norm_sq > 1e-12:
                        W_new[k, i, :] = P_ki / norm_sq
                    else:
                        W_new[k, i, :] = W[k, i, :]
            W = W_new
            
            # Check error
            T = np.einsum('ijc,jkc,kin->ijk', U, V, W)
            error = np.max(np.abs(T - 1.0))
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if error < 1e-8:
                break
        
        if best_error < 1e-8:
            break
    
    if best_error < 1e-8:
        # Convert to flat format
        U_flat = best_U.reshape(n * m, rank)
        V_flat = best_V.reshape(m * p, rank)
        W_flat = best_W.reshape(p * n, rank)
        return U_flat, V_flat, W_flat, best_error
    
    return None, None, None, best_error

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin for evaluation overhead
    
    # Start from schoolbook rank and try to go lower
    # Known: for 4x5*5x6, the rank is at most 120 (schoolbook)
    # Try to find the minimum rank
    
    # First, get the schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = (U_school, V_school, W_school, rank_school)
    
    # Try ranks from 120 downward
    # For each rank, try to find a valid decomposition
    for rank in range(rank_school, 110, -1):
        if time.time() > deadline:
            break
        
        max_time_for_rank = min(20.0, max(5.0, deadline - time.time() - 10))
        if max_time_for_rank < 3:
            break
            
        U, V, W, error = _try_rank_decomposition(n, m, p, rank, max_time=max_time_for_rank)
        
        if U is not None and error < 1e-8:
            # Verify
            if _verify_decomposition_fast(U, V, W, n, m, p):
                best_result = (U, V, W, rank)
                # Found a valid decomposition at this rank, continue trying lower
                continue
            else:
                # Error was small but verification failed, skip
                continue
        else:
            # Couldn't find a valid decomposition at this rank
            # If we've been failing for a while, the rank is likely too low
            # But we should still try a few more
            pass
    
    U, V, W, rank = best_result
    
    # Convert to lists of lists with integer/half-integer entries if possible
    # The entries should be integer or half-integer (i.e., multiples of 0.5)
    # If they're not, we may need to round or accept that the score is based on rank only
    
    # Convert numpy arrays to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Try to round to nearest half-integer if close
    def round_to_half(x):
        return round(x * 2) / 2
    
    U_rounded = [[round_to_half(x) for x in row] for row in U_list]
    V_rounded = [[round_to_half(x) for x in row] for row in V_list]
    W_rounded = [[round_to_half(x) for x in row] for row in W_list]
    
    # Check if rounding preserves validity
    U_r = np.array(U_rounded)
    V_r = np.array(V_rounded)
    W_r = np.array(W_rounded)
    
    if _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
        U_list = U_rounded
        V_list = V_rounded
        W_list = W_rounded
    else:
        # Use the unrounded values
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
