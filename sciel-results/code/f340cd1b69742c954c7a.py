"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import nnls
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if we're computing (AB)[i,k]
    # We want to decompose T as sum_{r=1}^{rank} U_r ⊗ V_r ⊗ W_r
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # and T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook rank is n*m*p = 90
    # We want to find a lower rank decomposition
    
    # Known bounds:
    # - For 3x5 * 5x6, the rank is at most 90 (schoolbook)
    # - Strassen-type algorithms can reduce rank for certain sizes
    # - The exact minimal rank is unknown but bounded
    
    # Let's try to find a decomposition with rank < 90
    
    # First, let's verify the schoolbook decomposition works
    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition with reduced rank
    # Strategy: Use a randomized approach to find a low-rank decomposition
    
    # The tensor T can be viewed as a 3D array of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    
    # Let's construct the tensor explicitly
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # For a rank-r decomposition, we need:
    # T[a,b,c] = sum_{r=1}^{rank} U[a,r] * V[b,r] * W[c,r]
    
    # This is equivalent to finding matrices U, V, W such that
    # for all a, b, c: T[a,b,c] = (U @ diag(V[b,:]) @ W.T)[a,c]... no that's not right
    
    # Actually, T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This can be written as: for fixed b, T[:,b,:] = V[:,b] (element-wise) (U @ W.T)
    # i.e., T[:,b,:] = U * V[:,b] (broadcast) * W.T... 
    
    # Let me think differently. For each b (which corresponds to a pair (j,k)),
    # the slice T[:,b,:] is an (n*m) x (p*n) matrix.
    # T[a,b,c] = 1 if a = i*m+j, c = k*n+i, where b = j*p+k
    
    # So for fixed j,k: T[i*m+j, j*p+k, k*n+i'] = 1 if i' = i, else 0
    # This means T[:, j*p+k, :] has 1s at positions (i*m+j, k*n+i) for all i
    
    # This is a permutation-like structure.
    
    # Let's try a direct optimization approach.
    # We'll use a heuristic: start with a random initialization and try to reduce rank.
    
    # Actually, let's try a more structured approach.
    # For matrix multiplication, there are known algorithms with rank less than n*m*p.
    
    # Let's try rank 85 first and see if we can find a valid decomposition.
    
    def try_find_decomposition(target_rank, max_time=10):
        """Try to find a rank-target_rank decomposition using iterative methods."""
        t0 = time.time()
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # We want to minimize: sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
        # subject to the constraint that the solution is exact (zero error)
        
        # Use alternating least squares (ALS)
        for iteration in range(200):
            if time.time() - t0 > max_time:
                break
            
            # Update U: for each a, solve for U[a,:] given V and W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,:,:] = U[a,:] (element-wise) (V * W.T)
            # i.e., T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let M_a[b,c] = V[b,:] * W[c,:] (element-wise product)
            # Then T[a,:,:] = U[a,:] @ M_a.T... no
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let's define for each r: B_r[b,c] = V[b,r] * W[c,r]
            # Then T[a,b,c] = sum_r U[a,r] * B_r[b,c]
            # This is a linear system: T[a,:,:] = sum_r U[a,r] * B_r[b,c]
            # Vectorize: T[a].flatten() = U[a,:] @ B.flatten() where B is stacked
            # Actually, let's form the matrix:
            # For fixed a, we have: T[a,b,c] for all b,c
            # This equals sum_r U[a,r] * V[b,r] * W[c,r]
            # Let's create a matrix A_a of shape (m*p*p*n, target_rank)
            # where A_a[idx, r] = V[b,r] * W[c,r] for the (b,c) corresponding to idx
            # Then T[a].flatten() = A_a.T @ U[a,:]
            
            for a in range(n*m):
                if time.time() - t0 > max_time:
                    break
                # Build the design matrix for this a
                # A has shape (m*p*p*n, target_rank)
                # A[b*p*n+c, r] = V[b,r] * W[c,r]
                A = np.zeros((m*p*p*n, target_rank))
                for b in range(m*p):
                    for c in range(p*n):
                        A[b*p*n+c, :] = V[b, :] * W[c, :]
                # Solve: T[a].flatten() = A.T @ U[a,:]
                # This is an underdetermined system if m*p*p*n > target_rank
                # Use least squares
                sol, _ = np.linalg.lstsq(A.T, T[a].flatten(), rcond=None)
                U[a, :] = sol
            
            # Update V: similar
            for b in range(m*p):
                if time.time() - t0 > max_time:
                    break
                A = np.zeros((n*m*p*n, target_rank))
                for a in range(n*m):
                    for c in range(p*n):
                        A[a*p*n+c, :] = U[a, :] * W[c, :]
                sol, _ = np.linalg.lstsq(A.T, T[:, b, :].flatten(), rcond=None)
                V[b, :] = sol
            
            # Update W: similar
            for c in range(p*n):
                if time.time() - t0 > max_time:
                    break
                A = np.zeros((n*m*m*p, target_rank))
                for a in range(n*m):
                    for b in range(m*p):
                        A[a*m*p+b, :] = U[a, :] * V[b, :]
                sol, _ = np.linalg.lstsq(A.T, T[:, :, c].flatten(), rcond=None)
                W[c, :] = sol
        
        # Check if we have an exact decomposition
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(target_rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (val - 1) ** 2
        
        if error < 1e-6:
            # Round to nearest half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Verify again after rounding
            if verify_decomposition(U, V, W):
                return U, V, W
        
        return None
    
    # Try various ranks from low to high
    for target_rank in range(70, 91):
        if time.time() > deadline:
            break
        
        result = try_find_decomposition(target_rank, max_time=5)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

