"""Matrix multiplication 3x4x7: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE_MARGIN = 10.0  # seconds before the 150s wall-clock limit

# Known lower bounds / upper bounds for 3x4x7
# Schoolbook rank = 3*4*7 = 84
# Known: rank(3x4x7) is at most 84 (schoolbook)
# Literature suggests rank(3x4x7) could be lower, but let's try to find it.
# For small matrices, known results:
#   2x2x2: rank 7 (Strassen)
#   2x2x3: rank 11 (best known)
#   3x3x3: rank 23 (best known exact)
#   3x4x4: ?
#   3x4x7: ?

# Strategy:
# 1. Start with schoolbook (rank 84)
# 2. Try to reduce rank via random initialization + gradient-free local search
#    or via structured ansatz.
# 3. Use a verification step to ensure exactness.

def _verify(U, V, W, n=3, m=4, p=7, tol=1e-10):
    """Verify that U, V, W give an exact decomposition of matrix mult tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and check it equals delta[i,j,k] (which is 1 iff the indices match the
    # standard matrix multiplication: T[i,j,k] should be 1 for the (i,j,k)
    # triple that corresponds to C[i,k] = sum_j A[i,j]*B[j,k]).
    # Actually the matrix multiplication tensor M has M[i,j,k] = 1 iff
    # we're computing C[i,k] = sum_j A[i,j] B[j,k].
    # So M[i,j,k] = 1 for all valid (i,j,k) in the standard product.
    # Wait: the tensor for matrix multiplication is T[i,j,k] = 1 for all
    # (i,j,k) with 0<=i<n, 0<=j<m, 0<=k<p. That's just the all-ones tensor
    # of shape (n,m,p)? No.
    
    # Let me re-read the task. The tensor for multiplying A (n x m) by B (m x p)
    # to get C (n x p) is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k in range.
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # with W indexed as W[k*n + i, c].
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's verify this.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for easy access
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)  # W[k*n+i, c] -> W_3d[k, i, c]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_3d[i, j, :], np.dot(V_3d[j, k, :], W_3d[k, i, :]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err < tol

def _verify_fast(U, V, W, n=3, m=4, p=7, tol=1e-10):
    """Faster verification using numpy broadcasting."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # U[i,j,:] * V[j,k,:] * W[k,i,:]
    # Shape: (n, m, p, rank) -> sum over rank
    # U: (n, m, 1, 1, rank)
    # V: (1, m, p, 1, rank)  -- wait, need to be careful
    # Let's use einsum
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol

def _schoolbook(n=3, m=4, p=7):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(n=3, m=4, p=7, target_rank=None, seed=42, time_budget=120.0):
    """
    Try to find a decomposition of rank < n*m*p using random initialization
    and iterative refinement.
    """
    if target_rank is None:
        target_rank = n * m * p - 1
    
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with random matrices
    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1
    
    # Scale to get closer to the target
    # The product U[i,j,c]*V[j,k,c]*W[k,i,c] should sum to 1
    # Let's normalize
    scale = (n * m * p) ** (1.0 / 6.0)
    U *= scale
    V *= scale
    W *= scale
    
    U_3d = U.reshape(n, m, target_rank)
    V_3d = V.reshape(m, p, target_rank)
    W_3d = W.reshape(p, n, target_rank)
    
    # Iterative refinement: update one factor at a time
    # For fixed V, W, the optimal U[i,j,c] can be found by solving a linear system
    # Similarly for V and W.
    
    max_iter = 5000
    for it in range(max_iter):
        if time.time() - start_time > time_budget - DEADLINE_MARGIN:
            break
        
        # Update U: for each (i,j), find U[i,j,:] to minimize
        # sum_{k} (target - sum_c U[i,j,c] V[j,k,c] W[k,i,c])^2
        # This is a linear least squares problem.
        # Let r = target_rank
        # For fixed (i,j), we want: sum_c U[i,j,c] * A[c] = b
        # where A[c] = sum_k V[j,k,c] W[k,i,c]  (this is a vector of length r)
        # and b = 1 (for each k, but we sum over k)
        # Actually: sum_k sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        # sum_c U[i,j,c] * (sum_k V[j,k,c] W[k,i,c]) = 1
        # This is one equation with r unknowns. Underdetermined if r > 1.
        # We want the minimum-norm solution: U[i,j,:] = A / (A.A)
        
        for i in range(n):
            for j in range(m):
                # A[c] = sum_k V[j,k,c] W[k,i,c]
                A = np.zeros(target_rank)
                for k in range(p):
                    A += V_3d[j, k, :] * W_3d[k, i, :]
                denom = np.dot(A, A)
                if denom > 1e-15:
                    U_3d[i, j, :] = A / denom
                else:
                    # Perturb
                    U_3d[i, j, :] = rng.randn(target_rank) * 0.01
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros(target_rank)
                for i in range(n):
                    A += U_3d[i, j, :] * W_3d[k, i, :]
                denom = np.dot(A, A)
                if denom > 1e-15:
                    V_3d[j, k, :] = A / denom
                else:
                    V_3d[j, k, :] = rng.randn(target_rank) * 0.01
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros(target_rank)
                for j in range(m):
                    A += U_3d[i, j, :] * V_3d[j, k, :]
                denom = np.dot(A, A)
                if denom > 1e-15:
                    W_3d[k, i, :] = A / denom
                else:
                    W_3d[k, i, :] = rng.randn(target_rank) * 0.01
        
        # Compute error
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_U = U_3d.copy().reshape(n * m, target_rank)
            best_V = V_3d.copy().reshape(m * p, target_rank)
            best_W = W_3d.copy().reshape(p * n, target_rank)
        
        if best_err < 1e-12:
            break
    
    if best_err < 1e-10:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_err
    
    return None, None, None, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    current_rank = n * m * p
    
    # Try to reduce rank
    # Try various target ranks from current_rank - 1 downwards
    for target_rank in range(current_rank - 1, max(n * m, m * p, p * n) - 1, -1):
        if time.time() - start_time > 120.0 - DEADLINE_MARGIN:
            break
        
        U, V, W, err = _try_reduce_rank(n, m, p, target_rank, seed=42, time_budget=120.0 - (time.time() - start_time))
        
        if U is not None and err < 1e-10:
            result = {"U": U, "V": V, "W": W}
            current_rank = target_rank
            # Continue trying to reduce further
    
    return result

# For testing
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    print(f"Valid: {_verify_fast(result['U'], result['V'], result['W'])}")
