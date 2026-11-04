"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B
    # contribute to (i,k) entry of C. The tensor is T[i*m+j, j*p+k, k*n+i] = 1.
    # We check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)
    # and == 0 for all other index combinations where the product would be nonzero.
    
    # Build the full tensor check
    # The multiplication tensor M has M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition says: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for "invalid" triples (where the matrix multiplication doesn't connect),
    # the sum should be 0. But actually the tensor is only defined on the valid triples.
    
    # Actually, the standard tensor for matrix multiplication C = A*B where A is n×m, B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j,k) with T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # with U indexed by (i*m+j), V by (j*p+k), W by (k*n+i).
    
    # Check all valid (i,j,k) triples
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k])  # This isn't right
                # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # This is not a simple dot product of two vectors unless we combine.
                # Let's compute it directly.
                prod = U_arr[i*m+j] * V_arr[j*p+k]  # element-wise product, shape (rank,)
                val = np.dot(prod, W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    
    # Also check that the decomposition doesn't introduce spurious terms.
    # For the tensor to be exact, we need the decomposition to hold for ALL index combinations.
    # The "tensor" for matrix multiplication is: T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # and T is 1 when the indices are consistent (same j for a and b, same k for b and c, same i for a and c)
    # and 0 otherwise.
    
    # Let's check all combinations of (a, b, c) where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # The tensor value is 1 if j == j' and k == k' and i == i', else 0.
    # The decomposition gives: sum_r U[a,r] * V[b,r] * W[c,r]
    
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                expected = 1.0 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0.0
                prod = U_arr[a] * V_arr[b]
                val = np.dot(prod, W_arr[c])
                if abs(val - expected) > tol:
                    return False
    return True

def _try_rank(r, U, V, W):
    """Try to complete a partial decomposition of rank r."""
    n, m, p = DIMENSIONS
    # Count how many columns are fully determined (all entries nonzero or already set)
    # We'll use a greedy approach: for each column, if any of U, V, W has all zeros,
    # we need to determine it.
    
    # Actually, let's use a different approach: iterative refinement.
    # Start with random initialization and use ALS (alternating least squares)
    # to minimize the tensor reconstruction error.
    
    rng = np.random.RandomState(42 + r)
    
    # Initialize
    U_mat = np.zeros((n * m, r))
    V_mat = np.zeros((m * p, r))
    W_mat = np.zeros((p * n, r))
    
    # For columns that are already determined in the input, keep them.
    # The input U, V, W are lists of lists. If a column is all zeros, it's undetermined.
    # If it has some non-zero entries, it's partially determined.
    
    # Let's just do a fresh random init and ALS.
    U_mat = rng.randn(n * m, r) * 0.1
    V_mat = rng.randn(m * p, r) * 0.1
    W_mat = rng.randn(p * n, r) * 0.1
    
    # Build the tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # ALS iterations
    best_err = float('inf')
    best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
    
    for it in range(200):
        # Update U: for each (a, b, c), we have T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Fix V and W, solve for U.
        # For each a, we want: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c] for all (b,c)
        # This is a linear system: for each a, U[a,:] * (V ⊙ W)^T = T[a,:,:].reshape(-1)
        # where V ⊙ W is the element-wise product matrix of shape (m*p, p*n, r) -> (m*p*p*n, r)
        
        # Build the matrix M_U of shape (m*p*p*n, r) where M_U[b*(p*n)+c, r] = V[b,r]*W[c,r]
        VW = np.einsum('br,cr->bc r', V_mat, W_mat)  # shape (m*p, p*n, r)
        M_U = VW.reshape(m * p * p * n, r)
        T_U = T.reshape(n * m, m * p * p * n)
        
        # Solve: for each a, M_U^T * U[a,:] = T_U[a,:]
        # This is an underdetermined system (m*p*p*n >> r typically), use least squares
        # U[a,:] = (M_U^T M_U)^{-1} M_U^T T_U[a,:]
        # Or use pinv for stability.
        
        # Actually, let's use the normal equations: (M_U^T M_U) U[a,:]^T = M_U^T T_U[a,:]^T
        M_UtM_U = M_U.T @ M_U  # (r, r)
        M_UtT = T_U @ M_U  # (n*m, r)
        
        try:
            U_new = np.linalg.solve(M_UtM_U + 1e-8 * np.eye(r), M_UtT.T).T
        except:
            U_new = np.linalg.lstsq(M_U, T_U.T, rcond=None)[0].T
        
        U_mat = U_new
        
        # Update V
        UW = np.einsum('ar,cr->ac r', U_mat, W_mat)  # shape (n*m, p*n, r)
        M_V = UW.reshape(n * m * p * n, r)
        T_V = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
        
        M_VtM_V = M_V.T @ M_V
        M_VtT = T_V @ M_V
        
        try:
            V_new = np.linalg.solve(M_VtM_V + 1e-8 * np.eye(r), M_VtT.T).T
        except:
            V_new = np.linalg.lstsq(M_V, T_V.T, rcond=None)[0].T
        
        V_mat = V_new
        
        # Update W
        UV = np.einsum('ar,br->ab r', U_mat, V_mat)  # shape (n*m, m*p, r)
        M_W = UV.reshape(n * m * m * p, r)
        T_W = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
        
        M_WtM_W = M_W.T @ M_W
        M_WtT = T_W @ M_W
        
        try:
            W_new = np.linalg.solve(M_WtM_W + 1e-8 * np.eye(r), M_WtT.T).T
        except:
            W_new = np.linalg.lstsq(M_W, T_W.T, rcond=None)[0].T
        
        W_mat = W_new
        
        # Compute error
        # err = sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        err = np.sum((T - recon)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-6:
        # Convert to lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        # Verify
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
    return None

def construct():
    start_time = time.time()
    deadline = 130.0  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 3*4*6 = 72
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Also rank >= n*p + m - 1 for some cases, but let's just search.
    
    # Start from a reasonable lower bound
    # For 3x4 * 4x6, the rank is at least max(12, 24, 18) = 24
    # Let's try from 24 upward
    
    # First, return schoolbook as a fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)  # 24
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(r, None, None, None)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
