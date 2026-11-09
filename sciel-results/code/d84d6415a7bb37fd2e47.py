"""Matrix multiplication 3x4x8 rank-minimal tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

def _trivial_decomp(n, m, p):
    """Return the trivial rank n*m*p decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * (m * p) + k * p + j
                U[i * m + k, r] = 1.0
                V[k * p + j, r] = 1.0
                W[j * n + i, r] = 1.0
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition is exact."""
    rank = U.shape[1]
    # Reshape U into (n, m, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # The tensor is T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # We need this to equal 1 if the (i,k,j) combination is valid, else 0.
    # For matrix multiplication C = A @ B where A is n x m, B is m x p, C is n x p:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # In tensor form: T[i,k,j] = 1 for all valid (i,k,j), 0 otherwise.
    # Actually, the tensor for matrix multiplication has T[i,k,j] = 1 for all i,k,j
    # (since every (i,k,j) triplet contributes to the product).
    
    # Let's verify: C[i,j] should equal sum_r (sum_k U[i,k,r]*A[i,k]) * (sum_k V[k,j,r]*B[k,j]) * W[j,i,r]
    # Wait, the standard tensor decomposition for matrix mult is:
    # C[i,j] = sum_r U[i,k,r] A[i,k] V[k,j,r] B[k,j] W[j,i,r]
    # No, that's not right either. Let me think again.
    # 
    # The matrix multiplication tensor T has T[i,k,j] = 1.
    # A rank-1 decomposition is: T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # where W is indexed by (j,i) because the output C is n x p, indexed by (i,j).
    # Actually, the convention says W[k*n+i, r] where k is column (j) and i is row.
    # So W is p x n, and W[j,i,r] is the (j,i,r) entry.
    
    # Let's just check: for random A, B, compute C = A @ B, then compute
    # C' from the decomposition and check they match.
    
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute from decomposition
    # U[i*m + k, r] * A[i,k] * V[k*p + j, r] * B[k,j] * W[j*n + i, r]
    C_computed = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    C_computed[i, j] += U[i*m + k, r] * A[i, k] * V[k*p + j, r] * B[k, j] * W[j*n + i, r]
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    rank = U.shape[1]
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # U reshaped to (n, m, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # C[i,j] = sum_r (sum_k U[i,k,r] * A[i,k]) * (sum_k V[k,j,r] * B[k,j]) * W[j,i,r]
    # This is not quite right. Let me re-derive.
    #
    # Actually, the standard form is:
    # C[i,j] = sum_{r=1}^{R} sum_{k=1}^{m} U[i,k,r] A[i,k] V[k,j,r] B[k,j] W[j,i,r]
    #
    # But this has a double sum over k and r, which is unusual. Let me check the seed code.
    # The seed code sets U[i*m+k, r] = 1, V[k*p+j, r] = 1, W[j*n+i, r] = 1
    # for r = i*m*p + k*p + j.
    # So for each (i,k,j), there's exactly one r where U, V, W are all 1.
    # C[i,j] = sum_r U[i*m+k(r), r] * A[i,k(r)] * V[k(r)*p+j(r), r] * B[k(r),j(r)] * W[j(r)*n+i(r), r]
    # = sum_{i,k,j} A[i,k] * B[k,j] * (indicator that r corresponds to (i,k,j))
    # = sum_k A[i,k] * B[k,j]
    # = (A @ B)[i,j]
    #
    # So the formula is: C[i,j] = sum_r U[i*m+k(r), r] * A[i,k(r)] * V[k(r)*p+j(r), r] * B[k(r),j(r)] * W[j(r)*n+i(r), r]
    # But k(r) and j(r) are implicit in the indices. We can't directly vectorize this easily.
    #
    # Alternative: think of it as
    # C[i,j] = sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * A[i,k] * ... )
    # No, that doesn't work either.
    #
    # Let me just use the slow verification for small dimensions. 3*4*8 = 96, rank up to 96.
    # 96*96*3*4*8 = ~880k ops per verification, which is fine.
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            s = 0.0
            for r in range(rank):
                w_val = W[j * n + i, r]
                if w_val == 0:
                    continue
                # We need to find which k this r corresponds to, but we don't know it.
                # Instead, let's sum over k:
                for k in range(m):
                    s += U[i * m + k, r] * A[i, k] * V[k * p + j, r] * B[k, j] * w_val
            C_computed[i, j] = s
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer (within tolerance)."""
    if x is None or np.isnan(x):
        return False
    x = float(x)
    return abs(x * 2 - round(x * 2)) < 1e-6

def _all_half_integers(mat, tol=1e-6):
    """Check if all entries in mat are integers or half-integers."""
    for val in mat.flat:
        if not _is_half_integer(val):
            return False
    return True

def _snap_to_half_integer(x):
    """Snap x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _snap_matrix(mat):
    """Snap all entries to nearest half-integer."""
    return np.array([[_snap_to_half_integer(x) for x in row] for row in mat])

def _rank_of_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Compute the actual rank of the decomposition (number of nonzero terms)."""
    rank = U.shape[1]
    count = 0
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        # A term is nonzero if any of the columns is nonzero
        if np.any(np.abs(u_col) > tol) or np.any(np.abs(v_col) > tol) or np.any(np.abs(w_col) > tol):
            count += 1
    return count

def _try_reduction(U, V, W, n, m, p, tol=1e-10):
    """Try to reduce the rank by identifying and removing redundant terms."""
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find pairs of terms that can be merged or cancelled
    for r in range(rank):
        if not _verify(U, V, W, n, m, p, tol):
            break
        # Check if removing term r still gives a valid decomposition
        # (This is expensive, so we only do it for small ranks)
        if rank <= 50:
            U_no_r = np.delete(U, r, axis=1)
            V_no_r = np.delete(V, r, axis=1)
            W_no_r = np.delete(W, r, axis=1)
            if _verify(U_no_r, V_no_r, W_no_r, n, m, p, tol):
                best_rank = rank - 1
                best_U, best_V, best_W = U_no_r, V_no_r, W_no_r
                # Continue trying to remove more
                break
    
    return best_U, best_V, best_W, best_rank

def _random_search(U0, V0, W0, n, m, p, max_iter=2000, time_budget=30):
    """Random search for better decompositions."""
    t_start = time.time()
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = U0.shape[1]
    
    for it in range(max_iter):
        if time.time() - t_start > time_budget:
            break
        # Perturb the current best
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        
        # Random perturbation
        U += np.random.randn(*U.shape) * 0.1
        V += np.random.randn(*V.shape) * 0.1
        W += np.random.randn(*W.shape) * 0.1
        
        # Snap to half-integers
        U = _snap_matrix(U)
        V = _snap_matrix(V)
        W = _snap_matrix(W)
        
        if _verify(U, V, W, n, m, p, 1e-8):
            r = _rank_of_decomposition(U, V, W, n, m, p)
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U, best_V, best_W, best_rank

def _structured_search(n, m, p, time_budget=60):
    """Structured search for low-rank decompositions."""
    t_start = time.time()
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Start from the trivial decomposition
    U, V, W = _trivial_decomp(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find a better decomposition using random initialization
    for trial in range(10):
        if time.time() - t_start > time_budget:
            break
        
        # Try various ranks
        for rank in range(10, min(40, n * m * p + 1), 2):
            if time.time() - t_start > time_budget:
                break
            
            U_rand = np.random.randn(n * m, rank) * 0.5
            V_rand = np.random.randn(m * p, rank) * 0.5
            W_rand = np.random.randn(p * n, rank) * 0.5
            
            # Snap to half-integers
            U_rand = _snap_matrix(U_rand)
            V_rand = _snap_matrix(V_rand)
            W_rand = _snap_matrix(W_rand)
            
            if _verify(U_rand, V_rand, W_rand, n, m, p, 1e-8):
                r = _rank_of_decomposition(U_rand, V_rand, W_rand, n, m, p)
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
    
    return best_U, best_V, best_W, best_rank

def _gradient_descent(U, V, W, n, m, p, max_iter=100, time_budget=30):
    """Gradient descent to minimize the Frobenius norm of the tensor residual."""
    t_start = time.time()
    rank = U.shape[1]
    
    # Define the objective: sum of squared residuals
    # T[i,k,j] should be 1 for all i,k,j
    # T_computed[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    A = np.ones((n, m))
    B = np.ones((m, p))
    
    def objective(x):
        U_flat = x[:n*m*rank].reshape(n*m, rank)
        V_flat = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute C = A @ B and C from decomposition
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                s = 0.0
                for r in range(rank):
                    w_val = W_flat[j * n + i, r]
                    if w_val == 0:
                        continue
                    for k in range(m):
                        s += U_flat[i * m + k, r] * A[i, k] * V_flat[k * p + j, r] * B[k, j] * w_val
                C_computed[i, j] = s
        return np.sum((C_computed - C_expected) ** 2)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Try to optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() - t_start > time_budget:
            return U, V, W, rank
        
        x_opt = res.x
        U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
        V_opt = x_opt[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Snap to half-integers
        U_opt = _snap_matrix(U_opt)
        V_opt = _snap_matrix(V_opt)
        W_opt = _snap_matrix(W_opt)
        
        if _verify(U_opt, V_opt, W_opt, n, m, p, 1e-8):
            r = _rank_of_decomposition(U_opt, V_opt, W_opt, n, m, p)
            if r < rank:
                return U_opt, V_opt, W_opt, r
    except Exception:
        pass
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 120.0  # 120 second limit
    
    # Start with the trivial decomposition
    U, V, W = _trivial_decomp(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Phase 1: Try structured random search
    time_left = deadline - time.time() - 5.0  # 5 second safety margin
    if time_left > 10:
        U, V, W, r = _structured_search(n, m, p, time_budget=time_left)
        if r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U, V, W
    
    # Phase 2: Try reduction
    time_left = deadline - time.time() - 5.0
    if time_left > 5:
        U, V, W, r = _try_reduction(best_U, best_V, best_W, n, m, p)
        if r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U, V, W
    
    # Phase 3: Random search from current best
    time_left = deadline - time.time() - 5.0
    if time_left > 5:
        U, V, W, r = _random_search(best_U, best_V, best_W, n, m, p, max_iter=500, time_budget=time_left)
        if r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U, V, W
    
    # Convert to list of lists for the return format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
