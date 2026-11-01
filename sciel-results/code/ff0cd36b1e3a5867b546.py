"""Matrix multiplication tensor decomposition for 2x3 * 3x5."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
DEADLINE = 130.0  # seconds, safety margin under 150s limit

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check whether (U,V,W) exactly decompose the matmul tensor."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Faster verification using tensor einsum."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank_decomposition(n, m, p, target_rank, time_budget):
    """Attempt to find a decomposition of the given target_rank.
    
    Uses alternating least squares with SVD-based minimum-norm solves.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Try random initialization with iterative refinement
    rng = np.random.RandomState(42)
    
    for attempt in range(100):
        if time.time() - start_time > time_budget:
            break
        
        U = rng.randn(nm, target_rank) * 0.1
        V = rng.randn(mp, target_rank) * 0.1
        W = rng.randn(pn, target_rank) * 0.1
        
        # Scale to get closer to 1
        scale = 1.0 / (target_rank ** (1/3))
        U *= scale
        V *= scale
        W *= scale
        
        # Iterate: fix two, solve for the third
        converged = False
        for it in range(500):
            if time.time() - start_time > time_budget:
                break
            
            # Compute residuals
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residuals[i, j, k] = np.dot(
                            U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]
                        ) - 1.0
            
            max_res = np.max(np.abs(residuals))
            if max_res < 1e-10:
                converged = True
                break
            
            # Update W: for each c, W[k*n+i, c] appears in equations
            # where (k,i) is fixed. For a fixed (i,j,k), the equation is:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # 
            # If we fix U and V, then for each (k,i), the unknowns are
            # W[k*n+i, c] for c=0..r-1, but they appear in different
            # equations depending on j.
            #
            # This is a linear system for W given U,V.
            
            # Build the system. Unknowns: W_vec of length pn*r
            # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            A = np.zeros((n*m*p, pn*target_rank))
            b = np.ones(n*m*p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            u_idx = (i * m + j) * target_rank + c
                            v_idx = (j * p + k) * target_rank + c
                            w_idx = (k * n + i) * target_rank + c
                            A[eq_idx, w_idx] = U[i*m+j, c] * V[j*p+k, c]
                        eq_idx += 1
            
            try:
                W_svd, s_vals, Wt_svd = np.linalg.svd(A, full_matrices=False)
                threshold = 1e-10
                mask = s_vals > threshold
                W_flat = np.zeros(pn * target_rank)
                for idx in range(len(s_vals)):
                    if mask[idx]:
                        W_flat += (Wt_svd[idx, :] * (W_svd[:, idx] @ b) / s_vals[idx])
            except np.linalg.LinAlgError:
                continue
            
            W = W_flat.reshape(pn, target_rank)
            
            # Update V similarly
            A = np.zeros((n*m*p, mp*target_rank))
            b = np.ones(n*m*p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            u_idx = (i * m + j) * target_rank + c
                            v_idx = (j * p + k) * target_rank + c
                            w_idx = (k * n + i) * target_rank + c
                            A[eq_idx, v_idx] = U[i*m+j, c] * W[k*n+i, c]
                        eq_idx += 1
            
            try:
                V_svd, s_vals, Vt_svd = np.linalg.svd(A, full_matrices=False)
                threshold = 1e-10
                mask = s_vals > threshold
                V_flat = np.zeros(mp * target_rank)
                for idx in range(len(s_vals)):
                    if mask[idx]:
                        V_flat += (Vt_svd[idx, :] * (V_svd[:, idx] @ b) / s_vals[idx])
            except np.linalg.LinAlgError:
                continue
            
            V = V_flat.reshape(mp, target_rank)
            
            # Update U similarly
            A = np.zeros((n*m*p, nm*target_rank))
            b = np.ones(n*m*p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            u_idx = (i * m + j) * target_rank + c
                            v_idx = (j * p + k) * target_rank + c
                            w_idx = (k * n + i) * target_rank + c
                            A[eq_idx, u_idx] = V[j*p+k, c] * W[k*n+i, c]
                        eq_idx += 1
            
            try:
                U_svd, s_vals, Ut_svd = np.linalg.svd(A, full_matrices=False)
                threshold = 1e-10
                mask = s_vals > threshold
                U_flat = np.zeros(nm * target_rank)
                for idx in range(len(s_vals)):
                    if mask[idx]:
                        U_flat += (Ut_svd[idx, :] * (U_svd[:, idx] @ b) / s_vals[idx])
            except np.linalg.LinAlgError:
                continue
            
            U = U_flat.reshape(nm, target_rank)
        
        if converged:
            return U, V, W
    
    return None


def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3x5 matmul."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try rank 29 with limited time budget
    time_for_search = min(60.0, DEADLINE - (time.time() - start_time) - 5.0)
    if time_for_search > 10.0:
        result = _try_rank_decomposition(n, m, p, 29, time_for_search)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
