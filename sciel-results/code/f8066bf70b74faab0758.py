"""Rewritten program: MILP-style greedy sparse basis selection for 4x5x6 matrix multiplication tensor."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 112.0

def _time_left():
    return _DEADLINE - time.time()

def _schoolbook_decomp(n, m, p):
    """Schoolbook decomposition for n x m @ m x p (rank n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-4):
    """Fast vectorized check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape approach:
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # We want T[i,j,k] for all i in [0,n), j in [0,m), k in [0,p)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Build tensor via einsum
    # U reshaped: (n, m, R)
    # V reshaped: (m, p, R)
    # W reshaped: (p, n, R)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _greedy_sparse_decomp(n, m, p, max_rank=80):
    """
    Greedy algorithm to find a low-rank decomposition.
    
    We model the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    Strategy: Start with a set of candidate "basis" tensors. Each basis tensor is
    defined by a triple of index patterns. We greedily select basis tensors and
    solve for their coefficients.
    
    Actually, a cleaner approach: we search over decompositions where each rank-1
    term has a specific sparse structure.
    
    Let's use a different approach: iterative refinement with random restarts.
    Start with a random decomposition of some rank R, then try to reduce rank
    by finding linear dependencies.
    """
    if _time_left() < 5.0:
        return _schoolbook_decomp(n, m, p)
    
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook_decomp(n, m, p)
    
    # Try various ranks from low to high
    for R in range(36, min(max_rank, n * m * p + 1)):
        if _time_left() < 10.0:
            break
        
        # Try to find a decomposition of rank R
        # Use iterative optimization
        found = _try_rank_decomp(n, m, p, R, max_iters=200)
        if found is not None:
            U, V, W = found
            if _verify_decomp(U, V, W, n, m, p, 1e-8):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break
        if _time_left() < 15.0:
            break
    
    return best_U, best_V, best_W

def _try_rank_decomp(n, m, p, R, max_iters=200):
    """
    Try to find a rank-R decomposition using iterative optimization.
    
    We parameterize:
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    The constraint is: for all i,j,k:
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    This is a system of n*m*p = 120 equations in 3*R*(n*m + m*p + p*n) unknowns.
    For R=36: 3*36*(20+30+24) = 3*36*74 = 7992 unknowns, 120 equations.
    
    We use a greedy approach: fix two of the three matrices, solve for the third.
    """
    if _time_left() < 5.0:
        return None
    
    # Initialize with random values
    np.random.seed(42)
    
    for attempt in range(5):
        if _time_left() < 5.0:
            return None
        
        # Random initialization
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Iterative refinement
        for it in range(max_iters):
            if _time_left() < 5.0:
                return None
            
            # Solve for W given U, V
            # For each (k,i): sum_r W[k*n+i,r] * (sum over j of U[i*m+j,r]*V[j*p+k,r]) = 1
            # Let A[k,i,r] = sum_j U[i*m+j,r] * V[j*p+k,r]
            # Then we need: sum_r W[k*n+i,r] * A[k,i,r] = 1 for all (k,i)
            
            # Build A: shape (p, n, R)
            A = np.zeros((p, n, R))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        A[k, i, :] += U[i * m + j, :] * V[j * p + k, :]
            
            # Now solve: for each (k,i), W[k*n+i,:] . A[k,i,:] = 1
            # This is an underdetermined system (R > 1 usually)
            # Use least squares with minimum norm
            W_new = np.zeros((p * n, R))
            for k in range(p):
                for i in range(n):
                    a = A[k, i, :]
                    if np.linalg.norm(a) < 1e-10:
                        # Cannot satisfy this equation
                        return None
                    # Minimum norm solution to W . a = 1
                    W_new[k * n + i, :] = a / (a @ a)
            
            W = W_new
            
            # Solve for U given V, W
            # For each (i,j): sum_r U[i*m+j,r] * (sum over k of V[j*p+k,r]*W[k*n+i,r]) = 1
            B = np.zeros((n, m, R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        B[i, j, :] += V[j * p + k, :] * W[k * n + i, :]
            
            U_new = np.zeros((n * m, R))
            for i in range(n):
                for j in range(m):
                    b = B[i, j, :]
                    if np.linalg.norm(b) < 1e-10:
                        return None
                    U_new[i * m + j, :] = b / (b @ b)
            
            U = U_new
            
            # Solve for V given U, W
            # For each (j,k): sum_r V[j*p+k,r] * (sum over i of U[i*m+j,r]*W[k*n+i,r]) = 1
            C = np.zeros((m, p, R))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        C[j, k, :] += U[i * m + j, :] * W[k * n + i, :]
            
            V_new = np.zeros((m * p, R))
            for j in range(m):
                for k in range(p):
                    c = C[j, k, :]
                    if np.linalg.norm(c) < 1e-10:
                        return None
                    V_new[j * p + k, :] = c / (c @ c)
            
            V = V_new
            
            # Check convergence
            if it % 10 == 0:
                if _verify_np(U, V, W, n, m, p, 1e-6):
                    if _verify_decomp(U, V, W, n, m, p, 1e-8):
                        return U, V, W
        
        # Final check
        if _verify_np(U, V, W, n, m, p, 1e-6):
            if _verify_decomp(U, V, W, n, m, p, 1e-8):
                return U, V, W
    
    return None

def _local_reduce(U, V, W, n, m, p, max_iters=10, budget_s=20.0):
    """Local column-removal search."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        for c in range(R):
            if _time_left() < 3.0:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
                    found = True
                    break
        if not found:
            break
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Fallback: schoolbook (always valid, rank 120)
    U, V, W = _schoolbook_decomp(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try greedy sparse decomposition
    try:
        if _time_left() > 20.0:
            cand = _greedy_sparse_decomp(n, m, p, max_rank=70)
            if cand[1].shape[1] < best_rank and _verify_decomp(*cand, n, m, p, 1e-9):
                best_rank = cand[1].shape[1]
                best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Local column-removal refinement
    try:
        if _time_left() > 10.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=10, budget_s=15.0)
            if cand[1].shape[1] < best_rank and _verify_decomp(*cand, n, m, p, 1e-9):
                best_rank = cand[1].shape[1]
                best_U, best_V, best_W = cand
    except Exception:
        pass
    
    if _time_left() < 3.0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _schoolbook_decomp(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
