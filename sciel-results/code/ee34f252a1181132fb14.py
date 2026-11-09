import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, T):
    I, J, K = T.shape
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
    # Check if entries are close to integers and match T
    if not np.allclose(reconstructed, np.round(reconstructed), atol=1e-9):
        return False
    return np.array_equal(np.round(reconstructed).astype(np.int64), T)

def _try_als_decomposition(T, rank, max_iterations=80, n_restarts=20, deadline=None, seed_base=42):
    """ALS with integer snapping for rank-r CP decomposition."""
    I, J, K = T.shape
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline:
            break
        
        np.random.seed(seed_base + restart * 1000)
        
        # Initialize with small integers
        U = np.random.randint(-3, 4, size=(I, rank))
        V = np.random.randint(-3, 4, size=(J, rank))
        W = np.random.randint(-3, 4, size=(K, rank))
        
        # Ensure non-trivial
        if np.all(U == 0):
            U[0, 0] = 1
        if np.all(V == 0):
            V[0, 0] = 1
        if np.all(W == 0):
            W[0, 0] = 1
        
        converged = False
        for iteration in range(max_iterations):
            if deadline and time.time() > deadline:
                break
            
            # Update U: for each i, solve min ||T[i] - U[i,:] * (V ⊙ W)||
            for i in range(I):
                # D has shape (J*K, rank), D[j*K+k, c] = V[j,c]*W[k,c]
                D = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(J * K, rank)
                target = T[i].flatten().astype(np.float64)
                try:
                    sol, _, _, _ = np.linalg.lstsq(D.T, target, rcond=None)
                    U[i, :] = np.round(sol)
                except:
                    pass
            
            # Update V
            for j in range(J):
                D = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(I * K, rank)
                target = T[:, j, :].flatten().astype(np.float64)
                try:
                    sol, _, _, _ = np.linalg.lstsq(D.T, target, rcond=None)
                    V[j, :] = np.round(sol)
                except:
                    pass
            
            # Update W
            for k in range(K):
                D = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(I * J, rank)
                target = T[:, :, k].flatten().astype(np.float64)
                try:
                    sol, _, _, _ = np.linalg.lstsq(D.T, target, rcond=None)
                    W[k, :] = np.round(sol)
                except:
                    pass
            
            # Check exactness
            T_recon = np.einsum('ic,jc,kc->ijk', U.astype(np.float64), V.astype(np.float64), W.astype(np.float64))
            if np.array_equal(np.round(T_recon).astype(np.int64), T):
                converged = True
                break
        
        if converged:
            return U.astype(int).tolist(), V.astype(int).tolist(), W.astype(int).tolist()
    
    return None

def _structured_initialization(n, m, p, rank):
    """Try to create a structured initialization based on the tensor structure."""
    I, J, K = n * m, m * p, p * n
    U = np.zeros((I, rank), dtype=int)
    V = np.zeros((J, rank), dtype=int)
    W = np.zeros((K, rank), dtype=int)
    
    # Use a subset of the schoolbook terms, randomly selecting which (i,j,k) triples to use
    # and allowing combinations
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
            if col >= rank:
                break
        if col >= rank:
            break
    
    return U, V, W

def _try_structured_search(n, m, p, rank, deadline=None):
    """Try structured initializations based on known patterns for matrix multiplication."""
    I, J, K = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # Strategy 1: Use all schoolbook terms but try to merge some
    # For 2x4x6, we have 48 schoolbook terms. Try to find a decomposition with rank < 48.
    
    # Try various structured patterns
    patterns = []
    
    # Pattern: Use Kronecker-like structure
    for a in range(1, min(n, m) + 1):
        for b in range(1, min(m, p) + 1):
            for c in range(1, min(p, n) + 1):
                if a * b * c >= rank:
                    continue
                patterns.append((a, b, c))
    
    for a, b, c in patterns:
        if deadline and time.time() > deadline:
            break
        
        # Create a structured initialization
        U = np.zeros((I, rank), dtype=float)
        V = np.zeros((J, rank), dtype=float)
        W = np.zeros((K, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    
                    # Assign based on modular structure
                    ia = i % a
                    ja = j % b
                    ka = k % c
                    
                    if ia == 0 and ja == 0 and ka == 0:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                    elif ia == 0 and ja == 0:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 0.5
                    elif ia == 0 and ka == 0:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 0.5
                        W[k * n + i, col] = 1.0
                    elif ja == 0 and ka == 0:
                        U[i * m + j, col] = 0.5
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                    
                    col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        if col >= rank:
            break
        
        # Verify
        if _verify_decomposition(U, V, W, T):
            return U.astype(int).tolist(), V.astype(int).tolist(), W.astype(int).tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    I, J, K = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((I, rank_schoolbook), dtype=int)
    V_school = np.zeros((J, rank_schoolbook), dtype=int)
    W_school = np.zeros((K, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Try ranks from 47 down to 30
    for rank in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline:
            break
        
        # First try structured search
        result = _try_structured_search(n, m, p, rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, T):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        # Then try ALS with multiple restarts
        result = _try_als_decomposition(T, rank, max_iterations=50, n_restarts=15, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, T):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
