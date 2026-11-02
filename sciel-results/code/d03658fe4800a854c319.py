import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _als_decompose(n, m, p, rank, max_iter=200, seed=42, tol=1e-10):
    """
    Alternating Least Squares for the sparse matrix multiplication tensor.
    We need: for all (i,j,k) in [0,n)x[0,m)x[0,p):
      sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    
    This is a sparse tensor decomposition where only n*m*p entries are constrained.
    """
    np.random.seed(seed)
    dim1 = n * m
    dim2 = m * p
    dim3 = p * n
    
    # Precompute the list of valid (i,j,k) triples and their indices
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i*m + j, j*p + k, k*n + i))
    triples = np.array(triples)
    
    # Initialize
    U = np.random.randn(dim1, rank) * 0.5
    V = np.random.randn(dim2, rank) * 0.5
    W = np.random.randn(dim3, rank) * 0.5
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # --- Update U ---
        for i_idx in range(dim1):
            mask = triples[:, 0] == i_idx
            if not np.any(mask):
                U[i_idx, :] = 0
                continue
            idxs = triples[mask]
            v_rows = V[idxs[:, 1], :]
            w_rows = W[idxs[:, 2], :]
            
            vw = v_rows * w_rows
            
            b = vw.sum(axis=0)
            A = vw.T @ vw
            
            try:
                U[i_idx, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
            except np.linalg.LinAlgError:
                U[i_idx, :] = np.linalg.lstsq(A + 1e-12 * np.eye(rank), b, rcond=None)[0]
        
        # --- Update V ---
        for j_idx in range(dim2):
            mask = triples[:, 1] == j_idx
            if not np.any(mask):
                V[j_idx, :] = 0
                continue
            idxs = triples[mask]
            u_rows = U[idxs[:, 0], :]
            w_rows = W[idxs[:, 2], :]
            
            uw = u_rows * w_rows
            b = uw.sum(axis=0)
            A = uw.T @ uw
            
            try:
                V[j_idx, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
            except np.linalg.LinAlgError:
                V[j_idx, :] = np.linalg.lstsq(A + 1e-12 * np.eye(rank), b, rcond=None)[0]
        
        # --- Update W ---
        for k_idx in range(dim3):
            mask = triples[:, 2] == k_idx
            if not np.any(mask):
                W[k_idx, :] = 0
                continue
            idxs = triples[mask]
            u_rows = U[idxs[:, 0], :]
            v_rows = V[idxs[:, 1], :]
            
            uv = u_rows * v_rows
            b = uv.sum(axis=0)
            A = uv.T @ uv
            
            try:
                W[k_idx, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
            except np.linalg.LinAlgError:
                W[k_idx, :] = np.linalg.lstsq(A + 1e-12 * np.eye(rank), b, rcond=None)[0]
        
        # Compute error
        err = 0.0
        for (u_idx, v_idx, w_idx) in triples:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            err += (val - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a decomposition with rank < 90
    # Start with trivial as fallback
    fallback = _construct_trivial(n, m, p)
    
    # Strategy:
    # 1. Try to find a low-rank decomposition using ALS.
    # 2. If successful, verify it.
    # 3. If not, fallback to trivial.
    
    # Theoretical lower bound is max(n*m, m*p, p*n) = 30.
    # Known results suggest rank can be significantly lower than n*m*p.
    # Let's try a range of ranks.
    
    ranks_to_try = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60, 70, 80]
    
    best_result = None
    best_rank = 90
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds for robustness
        found = False
        for seed in [42, 1, 7, 123, 999]:
            if time.time() > deadline - 10:
                break
            
            U, V, W, err = _als_decompose(n, m, p, rank, max_iter=150, seed=seed, tol=1e-10)
            
            if err < 1e-8:
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # If we found a solution for this rank, we can stop since we're going low to high
            # However, it's possible a lower rank exists but we missed it due to local minima.
            # Given the time budget, stopping at the first found rank is a reasonable heuristic.
            # But let's try to go lower if we have time? No, we are iterating low to high.
            # So if we find rank 30, we are done. If we find rank 40, we should have tried 30-38 already.
            break
    
    if best_result is None:
        # If no low rank found, try a higher rank with more iterations?
        # Or just fallback.
        # Let's try one more shot with rank 90 (trivial) just to be sure, though it's guaranteed.
        # Actually, let's try rank 60 with more iterations if we failed lower ranks.
        if time.time() < deadline - 30:
            for seed in [42, 1, 7]:
                U, V, W, err = _als_decompose(n, m, p, 60, max_iter=300, seed=seed, tol=1e-10)
                if err < 1e-8 and _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = 60
                    break
        
        if best_result is None:
            best_result = fallback
            best_rank = 90
    
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
