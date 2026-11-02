import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    if not U or not V or not W:
        return False
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check all required entries
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                if abs(val - 1) > tol:
                    return False
    # Check a sample of non-required entries to ensure they're 0
    rng = np.random.RandomState(42)
    for _ in range(200):
        u_idx = rng.randint(n * m)
        v_idx = rng.randint(m * p)
        w_idx = rng.randint(p * n)
        i, k = divmod(u_idx, m)
        k2, j = divmod(v_idx, p)
        j2, i2 = divmod(w_idx, n)
        if k != k2 or j != j2 or i != i2:
            val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
            if abs(val) > tol:
                return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_strip_decomposition(n, m, p):
    """
    Partition the p columns into strips and try to find low-rank
    decompositions for each strip, then concatenate.
    
    For 4x4 x 4x9, split into three 4x4 x 4x3 strips.
    For each 4x4 x 4x3 sub-problem, try to find a rank-r decomposition.
    """
    start_time = time.time()
    deadline = start_time + 120
    
    n_strips = 3
    cols_per_strip = 3  # 9 / 3 = 3
    
    all_U = []
    all_V = []
    all_W = []
    
    for s in range(n_strips):
        col_offset = s * cols_per_strip
        
        # Build the target tensor for this strip
        # U indices: i*m + k for i in [0,n), k in [0,m)
        # V indices: k*p + j for k in [0,m), j in [col_offset, col_offset+cols_per_strip)
        # W indices: j*n + i for j in [col_offset, col_offset+cols_per_strip), i in [0,n)
        
        # Try to find a low-rank decomposition for this strip
        # Start with schoolbook rank for the strip: n*m*cols_per_strip = 4*4*3 = 48
        # Try to find lower rank
        
        strip_rank_found = None
        best_strip = None
        
        # Try ranks from 36 down to 48 (schoolbook)
        for target_rank in range(36, 49):
            if time.time() > deadline:
                break
            found = _find_strip_decomposition(n, m, cols_per_strip, target_rank, 
                                               col_offset, p, timeout=15)
            if found is not None:
                strip_rank_found = target_rank
                best_strip = found
                break
        
        if best_strip is None:
            # Fall back to schoolbook for this strip
            best_strip = _schoolbook_strip(n, m, cols_per_strip, col_offset, p)
            strip_rank_found = n * m * cols_per_strip
        
        all_U.extend(best_strip['U'])
        all_V.extend(best_strip['V'])
        all_W.extend(best_strip['W'])
    
    # Combine
    U = np.array(all_U)
    V = np.array(all_V)
    W = np.array(all_W)
    
    # Verify the full decomposition
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _schoolbook_strip(n, m, cols_per_strip, col_offset, p):
    """Schoolbook decomposition for a single strip of columns."""
    rank = n * m * cols_per_strip
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for dj in range(cols_per_strip):
                j = col_offset + dj
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_strip_decomposition(n, m, cols_per_strip, target_rank, col_offset, p, timeout=15):
    """
    Try to find a rank-target_rank decomposition for a 4x4 x 4xcols_per_strip
    multiplication using random search with structured initialization.
    """
    start = time.time()
    Nm = n * m
    Mp = m * p
    Pn = p * n
    
    # The required constraints are:
    # For each (i, k, j) where j is in [col_offset, col_offset+cols_per_strip):
    #   sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] = 1
    # For all other combinations, the sum should be 0.
    
    # Use random initialization and check
    for trial in range(50):
        if time.time() - start > timeout:
            break
        rng = np.random.RandomState(trial * 7919 + 12345)
        
        U = rng.randn(Nm, target_rank) * 0.5
        V = rng.randn(Mp, target_rank) * 0.5
        W = rng.randn(Pn, target_rank) * 0.5
        
        if _verify_strip(U, V, W, n, m, cols_per_strip, col_offset, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _verify_strip(U, V, W, n, m, cols_per_strip, col_offset, p, tol=1e-6):
    """Verify a strip decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check required entries
    for i in range(n):
        for k in range(m):
            for dj in range(cols_per_strip):
                j = col_offset + dj
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                if abs(val - 1) > tol:
                    return False
    
    # Check non-required entries (sample)
    rng = np.random.RandomState(42)
    for _ in range(300):
        u_idx = rng.randint(n * m)
        v_idx = rng.randint(m * p)
        w_idx = rng.randint(p * n)
        i, k = divmod(u_idx, m)
        k2, j = divmod(v_idx, p)
        j2, i2 = divmod(w_idx, n)
        # This entry should be 1 only if k==k2, j==j2, i==i2, and j is in the strip
        if k == k2 and j == j2 and i == i2 and col_offset <= j < col_offset + cols_per_strip:
            continue  # Already checked
        val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
        if abs(val) > tol:
            return False
    
    return True

def _try_direct_random(n, m, p, target_rank, num_trials=20, timeout=20):
    """Try direct random search for full decomposition."""
    start = time.time()
    Nm, Mp, Pn = n * m, m * p, p * n
    for trial in range(num_trials):
        if time.time() - start > timeout:
            break
        rng = np.random.RandomState(trial * 31337 + 999)
        U = rng.randn(Nm, target_rank) * 0.3
        V = rng.randn(Mp, target_rank) * 0.3
        W = rng.randn(Pn, target_rank) * 0.3
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 144
    
    # Try strip decomposition approach
    strip_result = _construct_strip_decomposition(n, m, p)
    if strip_result is not None:
        strip_rank = len(strip_result['U'][0])
        if strip_rank < best_rank:
            best = strip_result
            best_rank = strip_rank
    
    # Try direct random search for even lower rank
    if time.time() < deadline - 30:
        for target_rank in range(best_rank - 1, max(best_rank - 20, 100), -1):
            if time.time() > deadline - 10:
                break
            result = _try_direct_random(n, m, p, target_rank, num_trials=5, timeout=5)
            if result is not None:
                best = result
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
