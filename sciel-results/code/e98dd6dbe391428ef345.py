"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Check shapes
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_decomposition():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    
    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 4 (schoolbook for that column)
    
    Total: 5*7 + 4 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]
        
        w_c00 = k0 * n + 0   # C[0, k0]
        w_c01 = k1 * n + 0   # C[0, k1]
        w_c10 = k0 * n + 1   # C[1, k0]
        w_c11 = k1 * n + 1   # C[1, k1]
        
        u_a = 0  # A[0,0]
        u_b = 1  # A[0,1]
        u_c = 2  # A[1,0]
        u_d = 3  # A[1,1]
        
        # m1 = (a+d)(e+h)
        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m2 = (c+d)e
        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1
        
        # m3 = a(f-h)
        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m4 = d(g-e)
        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1
        
        # m5 = (a+b)h
        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1
    
    # Handle remainder column (column 10)
    if remainder == 1:
        k2 = num_pairs * 2  # = 10
        
        v_e2 = k2        # B[0, 10]
        v_g2 = p + k2    # B[1, 10]
        w_c02 = k2 * n + 0   # C[0, 10]
        w_c12 = k2 * n + 1   # C[1, 10]
        
        # Term 1: A[0,0]*B[0,10]
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        # Term 2: A[0,1]*B[1,10]
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        # Term 3: A[1,0]*B[0,10]
        U[2, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
        
        # Term 4: A[1,1]*B[1,10]
        U[3, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _fast_alternating_minimization(n, m, p, rank, max_iters, deadline, seed):
    """
    Fast alternating minimization to find a rank-r decomposition.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize U, V, W
    # Using a structured initialization might help, but random is standard.
    # We scale to avoid overflow/underflow initially.
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Precompute indices for the tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # For each a, we need the list of (b, c)
    # For each b, we need the list of (a, c)
    # For each c, we need the list of (a, b)
    
    a_to_bc = [[] for _ in range(n * m)]
    b_to_ac = [[] for _ in range(m * p)]
    c_to_ab = [[] for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                a_to_bc[a].append((b, c))
                b_to_ac[b].append((a, c))
                c_to_ab[c].append((a, b))
    
    # Convert to numpy arrays for faster access if possible, but lists are fine for small sizes
    # Actually, since n*m is small (4), we can just loop.
    
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        
        # Update U
        for a in range(n * m):
            if not a_to_bc[a]:
                continue
            # Sum over (b, c)
            # U[a, r] = sum(V[b, r] * W[c, r]) / sum((V[b, r] * W[c, r])^2)
            num = np.zeros(rank)
            den = np.zeros(rank)
            for b, c in a_to_bc[a]:
                prod = V[b] * W[c]
                num += prod
                den += prod * prod
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            if not b_to_ac[b]:
                continue
            num = np.zeros(rank)
            den = np.zeros(rank)
            for a, c in b_to_ac[b]:
                prod = U[a] * W[c]
                num += prod
                den += prod * prod
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c in range(p * n):
            if not c_to_ab[c]:
                continue
            num = np.zeros(rank)
            den = np.zeros(rank)
            for a, b in c_to_ab[c]:
                prod = U[a] * V[b]
                num += prod
                den += prod * prod
            valid = den > 1e-12
            if np.any(valid):
                W[c, valid] = num[valid] / den[valid]
        
        # Check convergence every 20 iterations
        if it % 20 == 0:
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = np.dot(U[a], V[b] * W[c])
                        err += (s - 1.0) ** 2
            if err < 1e-10:
                return U, V, W
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, leave margin before 150s limit
    
    T = _build_target_tensor()
    
    # 1. Build the guaranteed rank-39 solution
    U39, V39, W39 = _build_strassen_decomposition()
    
    if not _verify(U39, V39, W39, T):
        # Fallback to schoolbook if Strassen fails (shouldn't happen)
        U, V, W = _build_schoolbook()
        if _verify(U, V, W, T):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        raise RuntimeError("No valid decomposition found")
    
    best_rank = 39
    best_solution = (U39, V39, W39)
    
    # 2. Try to find lower rank solutions (38, 37, ...)
    # We have about 140 seconds. Building Strassen is fast.
    # Let's allocate time to search.
    
    n, m, p = 2, 2, 11
    target_ranks = [38, 37, 36]
    
    # Estimate time per search
    # We'll try a few seeds for each rank.
    
    for rank in target_ranks:
        if time.time() > DEADLINE - 10:
            break
            
        # Try a few seeds
        for seed in [rank, rank + 1, rank + 100, rank + 200]:
            if time.time() > DEADLINE - 5:
                break
            
            # Run search with a time limit
            # We don't know how long it takes, so we run it until deadline or convergence
            # To be safe, we can run multiple short searches or one long one.
            # Let's run one search per seed with a sub-deadline.
            sub_deadline = DEADLINE - 2
            result = _fast_alternating_minimization(n, m, p, rank, max_iters=10000, deadline=sub_deadline, seed=seed)
            
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, T):
                    best_rank = rank
                    best_solution = (U, V, W)
                    # If we found 36, we can stop. If 37, maybe try 36.
                    # The loop structure handles this by continuing to lower ranks.
                    break # Break seed loop, try next rank
    
    U, V, W = best_solution
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
