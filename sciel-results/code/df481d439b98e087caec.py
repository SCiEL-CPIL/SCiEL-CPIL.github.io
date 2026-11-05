import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen decompositions. Rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    r2 = 7
    r = r2 * r2  # 49
    
    U2_3d = U2.reshape(2, 2, r2)
    V2_3d = V2.reshape(2, 2, r2)
    W2_3d = W2.reshape(2, 2, r2)
    
    U4_5d = np.einsum('abk,cdl->acbdkl', U2_3d, U2_3d)
    U4 = U4_5d.reshape(16, 49).astype(np.int16)
    V4 = np.einsum('abk,cdl->acbdkl', V2_3d, V2_3d).reshape(16, 49).astype(np.int16)
    W4 = np.einsum('abk,cdl->acbdkl', W2_3d, W2_3d).reshape(16, 49).astype(np.int16)
    
    return U4, V4, W4

def _build_4x4x8_direct():
    """
    Build 4x4x8 decomposition directly from two 4x4x4 blocks.
    Uses the direct sum structure: 4x4x8 = 4x4x4 ⊕ 4x4x4
    Rank = 49 + 49 = 98
    """
    U4, V4, W4 = _build_4x4x4_strassen()
    r4 = 49
    total_rank = 2 * r4  # 98
    
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: columns 0-48
    U[:, 0:r4] = U4
    # For V: row index j*8 + k where k in 0..3 (first half of 8 columns)
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:r4] = V4[j * 4 + k, :]
    # For W: row index k*4 + i where k in 0..3 (first half)
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:r4] = W4[k * 4 + i, :]
    
    # Block 1: columns 49-97
    U[:, r4:total_rank] = U4
    # For V: row index j*8 + k where k in 4..7 (second half of 8 columns)
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), r4:total_rank] = V4[j * 4 + k, :]
    # For W: row index k*4 + i where k in 4..7 (second half)
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, r4:total_rank] = W4[k * 4 + i, :]
    
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8, seed=42, tol=1e-9):
    """Verify the tensor decomposition on random test matrices."""
    np.random.seed(seed)
    A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
    B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    rank = U.shape[1]
    C_comp = np.zeros((n, p), dtype=np.float64)
    for r in range(rank):
        Ar = A * U[:, r].reshape(n, m)
        Br = B * V[:, r].reshape(m, p)
        Mr = Ar @ Br
        Wr = W[:, r].reshape(p, n).T
        C_comp += Mr * Wr
    err = np.max(np.abs(C_comp - C_expected))
    return err < tol, err

def _verify_multi_seed(U, V, W, n=4, m=4, p=8, seeds=None, tol=1e-9):
    """Verify with multiple random seeds for robustness."""
    if seeds is None:
        seeds = [42, 123, 999, 7, 12345]
    for seed in seeds:
        ok, err = _verify(U, V, W, n=n, m=m, p=p, seed=seed, tol=tol)
        if not ok:
            return False, err
    return True, 0.0

def _try_rank_reduction_greedy(U, V, W, n=4, m=4, p=8, time_budget=60.0):
    """
    Greedy rank reduction: try to remove one rank-1 component at a time
    by checking if the remaining components still span the tensor.
    Uses a linear algebra approach: check if the removed component's
    contribution can be absorbed by others.
    """
    start_time = time.time()
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    rank = U_f.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
    
    # Build the full tensor in vectorized form for efficiency
    # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We'll work with a flattened representation
    
    n, m, p = 4, 4, 8
    dim1 = n * m  # 16
    dim2 = m * p  # 32
    dim3 = p * n  # 32
    
    # Try removing each rank component and see if we can compensate
    # Strategy: for each rank r, try to express its contribution as a combination of others
    # This is a linear system: find alpha_s such that
    # U[:,r] * V[:,r] * W[:,r] ≈ sum_s alpha_s * U[:,s] * V[:,s] * W[:,s]
    
    # For efficiency, use a small number of test points
    np.random.seed(42)
    test_A = np.random.randn(n, m)
    test_B = np.random.randn(m, p)
    C_true = test_A @ test_B
    
    # Precompute contributions of each rank component
    contributions = np.zeros((rank, n, p))
    for r in range(rank):
        Ar = test_A * U_f[:, r].reshape(n, m)
        Br = test_B * V_f[:, r].reshape(m, p)
        Mr = Ar @ Br
        Wr = W_f[:, r].reshape(p, n).T
        contributions[r] = Mr * Wr
    
    # Try removing each component
    for r_remove in range(rank):
        if time.time() - start_time > time_budget:
            break
        
        # Check if the remaining components can still produce C_true
        # when we adjust their coefficients
        # Actually, for exact decomposition, we need the sum to equal C_true
        # If we remove component r_remove, we need:
        # sum_{s != r_remove} U[:,s] * V[:,s] * W[:,s] = C_true - contributions[r_remove]
        
        # This is a non-trivial linear algebra problem. Let's try a simpler approach:
        # Check if the removed component is "redundant" by seeing if its contribution
        # is in the span of the others (on the test point)
        
        C_without = np.zeros((n, p))
        for s in range(rank):
            if s != r_remove:
                C_without += contributions[s]
        
        residual = C_true - C_without
        if np.max(np.abs(residual)) < 1e-9:
            # Component r_remove is redundant!
            mask = np.ones(rank, dtype=bool)
            mask[r_remove] = False
            U_f = U_f[:, mask]
            V_f = V_f[:, mask]
            W_f = W_f[:, mask]
            rank -= 1
            best_rank = rank
            best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
            # Recompute contributions
            contributions = np.zeros((rank, n, p))
            for r in range(rank):
                Ar = test_A * U_f[:, r].reshape(n, m)
                Br = test_B * V_f[:, r].reshape(m, p)
                Mr = Ar @ Br
                Wr = W_f[:, r].reshape(p, n).T
                contributions[r] = Mr * Wr
    
    return best_U, best_V, best_W, best_rank

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    # Build the direct sum decomposition (rank 98)
    U, V, W = _build_4x4x8_direct()
    
    # Verify with multiple seeds
    ok, err = _verify_multi_seed(U, V, W)
    if not ok:
        # Fallback to schoolbook (rank 128)
        n, m, p = 4, 4, 8
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=np.int16)
        V_f = np.zeros((m * p, rank), dtype=np.int16)
        W_f = np.zeros((p * n, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Try greedy rank reduction
    time_remaining = _time_left()
    if time_remaining > 30:
        try:
            U_opt, V_opt, W_opt, new_rank = _try_rank_reduction_greedy(
                U, V, W, time_budget=min(80, time_remaining - 20)
            )
            if new_rank < U.shape[1]:
                ok2, err2 = _verify_multi_seed(U_opt.astype(np.int16), 
                                                V_opt.astype(np.int16), 
                                                W_opt.astype(np.int16))
                if ok2:
                    U, V, W = U_opt.astype(np.int16), V_opt.astype(np.int16), W_opt.astype(np.int16)
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
