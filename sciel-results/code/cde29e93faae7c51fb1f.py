"""Matrix multiplication 2,3,9 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_rank1(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] equals the multiplication tensor."""
    rank = len(U[0]) if U and len(U[0]) > 0 else 0
    # Build the tensor efficiently
    err = 0.0
    for c in range(rank):
        uc = np.array(U[:, c] if isinstance(U, np.ndarray) else [U[i][c] for i in range(len(U))])
        vc = np.array(V[:, c] if isinstance(V, np.ndarray) else [V[i][c] for i in range(len(V))])
        wc = np.array(W[:, c] if isinstance(W, np.ndarray) else [W[i][c] for i in range(len(W))])
        # u outer v outer w: shape (n*m, m*p, p*n)
        # But we only need to check the (i*m+j, j*p+k, k*n+i) entries equal 1
        # and all other entries equal 0
        # Full tensor would be (n*m)*(m*p)*(p*n) = 54*27*18 = 26244 for 2,3,9
        # That's manageable
        t = np.einsum('i,j,k->ijk', uc, vc, wc)
        # Check against expected
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j, j * p + k, k * n + i)
                    if abs(t[idx] - 1.0) > tol:
                        return False
        # Check that non-target entries are ~0
        # We can do a quick check by summing absolute values of non-target entries
        # But that's expensive. Instead, let's just check a few random non-target entries
        # Actually for correctness we should check all. The tensor is 26244 entries, that's fine.
        mask = np.ones(t.shape, dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    mask[i * m + j, j * p + k, k * n + i] = False
        if np.max(np.abs(t[mask])) > tol:
            return False
    return True

def _als_solve(U, V, W, n, m, p, rank, max_iters=200, tol=1e-10, deadline=None):
    """
    Alternating Least Squares for tensor decomposition.
    We want: for all (i,j,k) in {0..n-1}x{0..m-1}x{0..p-1}:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    For all other (a,b,c) entries in the tensor: sum = 0.
    
    We only enforce the target equations (n*m*p of them) plus we need the
    non-target entries to be zero. But the non-target constraint is:
      For (a,b,c) not of the form (i*m+j, j*p+k, k*n+i):
      sum_c U[a,c]*V[b,c]*W[c_index,c] = 0
    
    This is complex. Instead, let's just enforce the target equations and
    rely on the structure. Actually, the full tensor constraint is needed.
    
    Let's use a different approach: we have n*m*p target equations. We'll solve
    for the minimum-norm solution that satisfies these. If the decomposition
    also satisfies the zero constraints, great. Otherwise we need to add them.
    
    For simplicity, let's just enforce the n*m*p target equations and verify
    at the end.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Convert to numpy for speed
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve for U: for each (i,j), we need
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all k
        # This is a system: for fixed (i,j), unknowns are U[i*m+j, 0..rank-1]
        # Equations: for each k in 0..p-1:
        #   sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
        # So A_U[i,j] is a p x rank matrix with A_U[i,j][k, c] = V[j*p+k, c] * W[k*n+i, c]
        # b_U[i,j] = ones(p)
        
        U_new = np.zeros((nm, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank), dtype=float)
                for k in range(p):
                    for c in range(rank):
                        A[k, c] = Va[j * p + k, c] * Wa[k * n + i, c]
                b = np.ones(p, dtype=float)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-12)
                    U_new[i * m + j, :] = sol
                except:
                    U_new[i * m + j, :] = Ua[i * m + j, :]
        
        # Solve for V: for each (j,k), unknowns are V[j*p+k, 0..rank-1]
        # Equations: for each i in 0..n-1:
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # A_V[j,k] is n x rank: A_V[j,k][i, c] = U[i*m+j, c] * W[k*n+i, c]
        
        V_new = np.zeros((mp, rank), dtype=float)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank), dtype=float)
                for i in range(n):
                    for c in range(rank):
                        A[i, c] = Ua[i * m + j, c] * Wa[k * n + i, c]
                b = np.ones(n, dtype=float)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-12)
                    V_new[j * p + k, :] = sol
                except:
                    V_new[j * p + k, :] = Va[j * p + k, :]
        
        # Solve for W: for each (k,i), unknowns are W[k*n+i, 0..rank-1]
        # Equations: for each j in 0..m-1:
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # A_W[k,i] is m x rank: A_W[k,i][j, c] = U[i*m+j, c] * V[j*p+k, c]
        
        W_new = np.zeros((pn, rank), dtype=float)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank), dtype=float)
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = Ua[i * m + j, c] * Va[j * p + k, c]
                b = np.ones(m, dtype=float)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-12)
                    W_new[k * n + i, :] = sol
                except:
                    W_new[k * n + i, :] = Wa[k * n + i, :]
        
        # Check convergence
        diff = np.max(np.abs(U_new - Ua)) + np.max(np.abs(V_new - Va)) + np.max(np.abs(W_new - Wa))
        Ua, Va, Wa = U_new, V_new, W_new
        if diff < tol:
            break
    
    # Convert back to list-of-lists
    U_out = [[Ua[i, c] for c in range(rank)] for i in range(nm)]
    V_out = [[Va[i, c] for c in range(rank)] for i in range(mp)]
    W_out = [[Wa[i, c] for c in range(rank)] for i in range(pn)]
    return U_out, V_out, W_out

def _random_init(n, m, p, rank, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    U_out = [[U[i, c] for c in range(rank)] for i in range(n * m)]
    V_out = [[V[i, c] for c in range(rank)] for i in range(m * p)]
    W_out = [[W[i, c] for c in range(rank)] for i in range(p * n)]
    return U_out, V_out, W_out

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0
    
    # Try to find a low-rank decomposition using ALS
    best = None
    for rank in [9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 27, 30, 36, 45, 54]:
        if time.time() > deadline - 5.0:
            break
        # Try multiple random inits
        found = False
        for seed in range(5):
            if time.time() > deadline - 3.0:
                break
            U0, V0, W0 = _random_init(n, m, p, rank, seed=seed)
            U, V, W = _als_solve(U0, V0, W0, n, m, p, rank, max_iters=150, tol=1e-10, deadline=deadline - 3.0)
            if _verify(U, V, W, n, m, p, tol=1e-7):
                best = (U, V, W)
                found = True
                break
        if found:
            break
    
    if best is not None:
        return {"U": best[0], "V": best[1], "W": best[2]}
    
    # Fallback to rank-1 (trivial)
    U, V, W = _build_rank1(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
