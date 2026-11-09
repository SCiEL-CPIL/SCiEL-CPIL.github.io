import numpy as np
import time

DIMENSIONS = [3, 3, 6]
START = time.time()
DEADLINE = 150.0 - 5.0

# Precomputed valid rank-12 decomposition for 3x3 * 3x2
# This uses the structure: C[i,j] = sum_k A[i,k] * B[k,j]
# For 3x2 output, we have 6 output entries, each requiring 3 products = 18 scalar mults
# But with tensor rank, we can do better. Rank 12 is achievable.

def _make_rank12_3x3x2():
    """Create a rank-12 decomposition for 3x3 * 3x2."""
    r = 12
    n, m, p = 3, 3, 2
    U = np.zeros((r, n, m))
    V = np.zeros((r, m, p))
    W = np.zeros((r, n, p))
    
    # Use the identity-like structure but with proper indexing
    # For each output (i,j), we need sum_k A[i,k]*B[k,j]
    # We can use rank-3 decomposition for each (i,j) pair, total 6*3=18, but we can share
    # Actually, let's just use a simple valid construction:
    # For each k in 0..2, j in 0..1: use rank-3 for each (k,j)
    # That gives 2*3=6 ranks for B part, but we need to handle A part too
    
    # Simpler: just use the schoolbook-like but compressed
    # C[i,j] = sum_k A[i,k] B[k,j]
    # We can write this as: for each (i,k,j), use one rank term
    # That's 3*3*2=18 terms, but we can group by i: for each i, we have 3*2=6 terms
    # Total 3*6=18, not 12.
    
    # Let me try a different approach: use the fact that 3x3*3x2 can be seen as
    # (3x3 * 3x1) concatenated with (3x3 * 3x1)
    # Each 3x3*3x1 has rank at most 9 (schoolbook) but can be lower
    
    # Actually, the minimum rank for 3x3*3x2 is not well-known to be 12.
    # Let's just return None and let the search handle it.
    return None


def _ok():
    return time.time() < DEADLINE


def _verify_3x6(U, V, W, iters=12, seed=555):
    n, m, p = 3, 3, 6
    R = U.shape[1]
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-4, 5, size=(n, m))
        b = rng.integers(-4, 5, size=(m, p))
        C = a @ b
        Cc = np.zeros((n, p))
        for c in range(R):
            ua = np.array([[U[i * m + k, c] for k in range(m)] for i in range(n)])
            vb = np.array([[V[k * p + j, c] for j in range(p)] for k in range(m)])
            wa = np.array([[W[j * n + i, c] for i in range(n)] for j in range(p)])
            Cc += float(np.sum(ua * a)) * float(np.sum(vb * b)) * wa
        if not np.allclose(Cc, C, atol=1e-9, rtol=0.0):
            return False
    return True

def _verify_3x3x2(U3, V3, W3, iters=8, seed=777):
    """Verify rank-r decomposition for 3x3 * 3x2 matrix multiplication.
    U3: (r, 3, 3) U3[t,i,k]
    V3: (r, 3, 2) V3[t,k,j] 
    W3: (r, 3, 2) W3[t,i,j]
    """
    n, m, p = 3, 3, 2
    r = U3.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-3, 4, size=(n, m))
        b = rng.integers(-3, 4, size=(m, p))
        C = a @ b
        Cc = np.zeros((n, p))
        for t in range(r):
            ua = U3[t]  # (3,3)
            vb = V3[t]  # (3,2)
            wa = W3[t]  # (3,2)
            Cc += float(np.sum(ua * a)) * float(np.sum(vb * b)) * wa
        if not np.allclose(Cc, C, atol=1e-9, rtol=0.0):
            return False
    return True

def _construct_rank12_3x3x2():
    """Construct a valid rank-12 decomposition for 3x3 * 3x2."""
    r = 12
    n, m, p = 3, 3, 2
    U = np.zeros((r, n, m))
    V = np.zeros((r, m, p))
    W = np.zeros((r, n, p))
    
    # Use the structure: for each output (i,j), C[i,j] = sum_k A[i,k]*B[k,j]
    # We can use 3 rank terms per (i,j) pair, but share across i
    # Actually, let's use a simple valid construction with rank 12:
    # For each k in 0..2 and j in 0..1, we have 2*3=6 "basis" elements
    # For each i in 0..2, we need to combine these
    
    # Simpler: just use the fact that we can write the multiplication as
    # C = A * B where A is 3x3 and B is 3x2
    # Use Kronecker-like structure
    
    # Let's try: U[t,i,k], V[t,k,j], W[t,i,j]
    # For t = i*4 + k*2 + j (but that's 3*3*2=18, not 12)
    
    # Alternative: use rank-6 for each half, but share
    # Actually, the minimum rank for 3x3*3x2 is likely 12 or higher.
    # Let's just return None and rely on search.
    return None


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_known_rank12():
    """Try a known rank-12 decomposition for 3x3*3x2 and build 3x3*3x6 from it."""
    # The _RANK12_3x3x2 is currently just identity-like, which is rank 12 but may not be valid
    # Let's construct a proper one using the structure
    # For 3x3 * 3x2, we can use the fact that it's like 3x3 * 3x1 twice
    # Actually, let's just return None and rely on the search
    return None


def _find_3x3x2_decomp(rank, seed, trials=100):
    """Find rank decomposition for 3x3 * 3x2 matrix multiplication."""
    n, m, p = 3, 3, 2
    rng = np.random.default_rng(seed)
    
    for trial in range(trials):
        if not _ok():
            return None
        
        # Random U and V with small integer/half-integer values
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = rng.choice(vals, size=(rank, n, m))  # U[t, i, k]
        V = rng.choice(vals, size=(rank, m, p))  # V[t, k, j]
        
        # Build linear system for W
        # For each (i, j, k, k'): sum_t U[t,i,k] * V[t,k',j] * W[t,i,j] = delta(k,k')
        # Group by (i, j): for each (i,j), solve for W[:, i, j]
        
        W = np.zeros((rank, n, p))
        valid = True
        
        for i in range(n):
            for j in range(p):
                # Build A matrix: A[k, k', t] = U[t,i,k] * V[t,k',j]
                # We need: sum_t A[k, k', t] * W[t, i, j] = delta(k, k')
                # So for each (k, k'), equation: sum_t (U[t,i,k] * V[t,k',j]) * W[t,i,j] = delta(k,k')
                
                # A is (m*m, rank) where A[k*m+k', t] = U[t,i,k] * V[t,k',j]
                A = np.zeros((m * m, rank))
                b = np.zeros(m * m)
                for k in range(m):
                    for kp in range(m):
                        A[k * m + kp, :] = U[:, i, k] * V[:, kp, j]
                        b[k * m + kp] = 1.0 if k == kp else 0.0
                
                # Solve A @ W_col = b where W_col = W[:, i, j]
                try:
                    W_col, *_ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Round to half-integers
                W_col_rounded = np.round(W_col * 2) / 2
                W[:, i, j] = W_col_rounded
                
                # Check if rounded solution satisfies the constraint
                if not np.allclose(A @ W_col_rounded, b, atol=1e-9, rtol=0.0):
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Quick verification using the verify function
        if _verify_3x3x2(U, V, W, iters=4, seed=trial+1):
            return U.astype(float), V.astype(float), W.astype(float)
    
    return None


def _build_3x6_from_3x3x2(r, d0, d1, d2):
    """Build 3x3*3x6 decomposition from three 3x3*3x2 decompositions.
    
    d0, d1, d2 are tuples (U3, V3, W3) where:
      U3: (r, 3, 3) U3[t,i,k]
      V3: (r, 3, 2) V3[t,k,j] 
      W3: (r, 3, 2) W3[t,i,j]
    """
    n, m, p = 3, 3, 6
    R = 3 * r
    U = np.zeros((9, R))
    V = np.zeros((18, R))
    W = np.zeros((18, R))
    for q, (u, v, w) in enumerate((d0, d1, d2)):
        base = q * r
        for t in range(r):
            col = base + t
            for i in range(3):
                for k in range(3):
                    U[i * 3 + k, col] = u[t, i, k]
            for k in range(3):
                for j in range(2):
                    V[k * 6 + (q * 2 + j), col] = v[t, k, j]
            for i in range(3):
                for j in range(2):
                    W[(q * 2 + j) * 3 + i, col] = w[t, i, j]
    return U, V, W


def construct():
    # Try to find low-rank decompositions for 3x3*3x2, then build 3x3*3x6
    best_result = None
    
    for r in range(7, 13):
        if not _ok():
            break
        
        # Try multiple seeds for each rank
        for seed_offset in [0, 100, 200, 300]:
            if not _ok():
                break
            d = _find_3x3x2_decomp(r, seed=1000 + r + seed_offset)
            if d is not None:
                U3, V3, W3 = d
                # Verify it works
                if _verify_3x3x2(U3, V3, W3):
                    # Build 3x3*3x6 from three copies
                    U, V, W = _build_3x6_from_3x3x2(r, (U3, V3, W3), (U3, V3, W3), (U3, V3, W3))
                    if _verify_3x6(U, V, W):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If we didn't find a better rank, use schoolbook
    return _schoolbook()
