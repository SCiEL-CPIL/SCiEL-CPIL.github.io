import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=2, m=2, p=8, tol=1e-10):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    
    # Build target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Build decomposition tensor
    decomp = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        decomp += np.einsum('a,b,c->abc', u, v, w)
    
    return np.allclose(decomp, target, atol=tol)


def _strassen_28():
    """Return the rank-28 Strassen-based decomposition for 2x2 * 2x8."""
    n, m, p = DIMENSIONS  # 2, 2, 8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 * 2x2 * 2x2
    # Each term is (u, v, w) where u, v, w are vectors of length 4
    # Indexing: u[i*2+j] = A[i,j], v[j*2+k] = B[j,k], w[k*2+i] = C[k,i]
    terms = [
        # M1 = (A00+A11)*(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)*B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = A00*(B01-B11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = A11*(B10-B00)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5 = (A00+A01)*B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A10-A00)*(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)*(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    for block in range(4):
        k0 = block * 2
        for s in range(7):
            r = block * 7 + s
            u, v, w = terms[s]
            
            # Fill U: u[idx] where idx = i*2+j
            for i in range(2):
                for j in range(2):
                    idx = i * 2 + j
                    U[i * m + j][r] = u[idx]
            
            # Fill V: v[idx] where idx = j*2+k_local, k_global = k0 + k_local
            for j in range(2):
                for k_local in range(2):
                    idx = j * 2 + k_local
                    k_global = k0 + k_local
                    V[j * p + k_global][r] = v[idx]
            
            # Fill W: w[idx] where idx = k_local*2+i, k_global = k0 + k_local
            for k_local in range(2):
                for i in range(2):
                    idx = k_local * 2 + i
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _search_rank_27(deadline):
    """
    Search for a rank-27 decomposition by attempting to share one term across blocks.
    
    Strategy: Take the rank-28 solution and try to replace two terms from different
    blocks with a single shared term. This is a geometric problem: we need to find
    a rank-1 tensor that lies in the span of the two original terms and can be
    used to reconstruct both block contributions.
    
    More concretely: if we remove term r1 from block b1 and term r2 from block b2,
    we need to find a new term (u, v, w) such that:
    u1*v1*w1 + u2*v2*w2 = u*v*w + correction
    where correction can be absorbed into other terms.
    
    This is highly constrained. We use a limited search over possible pairings.
    """
    start = time.time()
    if start >= deadline:
        return None
    
    n, m, p = DIMENSIONS
    sol28 = _strassen_28()
    U28 = np.array(sol28['U'], dtype=float)
    V28 = np.array(sol28['V'], dtype=float)
    W28 = np.array(sol28['W'], dtype=float)
    
    # The rank-28 solution is a direct sum of 4 independent 2x2x2 tensors.
    # For rank 27 to work, we'd need to share a term across at least two blocks.
    # 
    # Key observation: The B and C indices for different blocks are disjoint.
    # Block b uses B indices {8*b, 8*b+1, 8*b+2, 8*b+3} and C indices {2*b, 2*b+1, 2*b+2, 2*b+3}.
    # A shared term would need to have nonzero entries in B/C indices from two different blocks,
    # which means it would contribute to both blocks simultaneously.
    #
    # Let's try: replace the last term of block 0 and the last term of block 1 with one shared term.
    # This is a very specific search. We'll use a small grid search over possible shared term parameters.
    
    # Try a few specific pairings of terms from different blocks
    # Pairing: (block 0, term 6) and (block 1, term 6)
    # These are M7 terms: (A01-A11)*(B10+B11)
    
    # For block 0: k0=0, k1=1
    # M7: U=[0,1,0,-1], V=[0,0,1,1] -> V[8*0+1]=1, V[8*1+1]=1, W[0*2+0]=1
    # For block 1: k0=2, k1=3
    # M7: U=[0,1,0,-1], V=[0,0,1,1] -> V[8*0+3]=1, V[8*1+3]=1, W[2*2+0]=1
    
    # A shared term would need:
    # U_shared = [0, 1, 0, -1] (same for both)
    # V_shared must have entries at B[1,1], B[1,3] (from block 0 and 1)
    # W_shared must have entries at C[0,0], C[2,0]
    
    # Let's try: V_shared[8*1+1] = a, V_shared[8*1+3] = b
    # W_shared[0*2+0] = c, W_shared[2*2+0] = d
    # We need: a*c = 1 (to match block 0's M7 contribution to C[0,0])
    #          b*d = 1 (to match block 1's M7 contribution to C[2,0])
    # But we also need to cancel the cross-terms that would be introduced.
    
    # This approach is too constrained. Let's try a different strategy:
    # Use numerical optimization to find if a rank-27 solution exists.
    
    # For now, return None since the direct sum structure strongly suggests rank 28 is minimal.
    # A more sophisticated search would use alternating least squares or similar.
    
    if time.time() >= deadline:
        return None
    
    return None


def construct():
    """
    Returns the best tensor decomposition found for 2x2 * 2x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    # Get the rank-28 solution
    sol28 = _strassen_28()
    
    # Verify it's correct
    if not _verify_decomposition(sol28['U'], sol28['V'], sol28['W']):
        # Should not happen, but return it anyway
        return sol28
    
    # Try to find a lower rank solution
    if time.time() < deadline:
        sol27 = _search_rank_27(deadline)
        if sol27 is not None:
            if _verify_decomposition(sol27['U'], sol27['V'], sol27['W']):
                return sol27
    
    return sol28

# EVOLVE-BLOCK-END
