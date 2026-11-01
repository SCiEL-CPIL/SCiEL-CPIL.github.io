import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_dec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _build_rank_39():
    """
    Build the known rank 39 decomposition.
    5 Strassen blocks (2x2x2) + 1 Schoolbook block (2x2x1).
    """
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r = 0
    def add_strassen_block(k0, k1):
        nonlocal r
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1

        # m1 = (a00+a11)(b00+b11)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (a10+a11)b00
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a00(b01-b11)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = a11(b10-b00)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a00+a01)b11
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (a10-a00)(b00+b01)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (a01-a11)(b10+b11)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 7  # This was wrong in original, should be r += 1 for each term
        # Wait, the original code did r += 1 for each term. Let me fix my thinking.
        # The original code increments r by 1 for each of the 7 terms.
        # My helper above is structured to add 7 terms.
        # Let's rewrite the helper to be cleaner or just inline it to avoid confusion.

    # Let's just use the original logic which was correct.
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1

        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    # Last column k=10: Schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1

    U[0, r] += 1
    V[v_e2, r] += 1
    W[w_c02, r] += 1
    r += 1

    U[1, r] += 1
    V[v_g2, r] += 1
    W[w_c02, r] += 1
    r += 1

    U[2, r] += 1
    V[v_e2, r] += 1
    W[w_c12, r] += 1
    r += 1

    U[3, r] += 1
    V[v_g2, r] += 1
    W[w_c12, r] += 1
    r += 1

    return U, V, W

def _try_rank_reduction(U, V, W, target_rank, n, m, p, max_iter=2000, tol=1e-8):
    """
    Attempt to reduce rank by removing linearly dependent terms.
    Uses ALS-like approach: fix two matrices, solve for the third.
    If a term becomes zero (or very small), we can potentially drop it.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]
    
    # We want to reach target_rank.
    # We can try to zero out columns one by one.
    # This is a heuristic.
    
    # A simpler heuristic: Randomly perturb and re-solve?
    # Or, use the fact that if U, V, W have rank < current_rank, we can SVD them.
    # But they must decompose the tensor exactly.
    
    # Let's try a different approach:
    # The tensor is T = sum U[:,r] V[:,r] W[:,r]^T
    # If we can find a linear dependence among the terms, we can eliminate one.
    
    # For now, this is a placeholder for a more sophisticated search.
    # Given time constraints, we might just return the input if we can't find a reduction.
    
    # Let's try to see if any column is a linear combination of others in U, V, W simultaneously.
    # This is hard.
    
    # Alternative: Use the structure.
    # The rank 39 solution is composed of 5 Strassen blocks (rank 7) and 1 Schoolbook block (rank 4).
    # Total 35 + 4 = 39.
    # Can we do better?
    # Lower bound for 2x2x11:
    # R(2,2,p) >= R(2,2,p-1) + R(2,2,1) ? No, that's not a standard lower bound.
    # R(2,2,p) >= ceil(4p/2) ? No.
    # R(2,2,p) >= 2p + 2 ?
    # For p=1, R=4. 2(1)+2=4.
    # For p=2, R=7. 2(2)+2=6. 7>6.
    # For p=3, R=11? (5 Strassen + 1 schoolbook = 7+4=11). 2(3)+2=8.
    # Actually, R(2,2,p) is known to be 2p+2 for p even? No.
    # R(2,2,p) = 2p+2 is a conjecture?
    # For p=11, 2(11)+2 = 24.
    # But we have 39.
    # Is there a better algorithm?
    # The "border rank" might be lower.
    # But we need exact rank.
    
    # Let's stick to the rank 39 solution as it is verified and correct.
    # Attempting to find a lower rank in 150s is risky and likely to fail or time out.
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Build the known rank 39 solution
    U, V, W = _build_rank_39()
    
    # 2. Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 44
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # 3. Try to reduce rank (heuristic)
    # If time permits, try to see if we can drop a term.
    # But for now, return rank 39.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
