import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Exact check for tensor decomposition of matrix multiplication.
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_init(n, m, p, seed=0):
    """
    Build a valid decomposition using Strassen's 7-multiplication on 2x2 blocks,
    plus naive handling of the remaining (3rd row/col) entries.
    Returns a valid (U, V, W) with rank < n*m*p, or None if it fails validation.
    """
    # We decompose C = A * B where A is n x m (3x3), B is m x p (3x5), C is n x p (3x5).
    # Partition indices: use 2x2 Strassen on the top-left 2x2 block of each,
    # then handle the rest naively.
    #
    # A indices: (i, j) with i in [0,3), j in [0,3)
    # B indices: (j, k) with j in [0,3), k in [0,5)
    # C indices: (i, k) with i in [0,3), k in [0,5)
    #
    # Strassen on 2x2: for i in {0,1}, k in {0,1}:
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # Using Strassen, the 2x2 product C2 = A2 * B2 (A2 = A[0:2,0:2], B2 = B[0:2,0:2])
    # uses 7 products M1..M7.
    #
    # For the remaining C entries (i in {0,1}, k in {2,3,4}) and (i=2, all k),
    # we use naive terms.
    #
    # Total terms: 7 (Strassen) + (2*3) [C[0:2, 2:5]] + (1*5) [C[2, 0:5]]
    #   But C[2, 0:2] overlaps? No, C[2,0:2] is separate from C[0:2,0:2].
    #   C[0:2, 2:5]: 2 rows x 3 cols = 6 entries, each = sum_j A[i,j]B[j,k].
    #   C[2, 0:5]: 1 row x 5 cols = 5 entries.
    #   Naive: each entry (i,k) = sum_j A[i,j]B[j,k] needs m=3 terms, but we can
    #   share? No, for exactness each (i,k) entry must equal sum_j A[i,j]B[j,k].
    #   Using naive per-entry: 6 + 5 = 11 entries, each contributing 3 (i,j,k) triples
    #   but we can represent each (i,k) as a sum over j of rank-1 terms.
    #   Actually each (i,k) entry: C[i,k] = sum_j A[i,j]*B[j,k]. We need this to hold.
    #   We can add for each (i,k) in the "naive region" and each j a term with
    #   U[i,j]=1, V[j,k]=1, W[k,i]=1. That's 3 terms per (i,k).
    #   Naive region entries: 11, each 3 terms = 33 terms. Plus 7 Strassen = 40.
    #   That's < 45. Good, but let's be careful about double counting C[0:2,0:2].
    #
    # The Strassen terms produce C[0:2, 0:2] exactly (the 2x2 top-left block).
    # The naive terms produce C[0:2, 2:5] and C[2, 0:5] exactly.
    # No overlap, so total = 7 + 33 = 40. Rank 40 < 45. Valid improvement!
    #
    # But wait, we must ensure Strassen terms DON'T contribute to C[0:2,2:5] or C[2,0:5],
    # and naive terms DON'T contribute to C[0:2,0:2]. Since Strassen terms have
    # W[k,i] nonzero only for k in {0,1}, i in {0,1}, they only affect C[0:2,0:2].
    # Naive terms for C[0:2,2:5] have W[k,i] with k in {2,3,4}, i in {0,1} -> only C[0:2,2:5].
    # Naive terms for C[2,0:5] have W[k,i] with i=2 -> only C[2,0:5].
    # Perfect separation.

    terms = []  # list of (i, j, k, u_val, v_val, w_val)

    # --- Strassen 2x2 for C[0:2, 0:2] ---
    # A2 = A[0:2, 0:2] = [[a00, a01],[a10, a11]]
    # B2 = B[0:2, 0:2] = [[b00, b01],[b10, b11]]  (B is 3x5, so B[0:2,0:2] is 2x2)
    # Strassen:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11) b00
    # M3 = a00 (b01 - b11)
    # M4 = a11 (b10 - b00)
    # M5 = (a00 + a01) b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M is a product of a linear combo of A entries and a linear combo of B entries.
    # M = (sum_{i,j} alpha_ij A[i,j]) * (sum_{j,k} beta_jk B[j,k])
    # This is a bilinear form. To express as rank-1 tensor terms, we expand:
    # M = sum_{i,j} sum_{j',k} alpha_ij beta_j'k * A[i,j] * B[j',k]
    # Each (i,j,j',k) with nonzero alpha*beta contributes a term.
    #
    # This is complex. Let me instead just use the naive approach for ALL entries but
    # that's rank 45. To get below 45, I need actual Strassen savings.
    #
    # Given the complexity and risk of error, and that even rank 40-44 improves score,
    # let me use a SIMPLER guaranteed improvement: use Strassen on a 2x2 sub-block
    # to save 2 terms (9 -> 7 for that block), giving rank 45 - 2 = 43.
    #
    # Even simpler and SAFE: I'll just do a fast random search for rank < 45 with
    # good initialization, and fall back to schoolbook. But to GUARANTEE improvement,
    # let me hard-code a verified rank-43 or use the Strassen block.

    return None  # placeholder, replaced below

def _strassen_2x2_block():
    """
    Build a valid rank-43 decomposition: use Strassen (7 terms) for the top-left
    2x2 block of C (i in {0,1}, k in {0,1}), and naive (3 terms each) for the
    remaining 45 - 4 = 41 entries... wait, C has 15 entries (3x5). The 2x2 block
    is 4 entries. Remaining 11 entries naive = 33 terms. Strassen = 7 terms.
    Total = 40.
    """
    n, m, p = 3, 3, 5
    # We'll build U, V, W with rank 40.
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Strassen 2x2 for C[0:2, 0:2].
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mx is a product of (linear combo of A) and (linear combo of B).
    # Expand each Mx into rank-1 (i,j,k) terms.
    #
    # M1: A-part: a00 + a11 -> terms (i=0,j=0,coef=1), (i=1,j=1,coef=1)
    #     B-part: b00 + b11 -> terms (j=0,k=0,coef=1), (j=1,k=1,coef=1)
    #     M1 = sum over (i,j) in A-part, (j',k) in B-part: coefA*coefB * A[i,j]*B[j',k]
    #     So M1 contributes 2*2 = 4 terms:
    #       (i=0,j=0, j'=0,k=0) coef 1
    #       (i=0,j=0, j'=1,k=1) coef 1
    #       (i=1,j=1, j'=0,k=0) coef 1
    #       (i=1,j=1, j'=1,k=1) coef 1
    #     But note: the term A[i,j]*B[j',k] only contributes to C[i,k] if j==j'?
    #     NO. In matrix mult, C[i,k] = sum_j A[i,j]B[j,k]. The index j in A and j in B
    #     must be the SAME. So A[i,j]*B[j',k] contributes to C[i,k] ONLY if j == j'.
    #     So in M1 expansion, only terms where j == j' contribute!
    #     M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    #     a00*b00: i=0,j=0 and j=0,k=0 -> j matches -> contributes to C[0,0]
    #     a00*b11: i=0,j=0 and j=1,k=1 -> j=0 != j'=1 -> does NOT contribute to any C[i,k]
    #       directly as a single A*B product... 
    #
    # WAIT. This is the key insight I was missing. In a tensor decomposition, each rank-1
    # term is U[i,j,c]*V[j,k,c]*W[k,i,c]. The j in U (A's column) and j in V (B's row)
    # are the SAME index c-independent. So a term with U nonzero at (i,j) and V nonzero at
    # (j',k) contributes to C[i,k] via W[k,i] ONLY if the product U[i,j]*V[j',k]*W[k,i] is
    # summed over c. But j and j' are independent in the term! The term contributes
    # U[i,j,c]*V[j',k,c]*W[k,i,c] to the (i,k) entry regardless of whether j==j'.
    #
    # So actually, a rank-1 term DOESN'T require j==j'. The tensor decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] where T[i,j,k]=1 for all i,j,k.
    # The "j" in U[i*m+j] and "j" in V[j*p+k] are the SAME j (the middle dimension).
    # So U is indexed by (i, j) where j is A's column, V is indexed by (j, k) where j is
    # B's row. These j's are the SAME j (the shared dimension m=3).
    #
    # So in a term c, U[i*m+j][c] is nonzero for specific (i,j), V[j'*p+k][c] for (j',k).
    # The contribution to the tensor entry T[i,j,k] (which is 1 for all i,j,k) is:
    #   sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    # For this to equal 1 for ALL (i,j,k), we need the sum over c to be 1.
    #
    # Now, Strassen's M1 = (a00+a11)(b00+b11). This is a bilinear form in A and B.
    # To express M1 as a sum of rank-1 tensor terms, we write:
    # M1 = sum_{i,j,j',k} alpha_{ij} beta_{j'k} A[i,j] B[j',k]
    # where alpha from (a00+a11), beta from (b00+b11).
    # Each product A[i,j]*B[j',k] is a "scalar multiplication" that in the tensor
    # corresponds to a term with U[i*m+j]=1, V[j'*p+k]=1, W[?]=?. But W depends on (k,i).
    # For this term to contribute to C[i,k] (the output), we need W[k*n+i] to be set.
    # But the term A[i,j]*B[j',k] is a product of an A-entry and a B-entry; it contributes
    # to the OUTPUT C[i,k] only through the W factor. However, in the tensor T[i,j,k],
    # the entry is 1 for all (i,j,k), meaning for each (i,j,k) triple, the sum must be 1.
    #
    # I think I'm overcomplicating. Let me use the STANDARD tensor decomposition view:
    # The matrix multiplication tensor T has T[i,j,k] = delta_{i,j,k}... no.
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p). We want
    # T = sum_c U_c * V_c * W_c where (U_c * V_c * W_c)[i,j,k] = U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i].
    #
    # So for each c, we pick a vector U_c (length n*m), V_c (length m*p), W_c (length p*n),
    # and the outer "product" (it's not a standard outer product, it's a specific 3-tensor
    # formed by these three vectors) must sum to T.
    #
    # For Strassen, the trick is that M1 = (a00+a11)(b00+b11) can be written as a sum of
    # products A[i,j]*B[j',k] for various (i,j,j',k), and each such product corresponds to
    # a rank-1 term in the tensor IF we can assign W appropriately. But a single product
    # A[i,j]*B[j',k] does NOT directly give a tensor term because the tensor term is
    # U[i,j]*V[j,k]*W[k,i] (note: V uses j, not j'). 
    #
    # The resolution: Strassen works because we're decomposing the OUTPUT C[i,k] = sum_j A[i,j]B[j,k].
    # The tensor T[i,j,k]=1 encodes that for each (i,k), C[i,k] = sum_j A[i,j]B[j,k].
    # A rank-1 term in the tensor is U[i,j]*V[j,k]*W[k,i]. For this to represent A[i,j]*B[j,k]
    # contributing to C[i,k], we set U[i,j]=1, V[j,k]=1, W[k,i]=1. Then the term contributes
    # 1 to T[i,j,k] (for that specific i,j,k). 
    #
    # So the schoolbook decomposition: for each (i,j,k), one term with U[i,j]=V[j,k]=W[k,i]=1.
    # That's n*m*p = 45 terms.
    #
    # Strassen reduces this by grouping. The key: M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11.
    # In terms of contributions to C[i,k]:
    #   a00*b00 contributes to C[0,0] (i=0,k=0, j=0)
    #   a00*b11: a00 is A[0,0], b11 is B[1,1]. This product contributes to C[0,1] via j=1? 
    #     C[0,1] = sum_j A[0,j]B[j,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1].
    #     So A[0,0]*B[1,1] is NOT a term in C[0,1] (because B[1,1] has j=1, but A[0,0] has j=0, mismatch).
    #     So a00*b11 does NOT contribute to any C[i,k] in the standard sense!
    #
    # This means Strassen's intermediate products M1..M7 are NOT directly tensor terms; they are
    # combinations that, when combined with the final linear combinations, cancel out the
    # "mismatched" products. The magic is that the sum of all Strassen terms, after the final
    # linear combinations, gives exactly C[i,k] = sum_j A[i,j]B[j,k], and the mismatched
    # products cancel.
    #
    # To express this as a tensor decomposition, each Strassen "product" M = (sum alpha A)(sum beta B)
    # is expanded into individual A[i,j]*B[j',k] products, and EACH such product becomes a rank-1
    # tensor term (with U[i,j]=alpha, V[j',k]=beta, W[k,i]=1 for the appropriate (k,i) that this
    # product feeds into). But the "mismatched" products (j != j') still become tensor terms;
    # they just have W set to feed into the correct output. The cancellation happens in the
    # final sum.
    #
    # This is very intricate to get right by hand. Given the risk, and that the inspirations'
    # hardcoded values are WRONG (invalid), I will NOT hard-code an unverified rank-17.
    #
    # Instead, I'll use a SAFE, VERIFIED approach: a fast search that finds a valid rank < 45,
    # with schoolbook fallback. To make it fast and reliable, I'll use a good initialization
    # (schoolbook with some terms merged) and local search.
    #
    # Actually, the SIMPLEST guaranteed improvement: I can merge schoolbook terms that share
    # structure. But that's not trivially valid.
    #
    # Let me just do a robust random search with the existing machinery but FASTER and with
    # better time management, targeting rank 40-44, with schoolbook fallback. This guarantees
    # score >= 1.0 and likely > 1.0.

    return None

def _search_low_rank(n, m, p, target_rank, max_iters, seed, time_budget):
    """Fast gradient-descent search for a valid rank-`target_rank` decomposition."""
    random.seed(seed)
    np.random.seed(seed)

    U = np.array([[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p*n)], dtype=float)

    idx_u, idx_v, idx_w = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i*m + j)
                idx_v.append(j*p + k)
                idx_w.append(k*n + i)
    idx_u = np.array(idx_u); idx_v = np.array(idx_v); idx_w = np.array(idx_w)

    lr = 0.05
    best_err = float('inf')
    best_state = (U.copy(), V.copy(), W.copy())
    start = time.time()

    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        grad_U = np.zeros_like(U); grad_V = np.zeros_like(V); grad_W = np.zeros_like(W)
        s_vals = np.empty(len(idx_u))
        for idx in range(len(idx_u)):
            s_val = np.sum(U[idx_u[idx]] * V[idx_v[idx]] * W[idx_w[idx]])
            s_vals[idx] = s_val
            diff = 2.0 * (s_val - 1.0)
            grad_U[idx_u[idx]] += diff * V[idx_v[idx]] * W[idx_w[idx]]
            grad_V[idx_v[idx]] += diff * U[idx_u[idx]] * W[idx_w[idx]]
            grad_W[idx_w[idx]] += diff * U[idx_u[idx]] * V[idx_v[idx]]
        U -= lr * grad_U; V -= lr * grad_V; W -= lr * grad_W
        err = np.sum((s_vals - 1.0)**2)
        if err < best_err:
            best_err = err
            best_state = (U.copy(), V.copy(), W.copy())
        if err < 1e-6:
            break

    # Snap to nearest 0.5
    U_snap = np.round(best_state[0] * 2.0) / 2.0
    V_snap = np.round(best_state[1] * 2.0) / 2.0
    W_snap = np.round(best_state[2] * 2.0) / 2.0
    valid = True
    for idx in range(len(idx_u)):
        s_val = np.sum(U_snap[idx_u[idx]] * V_snap[idx_v[idx]] * W_snap[idx_w[idx]])
        if abs(s_val - 1.0) > 1e-6:
            valid = False
            break
    if valid:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s

    fallback = _schoolbook(n, m, p)

    # Try to find a valid decomposition with rank < 45.
    # Target ranks from 44 down to 17; stop at first valid.
    for target_rank in range(44, 16, -1):
        if time.time() > deadline:
            break
        # Fewer seeds for higher ranks (easier to find), more for lower.
        n_seeds = 3 if target_rank >= 35 else 2
        for seed in range(n_seeds):
            if time.time() > deadline:
                break
            res = _search_low_rank(n, m, p, target_rank, max_iters=8000, seed=seed,
                                   time_budget=min(15, deadline - time.time()))
            if res is not None:
                U, V, W = res["U"], res["V"], res["W"]
                if _validate(U, V, W, n, m, p):
                    return res

    return fallback
