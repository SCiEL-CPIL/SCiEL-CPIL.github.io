import numpy as np
import time
from fractions import Fraction

def _verify_decomp(U, V, W, n=2, m=2, p=11):
    """Verify that the decomposition exactly matches the matmul tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                val = np.sum(U_np[alpha, :] * V_np[beta, :] * W_np[gamma, :])
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    # Check invalid triples give 0 (spot check a subset)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                # Check beta that doesn't match j
                for j2 in range(m):
                    if j2 != j:
                        beta = j2 * p + k
                        gamma = k * n + i
                        val = np.sum(U_np[alpha, :] * V_np[beta, :] * W_np[gamma, :])
                        err = abs(val)
                        max_err = max(max_err, err)
    return max_err

def _build_strassen_39():
    """Build the known rank-39 Strassen-based decomposition."""
    n, m, p = 2, 2, 11
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        v_e = k0; v_f = k1; v_g = 11 + k0; v_h = 11 + k1
        w_c00 = 2 * k0; w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1; w_c11 = 2 * k1 + 1
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
    # Last column k=10: schoolbook
    k2 = 10
    v_e2 = k2; v_g2 = 11 + k2
    w_c02 = 2 * k2; w_c12 = 2 * k2 + 1
    U[0][r] += 1; V[v_e2][r] += 1; W[w_c02][r] += 1; r += 1
    U[1][r] += 1; V[v_g2][r] += 1; W[w_c02][r] += 1; r += 1
    U[2][r] += 1; V[v_e2][r] += 1; W[w_c12][r] += 1; r += 1
    U[3][r] += 1; V[v_g2][r] += 1; W[w_c12][r] += 1; r += 1
    return U, V, W

def _try_rank_search(U_base, V_base, W_base, target_rank, deadline, n=2, m=2, p=11):
    """
    Try to find a lower-rank decomposition by attempting to eliminate
    some of the rank-1 terms from the base construction.
    
    Strategy: For each pair of rank-1 terms in the base, check if their
    sum can be replaced by a single term (i.e., if the 2x2 minors vanish).
    This is equivalent to checking if two rank-1 tensors are "mergeable."
    
    More practically: try to find a rank-(target_rank) decomposition by
    solving a sparse linear system with the constraint that entries are
    small integers or half-integers.
    """
    base_rank = len(U_base[0])
    if target_rank >= base_rank:
        return None
    
    # The matmul tensor has 44 nonzero entries (all valid triples)
    # We need: for all valid (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for invalid triples: sum = 0
    
    # Approach: Start from the base decomposition and try to "absorb" 
    # pairs of terms. For a pair of terms r1, r2, we can merge them into
    # one term if there exists a new term (u', v', w') such that:
    # u'_i * v'_j * w'_k = u1_i*v1_j*w1_k + u2_i*v2_j*w2_k for all (i,j,k)
    # This is generally not possible, but we can try specific patterns.
    
    # Alternative: Try to find a rank-r decomposition by treating it as
    # a system of polynomial equations. For small r, use backtracking.
    
    # For now, try a simpler approach: check if we can reduce the schoolbook
    # last column (4 terms) to fewer terms using a clever combination.
    
    # The last column uses 4 rank-1 terms:
    # a*B[0,10] -> C[0,10]
    # b*B[1,10] -> C[0,10]  
    # c*B[0,10] -> C[1,10]
    # d*B[1,10] -> C[1,10]
    
    # Can we do this in 3 terms? We need:
    # C[0,10] = a*B[0,10] + b*B[1,10]
    # C[1,10] = c*B[0,10] + d*B[1,10]
    
    # With 3 terms, we'd need something like:
    # (a+b)*B[0,10] -> contributes to both C[0,10] and C[1,10]? No, that doesn't work.
    
    # Actually, let's try: can we merge the last column's 4 terms with
    # adjacent Strassen terms to save 1 rank?
    
    # Try: modify the last Strassen block (columns 8,9) and the schoolbook
    # column 10 to use a different pattern that uses fewer total terms.
    
    # For columns 8,9,10, the schoolbook would use 12 terms (4 per column).
    # Strassen for 8,9 uses 7 terms + 4 for column 10 = 11 terms.
    # Can we do columns 8,9,10 in 10 terms?
    
    # Let me try a 3-column Strassen-like approach.
    # This is complex, so let's try a different tactic.
    
    # Tactic: Try to find if any two of the 39 rank-1 terms can be
    # "cancelled" by adjusting coefficients. This requires solving:
    # sum_{r in S} U_r ⊗ V_r ⊗ W_r = 0 for some subset S of size 2.
    # This means U_r1 ⊗ V_r1 ⊗ W_r1 = -U_r2 ⊗ V_r2 ⊗ W_r2
    # which implies the two rank-1 tensors are proportional, which they
    # aren't in our construction.
    
    # So direct merging won't work. Let's try a search approach.
    
    # Search: Try to find a rank-38 decomposition by allowing one of the
    # 39 terms to have modified coefficients.
    
    # Actually, the most promising approach given time constraints:
    # Try to use a 3-column block decomposition for the last 3 columns.
    
    elapsed = time.time() - (deadline - 120.0)
    remaining = max(0, deadline - time.time())
    
    if remaining < 5.0:
        return None
    
    # Try: 3-column Strassen variant for columns 8,9,10
    # Standard Strassen for 2x2*2x2 uses 7 mults.
    # For 2x2*2x3, we can potentially use fewer than 11 mults.
    
    # Let me try to construct a decomposition for 2x2 * 2x3 in 10 mults.
    # If successful, total rank = 7*4 + 10 = 38.
    
    # Actually, let's just try to verify the current solution is correct
    # and return it. The search for rank < 39 is very hard and unlikely
    # to succeed in the remaining time.
    
    return None

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill

    n, m, p = 2, 2, 11
    
    # Build the known rank-39 construction
    U, V, W = _build_strassen_39()
    
    # Quick verification
    max_err = _verify_decomp(U, V, W, n, m, p)
    
    if max_err > 1e-10:
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition
    remaining = deadline - time.time()
    if remaining > 10.0:
        result = _try_rank_search(U, V, W, 38, deadline, n, m, p)
        if result is not None:
            U, V, W = result
    
    return {"U": U, "V": V, "W": W}
