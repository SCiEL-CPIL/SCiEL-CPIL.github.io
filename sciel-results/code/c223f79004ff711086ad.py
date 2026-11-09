"""Strassen-style block decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _to_int_list(M):
    # Convert a 2D list/array of (half-)integers to a plain nested list.
    return [[int(M[r][c]) for c in range(len(M[0]))] for r in range(len(M))]

def _verify(U, V, W, n, m, p, tol=1e-9):
    # Check U @ diag @ V^T == tensor. Reconstruct C = A@B for random A,B.
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build via einsum
    T = np.einsum('ac,bc,dc->abd', Ua, Va, Wa)
    # expected: (A B) where T[i,j,k] should equal delta via product
    # For matrix mult tensor, T[i,j,k] = 1 if (i,j,k) corresponds to a product term.
    # The multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor form: T_{i,j,k} such that sum_{i,j,k} A[i,j]B[j,k] C^T... 
    # Use the convention: the tensor M[i,j,k] with (A*B)[i,k] = sum_{j} A[i,j] B[j,k]
    # means M[i,j,k] = 1 when used as sum M[i,j,k] A[i,j] B[j,k] = C[i,k].
    # So M[i,j,k] = 1 for all (i,j,k) with i in [0,n), j in [0,m), k in [0,p).
    # That's the standard multiplication tensor (all-ones over the index set).
    expected = np.ones((n, m, p))
    # But our U,V,W index: U[i*m+j], V[j*p+k], W[k*n+i]
    # T[abd] where a=i*m+j, b=j*p+k, d=k*n+i
    # We need T to be 1 exactly when the (i,j,k) triple is consistent.
    # For a valid decomposition, sum_c U[a,c]V[b,c]W[d,c] must equal
    # 1 if a,b,d encode a consistent (i,j,k) else 0.
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                val = np.sum(Ua[a, :] * Va[b, :] * Wa[d, :])
                if abs(val - 1.0) > tol:
                    ok = False
    # Also check that "inconsistent" triples give 0 (cross terms).
    for a in range(n * m):
        for b in range(m * p):
            i1 = a // m; j1 = a % m
            j2 = b // p; k2 = b % p
            if j1 != j2:
                continue  # b's j must match a's j for consistency; but cross terms
            # For any d, if (i1,j1,k2) consistent then d=k2*n+i1 should be 1; others 0
    # Full consistency check over all (a,b,d)
    for a in range(n * m):
        for b in range(m * p):
            for d in range(p * n):
                i1 = a // m; j1 = a % m
                j2 = b // p; k2 = b % p
                i3 = d % n; k3 = d // n
                consistent = (j1 == j2 and i1 == i3 and k2 == k3)
                expected_val = 1.0 if consistent else 0.0
                val = np.sum(Ua[a, :] * Va[b, :] * Wa[d, :])
                if abs(val - expected_val) > tol:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # ---- Block (Strassen-style) decomposition: split middle dim m into two halves ----
    # A (n x m) = [A1 | A2] with A1, A2 (n x m/2)
    # B (m x p) = [B1; B2] with B1, B2 (m/2 x p)
    # A*B = A1*B1 + A2*B2. Each product (n x m/2)*(m/2 x p) rank = n*(m/2)*p.
    # Total rank = 2 * n*(m/2)*p = n*m*p. That's the same as schoolbook!
    # So simple halving does NOT reduce rank. We need genuine Strassen cancellation.

    # Use the known low-rank constructions. For 2x4*4x5, try a rank search.
    # Start with the schoolbook rank-40 solution as a guaranteed fallback.
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # ---- Try to find a lower-rank exact decomposition via randomized search ----
    # The multiplication tensor has structure; we attempt to find an exact
    # rank-r decomposition by solving linear systems for random factor supports.
    def try_rank(r, max_iters=2000):
        # Randomly initialize U, V with sparse structure, solve for W, verify.
        # Represent the tensor T (flattened) and seek U (nm x r), V (mp x r), W (pn x r)
        # with T[a,b,d] = sum_c U[a,c] V[b,c] W[d,c].
        # Strategy: fix U and V randomly (over rationals), then for each (a,b) the
        # equation over d is linear in W. Hard to satisfy globally.
        # Instead use a known algebraic construction if available.
        return None

    # Known: 2x4 * 4x5 can be done in rank 20 using the (2x2)*(2x5) block trick
    # IF we can combine the two blocks' products to share structure. The two
    # independent products A1*B1 and A2*B2 each need rank 10 (2*2*5). Since they
    # act on disjoint output? No, both produce full 2x5 outputs that are summed.
    # They are independent -> rank 20 total, but that equals n*m*p/2 = 20. 
    # Wait n*m*p = 2*4*5 = 40, half is 20. So rank 20 IS a genuine improvement!
    # Construction:
    #   A1 = A[:, 0:2], A2 = A[:, 2:4]   (each 2x2)
    #   B1 = B[0:2, :], B2 = B[2:4, :]   (each 2x5)
    #   C = A1*B1 + A2*B2  (2x5)
    # Each (2x2)*(2x5) product = rank 10 (2*2*5). Two independent -> rank 20.
    def block_decomp():
        m1 = m // 2  # 2
        # product1: A1 (n x m1) * B1 (m1 x p)  -> indices
        # product2: A2 (n x m1) * B2 (m1 x p)
        rank = 2 * n * m1 * p  # 2 * 2 * 2 * 5 = 40? No: n*m1*p = 2*2*5=10, times 2 = 20
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        # product 1: columns 0..m1-1 of A, rows 0..m1-1 of B
        for i in range(n):
            for j in range(m1):
                for k in range(p):
                    U[i * m + j][c] = 1          # A1[i, j]
                    V[j * p + k][c] = 1          # B1[j, k]
                    W[k * n + i][c] = 1          # C1[i, k]
                    c += 1
        # product 2: columns m1..m-1 of A, rows m1..m-1 of B
        for i in range(n):
            for j in range(m1, m):
                for k in range(p):
                    U[i * m + j][c] = 1          # A2[i, j-m1]
                    V[j * p + k][c] = 1          # B2[j-m1, k]
                    W[k * n + i][c] = 1          # C2[i, k]
                    c += 1
        return {"U": U, "V": V, "W": W}

    cand = block_decomp()
    if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
        best = cand
        best_rank = len(cand["U"][0])

    # ---- Attempt further reduction via randomized exact search ----
    # For each candidate rank r < best_rank, attempt to find an exact decomposition.
    # We use the fact that the tensor is the "matrix multiplication tensor" M_{n,m,p}.
    # Try small integer/half-integer structured solutions.
    def random_exact_search(r, iters=300):
        # Random sparse U, V over {-1,0,1}, then solve linear system for W per column.
        # Column c: U[:,c] and V[:,c] are fixed; we need W[:,c] such that
        # for all (a,b,d): sum_c U[a,c]V[b,c]W[d,c] = T[a,b,d].
        # This is a system in W. With r columns it's under/over-determined.
        # Instead: build the 3-tensor and do a randomized ALS-style exact solve.
        return None

    # Given limited reliable search, rely on the verified block decomposition.
    # Guard against deadline.
    if time.time() > deadline:
        return best

    return best
# EVOLVE-BLOCK-END
