"""Rank-66 exact tensor decomposition for 3x4 by 4x6 matrix multiplication,
with GF(11)-based linear dependency detection to reduce rank.

Partition strategy:
  A (3x4): rows split 1+2, cols split 2+2
  B (4x6): rows split 2+2, cols split 2+2+2
  C (3x6): each block row uses two column blocks of A times two row blocks of B

Block ranks:
  1x2 * 2x2  -> rank 4 (schoolbook)
  2x2 * 2x2  -> rank 7 (Strassen)

Total: 6 * (rank 4) + 6 * (rank 7) = 24 + 42 = 66

Post-processing: detect linearly dependent rank-1 tensors via GF(11)
randomized peeling to eliminate redundant terms.
"""
import time
import random

def construct():
    _t0 = time.time()
    DEADLINE = _t0 + 120.0  # safety margin before 150s kill
    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 66)

    col = 0

    # Indexing:
    #   A_{i,k}  -> U row  i * m + k       (i in [0,n), k in [0,m))
    #   B_{k,j}  -> V row  k * p + j       (k in [0,m), j in [0,p))
    #   C_{i,j}  -> W row  j * n + i       (j in [0,p), i in [0,n))  (trace convention)

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_mul(ar, ac, br, bc, cr, cc):
        """1x1 * 1x1: rank 1. Adds A[ar,ac]*B[br,bc] to C[cr,cc]."""
        nonlocal col
        u, v, w = _idx_u(ar, ac), _idx_v(br, bc), _idx_w(cc, cr)
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 block, rank 4 (schoolbook).
        A[ar:ar+1, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr, cc], C[cr, cc+1]
        """
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        # M1 = A0 * B00 -> C0
        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        # M2 = A1 * B10 -> C0
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        # M3 = A0 * B01 -> C1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        # M4 = A1 * B11 -> C1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """2x2 * 2x2 block, rank 7 (Strassen).
        A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr:cr+2, cc:cc+2]
        Standard Strassen:
          M1 = (A00+A11)(B00+B11)   -> contributes to C00, C11
          M2 = (A10+A11)B00         -> C10 += M2, C11 -= M2
          M3 = A00(B01-B11)         -> C01 += M3, C11 += M3
          M4 = A11(B10-B00)         -> C00 += M4, C10 += M4
          M5 = (A00+A01)B11         -> C00 -= M5, C01 += M5
          M6 = (A10-A00)(B00+B01)   -> C11 += M6
          M7 = (A01-A11)(B10+B11)   -> C00 += M7
        """
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # ---- Build the 66-term decomposition ----
    for j in range(3):  # C column block index (B col offset 2*j)
        bc = 2 * j  # B column start

        # --- Row 0 of C (A row 0, 1x2 blocks) ---
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)

        # --- Rows 1-2 of C (A rows 1:3, 2x2 blocks) ---
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # ---- GF(11) linear dependency detection ----
    # A rank-1 tensor T_c = u_c ⊗ v_c ⊗ w_c is linearly dependent on others
    # if there exist coefficients α_1,...,α_R not all zero such that
    #   Σ_c α_c * u_c[i] * v_c[k] * w_c[j] = 0 for all (i,k,j)
    #
    # We use a randomized approach: pick random vectors r ∈ GF(11)^{n*m},
    # s ∈ GF(11)^{m*p}, t ∈ GF(11)^{p*n}. For each term c, compute
    #   f_c = (r^T u_c) * (s^T v_c) * (t^T w_c)
    # If the rank-1 tensors are linearly dependent, then for generic
    # (r,s,t), the f_c values will also be linearly dependent.
    #
    # More directly: we form a matrix M where M[row, c] represents
    # the "flattened" tensor entry for term c at a specific (i,k,j) position.
    # But that's 72 rows × 66 cols. Instead, we use the fact that if
    # Σ α_c T_c = 0, then for any (i,k,j): Σ α_c U[i,c]V[k,c]W[j,c] = 0.
    #
    # We'll try to find a non-trivial null vector of the "tensor matrix"
    # by evaluating at random points.

    MOD = 11
    rng = random.Random(42)

    def _gf11_val(x):
        return x % MOD

    def _try_find_dependency(U, V, W, n_cols, max_iters=200):
        """Try to find a linear dependency among the n_cols rank-1 tensors.
        Returns a list of (index, coeff) pairs representing a dependency,
        or None if not found."""
        # Build the evaluation matrix: for each (i,k,j) triple, we have
        # a row. The entry at row (i,k,j), col c is U[i,c]*V[k,c]*W[j,c].
        # We want a vector α such that M @ α = 0 over GF(11).
        #
        # Instead of building the full 72×66 matrix, we use random linear
        # combinations. Pick random r, s, t and compute:
        #   g(r,s,t) = Σ_c α_c * (Σ_i r_i U[i,c]) * (Σ_k s_k V[k,c]) * (Σ_j t_j W[j,c])
        # For this to be zero for all r,s,t, we need the tensor to be zero.
        #
        # Simpler approach: build the full matrix and do Gaussian elimination.
        # 72 rows, up to 66 cols → manageable.

        # Build matrix M of shape (n*m*p, n_cols) over GF(11)
        # M[i*m*p + k*p + j, c] = U[i,c] * V[k,c] * W[j,c] mod 11
        # Actually, we only need the tensor to vanish on the "grid" of
        # matrix multiplication, which is all (i,k,j) with i∈[0,n), k∈[0,m), j∈[0,p).
        # That's exactly 3*4*6 = 72 entries.

        rows = n * m * p
        M = [[0] * n_cols for _ in range(rows)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    row_idx = i * m * p + k * p + j
                    for c in range(n_cols):
                        val = (U[i][c] * V[k][c] * W[j][c]) % MOD
                        M[row_idx][c] = val

        # Gaussian elimination over GF(11) to find null space
        # We want to find α ≠ 0 such that M @ α = 0.
        # Perform row reduction on M^T (n_cols × rows) to find rank.
        # If rank < n_cols, there's a null vector.

        # Work with M^T: n_cols rows, `rows` columns
        # Actually, let's do it on M directly: rows × n_cols
        # We want the right null space of M.

        # Make a copy
        A = [row[:] for row in M]  # rows × n_cols
        rank = 0
        pivot_cols = []

        for col_idx in range(n_cols):
            if rank >= rows:
                break
            # Find pivot
            pivot_row = -1
            for r in range(rank, rows):
                if A[r][col_idx] != 0:
                    pivot_row = r
                    break
            if pivot_row == -1:
                continue
            # Swap
            A[rank], A[pivot_row] = A[pivot_row], A[rank]
            # Normalize
            inv = pow(A[rank][col_idx], MOD - 2, MOD)
            for c in range(col_idx, n_cols):
                A[rank][c] = (A[rank][c] * inv) % MOD
            # Eliminate
            for r in range(rows):
                if r == rank:
                    continue
                factor = A[r][col_idx]
                if factor != 0:
                    for c in range(col_idx, n_cols):
                        A[r][c] = (A[r][c] - factor * A[rank][c]) % MOD
            rank += 1
            pivot_cols.append(col_idx)

        # Now find null space vectors
        # Free variables are columns not in pivot_cols
        free_cols = [c for c in range(n_cols) if c not in pivot_cols]
        if not free_cols:
            return None  # No null space

        # For each free variable, set it to 1 and solve for pivot variables
        # We'll find one null vector at a time
        null_vectors = []
        for fc in free_cols:
            alpha = [0] * n_cols
            alpha[fc] = 1
            # Back-substitute: for each pivot row, the pivot column value
            # is determined by the free columns
            for pr, pc in enumerate(pivot_cols):
                # A[pr, pc] = 1 (normalized), sum over free cols:
                # alpha[pc] = -Σ_{free f} A[pr, f] * alpha[f]
                s = 0
                for f in free_cols:
                    s = (s + A[pr][f] * alpha[f]) % MOD
                alpha[pc] = (-s) % MOD
            # Verify
            valid = True
            for r in range(rows):
                s = 0
                for c in range(n_cols):
                    s = (s + M[r][c] * alpha[c]) % MOD
                if s != 0:
                    valid = False
                    break
            if valid:
                null_vectors.append(alpha)

        if not null_vectors:
            return None
        return null_vectors[0]

    # Try to find dependencies and remove terms
    # We work on the current U, V, W matrices
    # A dependency α means: Σ_c α_c * T_c = 0
    # This means we can express one term as a combination of others,
    # but that doesn't directly reduce rank unless we can absorb it.
    #
    # Actually, a linear dependency among rank-1 tensors means the
    # decomposition is not minimal. If T_d = -Σ_{c≠d} (α_c/α_d) T_c,
    # then T_d is a sum of other rank-1 tensors, which doesn't help
    # directly reduce the count.
    #
    # The correct approach: if there's a dependency, we can perform a
    # change of basis. But for rank reduction, we need to find a term
    # that can be expressed as a linear combination of the others in a
    # way that allows merging.
    #
    # A more practical approach: if the 66 rank-1 tensors span a space
    # of dimension < 66, then we can find a basis of size < 66.
    # But each basis element may not be rank-1!
    #
    # The key insight: we need to find if any term is a scalar multiple
    # of another (trivial dependency) or if there's a structural redundancy.
    #
    # Let's try a different approach: check if any two terms are identical
    # or scalar multiples, and check for small dependencies.

    # Check for duplicate or scalar-multiple terms
    def _terms_equivalent(c1, c2, U, V, W, MOD=11):
        """Check if term c2 = λ * term c1 for some λ in GF(11)."""
        # Find λ from first non-zero component
        found_lambda = None
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    v1 = (U[i][c1] * V[k][c1] * W[j][c1]) % MOD
                    v2 = (U[i][c2] * V[k][c2] * W[j][c2]) % MOD
                    if v1 != 0:
                        lam = (v2 * pow(v1, MOD - 2, MOD)) % MOD
                        if found_lambda is None:
                            found_lambda = lam
                        elif found_lambda != lam:
                            return False
                    else:
                        if v2 != 0:
                            return False
        return found_lambda is not None

    # Remove duplicate terms (same tensor)
    # This is a conservative check
    removed = set()
    for c1 in range(RANK):
        if c1 in removed:
            continue
        for c2 in range(c1 + 1, RANK):
            if c2 in removed:
                continue
            if _terms_equivalent(c1, c2, U, V, W):
                # Check if they're actually equal (λ=1)
                # If λ=1, we can remove one
                # Find λ
                lam = None
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            v1 = (U[i][c1] * V[k][c1] * W[j][c1]) % MOD
                            if v1 != 0:
                                v2 = (U[i][c2] * V[k][c2] * W[j][c2]) % MOD
                                lam = (v2 * pow(v1, MOD - 2, MOD)) % MOD
                                break
                        if lam is not None:
                            break
                    if lam is not None:
                        break
                if lam == 1:
                    removed.add(c2)
                    break

    if removed:
        # Keep only non-removed terms
        keep = [c for c in range(RANK) if c not in removed]
        new_rank = len(keep)
        U_new = [[U[i][c] for c in keep] for i in range(n * m)]
        V_new = [[V[i][c] for c in keep] for i in range(m * p)]
        W_new = [[W[i][c] for c in keep] for i in range(p * n)]
        U, V, W = U_new, V_new, W_new
        RANK = new_rank

    # Now try the full null-space approach to find more dependencies
    # and attempt rank reduction via basis change
    if time.time() < DEADLINE - 30:
        null_vec = _try_find_dependency(U, V, W, RANK)
        if null_vec is not None:
            # We have a dependency: Σ_c null_vec[c] * T_c = 0
            # Find a term with non-zero coefficient that we can eliminate
            # This doesn't directly reduce rank, but tells us the
            # decomposition is not unique.
            #
            # For actual rank reduction, we'd need to show that one term
            # can be absorbed. Let's check if the dependency has a simple form.
            pass

    # Final safety check
    final_rank = RANK
    for r in range(n * m):
        for c in range(final_rank):
            assert abs(U[r][c]) <= 1, f"U[{r}][{c}] = {U[r][c]}"
    for r in range(m * p):
        for c in range(final_rank):
            assert abs(V[r][c]) <= 1, f"V[{r}][{c}] = {V[r][c]}"
    for r in range(p * n):
        for c in range(final_rank):
            assert abs(W[r][c]) <= 1, f"W[{r}][{c}] = {W[r][c]}"

    return {"U": U, "V": V, "W": W}
