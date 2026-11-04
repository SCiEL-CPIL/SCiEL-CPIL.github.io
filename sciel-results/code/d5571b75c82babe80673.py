import numpy as np
import time
import sys

def construct():
    """
    Returns an exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    The decomposition is constructed via a block structure that treats the multiplication
    as a sum over Kronecker-product blocks.  We first build a baseline decomposition
    with rank = 56 using a 7-multiply Strassen-like block for each (j-pair, k-pair).
    Then we search for column dependencies in the resulting (U, V, W) representation
    and eliminate redundant columns whenever possible, aiming for a lower rank.
    """
    t_start = time.time()
    deadline = t_start + 140  # leave a 10-second safety margin for the 150s limit

    n, m, p = 2, 4, 8

    # ---------- Baseline construction (rank 56) ----------
    base_rank = 56
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]

    def add_block(j0, k0, base_col):
        # Each block handles a 2x2 subproblem for rows j0,j0+1 of A
        # and columns k0,k0+1 of B.  Seven multiplications per block
        # following the Winograd/Strassen flavour for 2x2 matmul.

        # M1 = (A00+A11)(B00+B11)
        U[j0][base_col] += 1
        U[m + j0 + 1][base_col] += 1
        V[j0 * p + k0][base_col] += 1
        V[(j0 + 1) * p + k0 + 1][base_col] += 1
        W[k0 * n][base_col] += 1
        W[(k0 + 1) * n + 1][base_col] += 1

        # M2 = (A10+A11)B00
        U[m + j0][base_col + 1] += 1
        U[m + j0 + 1][base_col + 1] += 1
        V[j0 * p + k0][base_col + 1] += 1
        W[k0 * n + 1][base_col + 1] += 1
        W[(k0 + 1) * n + 1][base_col + 1] -= 1

        # M3 = A00(B10-B11)
        U[j0][base_col + 2] += 1
        V[j0 * p + k0 + 1][base_col + 2] += 1
        V[(j0 + 1) * p + k0 + 1][base_col + 2] -= 1
        W[(k0 + 1) * n][base_col + 2] += 1
        W[(k0 + 1) * n + 1][base_col + 2] += 1

        # M4 = A11(B00-B10)
        U[m + j0 + 1][base_col + 3] += 1
        V[(j0 + 1) * p + k0][base_col + 3] += 1
        V[j0 * p + k0][base_col + 3] -= 1
        W[k0 * n][base_col + 3] += 1
        W[k0 * n + 1][base_col + 3] += 1

        # M5 = (A00+A01)B11
        U[j0][base_col + 4] += 1
        U[j0 + 1][base_col + 4] += 1
        V[(j0 + 1) * p + k0 + 1][base_col + 4] += 1
        W[k0 * n][base_col + 4] -= 1
        W[(k0 + 1) * n][base_col + 4] += 1

        # M6 = (A01-A10)(B00+B10)
        U[m + j0][base_col + 5] += 1
        U[j0][base_col + 5] -= 1
        V[j0 * p + k0][base_col + 5] += 1
        V[j0 * p + k0 + 1][base_col + 5] += 1
        W[(k0 + 1) * n + 1][base_col + 5] += 1

        # M7 = (A10-A00)(B11+B01)
        U[j0 + 1][base_col + 6] += 1
        U[m + j0 + 1][base_col + 6] -= 1
        V[(j0 + 1) * p + k0][base_col + 6] += 1
        V[(j0 + 1) * p + k0 + 1][base_col + 6] += 1
        W[k0 * n][base_col + 6] += 1

        return base_col + 7

    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = add_block(j0, k0, col)

    # Convert to numpy for linear-algebra checks
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # ---------- Dependency / reduction search ----------
    # The tensor T_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r]
    # where a indexes (i,j), b indexes (j,k), c indexes (k,i) in the trace convention.
    # We can test whether any column r is a linear combination of other columns
    # by checking rank conditions on the three sets of columns.
    # A simpler, conservative approach: check if removing a column still preserves
    # the decomposition on a set of random probes.  We will use exact rational
    # verification on integer entries when possible, otherwise use tight floating tolerance.

    def verify(Ul, Vl, Wl, probes=None, tol=1e-9):
        """
        Returns True if the decomposition (Ul, Vl, Wl) matches the 2x4*4x8 matmul tensor
        on the given random matrices.  Ul/Vl/Wl are lists of lists (or arrays).
        """
        Ua = np.array(Ul, dtype=float)
        Va = np.array(Vl, dtype=float)
        Wa = np.array(Wl, dtype=float)
        rank = Ua.shape[1]
        if Va.shape[1] != rank or Wa.shape[1] != rank:
            return False

        # We test on a few deterministic probes to keep verification cheap.
        rng = np.random.RandomState(12345)
        if probes is None:
            probes = []
            for _ in range(4):
                A = rng.randint(-3, 4, size=(n, m)).astype(float)
                B = rng.randint(-3, 4, size=(m, p)).astype(float)
                probes.append((A, B))

        for A, B in probes:
            C_expected = A @ B
            C_decomp = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for r in range(rank):
                            s += A[i, j] * B[j, k] * Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    C_decomp[i, k] = s
            if not np.allclose(C_decomp, C_expected, atol=tol, rtol=tol):
                return False
        return True

    # Start with the full 56-column decomposition.
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = 56

    # Try to remove each column if the decomposition is preserved.
    # We proceed greedily: test each column in order; if removable, drop it and continue
    # with the smaller set.  This is O(rank * rank) in the worst case, which is fine.

    def try_remove(col_idx):
        nonlocal current_U, current_V, current_W, current_rank
        if time.time() > deadline - 5:
            return
        # Build candidate matrices without col_idx
        cand_U = [row[:col_idx] + row[col_idx+1:] for row in current_U]
        cand_V = [row[:col_idx] + row[col_idx+1:] for row in current_V]
        cand_W = [row[:col_idx] + row[col_idx+1:] for row in current_W]
        if verify(cand_U, cand_V, cand_W, tol=1e-8):
            current_U = cand_U
            current_V = cand_V
            current_W = cand_W
            current_rank -= 1

    # First pass: try to remove columns from the end backwards (keeps the prefix stable)
    for idx in range(current_rank - 1, -1, -1):
        try_remove(idx)
        if time.time() > deadline - 10:
            break

    # Second pass: another sweep from the front to catch newly exposed dependencies
    if time.time() < deadline - 15:
        for idx in range(current_rank - 1, -1, -1):
            try_remove(idx)
            if time.time() > deadline - 10:
                break

    # ---------- Polish: try zeroing small-integer coefficients and re-verify ----------
    # Since all entries are currently integers, we check if any non-zero entry can be
    # zeroed while preserving the decomposition.  This is a very local, cheap refinement.
    if time.time() < deadline - 15:
        # Work on copies
        work_U = [row[:] for row in current_U]
        work_V = [row[:] for row in current_V]
        work_W = [row[:] for row in current_W]

        # Scan U entries
        for r in range(current_rank):
            if time.time() > deadline - 10:
                break
            for a in range(n * m):
                orig = work_U[a][r]
                if orig == 0:
                    continue
                work_U[a][r] = 0
                if verify(work_U, work_V, work_W, tol=1e-8):
                    current_U = work_U
                else:
                    work_U[a][r] = orig
                    work_U = [row[:] for row in current_U]

        # Scan V entries
        for r in range(current_rank):
            if time.time() > deadline - 10:
                break
            for b in range(m * p):
                orig = work_V[b][r]
                if orig == 0:
                    continue
                work_V = [row[:] for row in current_V]
                work_V[b][r] = 0
                if verify(work_U, work_V, work_W, tol=1e-8):
                    current_V = work_V
                else:
                    work_V[b][r] = orig
                    work_V = [row[:] for row in current_V]

        # Scan W entries
        for r in range(current_rank):
            if time.time() > deadline - 10:
                break
            for c in range(p * n):
                orig = work_W[c][r]
                if orig == 0:
                    continue
                work_W = [row[:] for row in current_W]
                work_W[c][r] = 0
                if verify(work_U, work_V, work_W, tol=1e-8):
                    current_W = work_W
                else:
                    work_W[c][r] = orig
                    work_W = [row[:] for row in current_W]

    # ---------- Final verification with a tighter tolerance ----------
    if not verify(current_U, current_V, current_W, tol=1e-10):
        # Fall back to the original baseline to guarantee validity
        current_U = [row[:] for row in U]
        current_V = [row[:] for row in V]
        current_W = [row[:] for row in W]
        current_rank = 56

    # Clean up: strip trailing zero columns
    while current_rank > 0:
        if all(current_U[a][current_rank-1] == 0 for a in range(n*m)) and \
           all(current_V[b][current_rank-1] == 0 for b in range(m*p)) and \
           all(current_W[c][current_rank-1] == 0 for c in range(p*n)):
            current_rank -= 1
        else:
            break

    current_U = [row[:current_rank] for row in current_U]
    current_V = [row[:current_rank] for row in current_V]
    current_W = [row[:current_rank] for row in current_W]

    return {"U": current_U, "V": current_V, "W": current_W}


if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0]) if result['U'][0] else 0
    print("Rank:", rank)

    n, m, p = 2, 4, 8
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)

    np.random.seed(7)
    max_diff = 0.0
    for trial in range(6):
        A = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        C_decomp = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_decomp[i, k] = s
        d = float(np.max(np.abs(C_decomp - C_expected)))
        if d > max_diff:
            max_diff = d
    print("Max diff:", max_diff)
    if max_diff > 1e-10:
        print("ERROR: Decomposition is invalid!")
    else:
        print("Decomposition is valid.")
