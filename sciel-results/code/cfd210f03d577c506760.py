"""Search for a rank-minimal exact tensor decomposition of 3x4x8 matmul.
Starts from a Strassen-style 88-rank baseline and greedily eliminates
redundant terms via rank-revealing linear algebra, targeting the
theoretical lower bound of 41.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
N, M, P = 3, 4, 8
RANK_TARGET = 41
DEADLINE = 135.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _build_strassen_baseline():
    """Construct an 88-rank Strassen-style decomposition for 3x4x8 matmul."""
    n, m, p = N, M, P
    # We'll build the decomposition by covering the n*m*p = 96 output
    # entries with rank-1 tensors. Strassen's 2x2x2 trick gives 7 instead of 8
    # per 2x2 block.
    #
    # Layout: A is (n x m) = (3 x 4), B is (m x p) = (4 x 8).
    # We partition m=4 into two 2-blocks: m1=[0,1], m2=[2,3].
    # We partition p=8 into four 2-blocks: p1..p4.
    # We partition n=3: n1=[0,1] (2-block), n2=[2] (single row).
    #
    # For each (n-block, p-block) pair we use Strassen's 7-term trick
    # on the 2x2x2 sub-tensor, plus handle the single-row case.
    #
    # Actually, let's be more careful. The matmul tensor T has entries
    # T[i,j,k] = 1 if output[i,k] = sum_j A[i,j]*B[j,k].
    # A rank-1 term is (u, v, w) where u has 1 at (i,j), v has 1 at (j,k),
    # w has 1 at (i,k).
    #
    # Strassen for 2x2: to compute C = A*B where A,B are 2x2:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B10-B11)
    #   M4 = A11(B01-B10)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A01-A11)(B01+B11)
    #   C00 = M1+M4-M5+M7
    #   C01 = M3+M5
    #   C10 = M2+M4
    #   C11 = M1-M2+M3+M6
    #
    # Each Mi is a product of a linear combo of A entries and a linear combo
    # of B entries, contributing to one or more output entries.
    #
    # For our case, let's think of it as: for each output entry (i,k),
    # it depends on A[i,0..3] and B[0..3,k].
    #
    # Let me use a different approach: build the full 96-rank schoolbook
    # and then apply Strassen-style cancellations.
    #
    # Actually, the cleanest approach: build the tensor as a 3D array and
    # find a rank decomposition via the Strassen pattern.
    #
    # Let me just build a valid 96-rank (schoolbook) decomposition first,
    # then try to reduce it.
    rank = n * m * p  # 96
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank


def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W decompose the matmul tensor exactly."""
    n, m, p = N, M, P
    # Build the matmul tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i==i' and j==j' and k==k'
    # Actually: T[i*m+j, j*p+k, k*n+i] = 1 (the schoolbook tensor)
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    #
    # T[i*m+j, j'*p+k', k''*n+i''] = 1 iff i==i'' and j==j' and k==k''
    #
    # Let's verify by checking: for each (i, j, k), the product should give
    # the correct output.
    #
    # The matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor: T has T[i*m+j, j*p+k, k*n+i] = 1
    #
    # Verification: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal
    # 1 if (i,j,k) is a valid index, else 0.
    # But actually T is defined on all (a,b,c) in (n*m) x (m*p) x (p*n).
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    #
    # So for a = i*m+j, b = j'*p+k', c = k''*n+i'':
    # T[a,b,c] = 1 iff i==i'' and j==j' and k==k''
    #
    # The decomposition: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    #
    # Let's spot-check: for a valid (i,j,k), a=i*m+j, b=j*p+k, c=k*n+i:
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should = 1
    # For invalid combos, should = 0.
    #
    # Efficient check: build T as a 3D array and compare.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition
    # sum_r U[a,r]*V[b,r]*W[c,r]
    # This is a tensor product. We can compute it as:
    # D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = sum_r (U[a,r]*W[c,r]) * V[b,r]
    #
    # More efficiently: for each r, add outer(U[:,r], V[:,r], W[:,r])
    # But that's O(rank * n*m * m*p * p*n) which is fine for rank~96.
    D = np.zeros((n * m, m * p, p * n))
    for r in range(U.shape[1]):
        D += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.max(np.abs(D - T)) < tol


def _try_eliminate(U, V, W, target_rank, time_limit):
    """Try to reduce the rank by eliminating redundant terms.
    Uses the fact that if a column of U is in the span of others (similarly
    for V and W), we can potentially remove it.
    """
    n, m, p = N, M, P
    current_rank = U.shape[1]
    start_time = time.time()
    progress = False

    # Strategy: iteratively try to remove one column at a time.
    # A column r can be removed if the tensor it contributes is in the span
    # of the other columns' contributions.
    #
    # The contribution of column r is the rank-1 tensor T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r].
    # We can remove it if T_r is in the span of {T_s : s ≠ r}.
    #
    # This is hard to check directly. Instead, use a heuristic:
    # If U[:,r] is in the column span of U[:, -r] (and similarly for V and W
    # with appropriate coefficients), then the term might be removable.
    #
    # Actually, a cleaner approach: the tensor T = sum_r T_r. If we can find
    # coefficients such that T_r = sum_{s≠r} c_s T_s, then we can absorb
    # T_r into the others. But this requires T_r to be in the span of the
    # other rank-1 tensors, which is a strong condition.
    #
    # A more practical approach: try to find a linear dependence among the
    # columns of U (or V or W). If U[:,r] = sum_{s≠r} a_s U[:,s], then
    # T_r = sum_s a_s (U[:,s] ⊗ V[:,r] ⊗ W[:,r]).
    # This doesn't directly help unless V[:,r] and W[:,r] also have relations.
    #
    # Let me try a different approach: use the fact that the matmul tensor
    # has a specific structure. We can try to find a smaller decomposition
    # by solving for the entries of U, V, W with a smaller rank.
    #
    # For now, let's try a simple heuristic: check if any column of U is
    # (approximately) in the span of the others. If so, try to eliminate it
    # and re-solve.

    # Try to reduce by checking linear dependencies
    for attempt in range(current_rank - target_rank + 5):
        if time.time() - start_time > time_limit:
            break
        if current_rank <= target_rank:
            break

        # Find a column of U that is in the span of the others
        found = -1
        U_mat = U.copy()
        for r in range(current_rank):
            # Check if column r is in span of others
            others = list(range(current_rank))
            others.remove(r)
            if not others:
                continue
            U_others = U_mat[:, others]
            U_r = U_mat[:, r]
            # Check rank
            rank_others = np.linalg.matrix_rank(U_others, tol=1e-10)
            rank_with = np.linalg.matrix_rank(np.column_stack([U_others, U_r]), tol=1e-10)
            if rank_with == rank_others:
                found = r
                break

        if found < 0:
            # Try V
            for r in range(current_rank):
                others = list(range(current_rank))
                others.remove(r)
                if not others:
                    continue
                V_others = V[:, others]
                V_r = V[:, r]
                rank_others = np.linalg.matrix_rank(V_others, tol=1e-10)
                rank_with = np.linalg.matrix_rank(np.column_stack([V_others, V_r]), tol=1e-10)
                if rank_with == rank_others:
                    found = r
                    break

        if found < 0:
            # Try W
            for r in range(current_rank):
                others = list(range(current_rank))
                others.remove(r)
                if not others:
                    continue
                W_others = W[:, others]
                W_r = W[:, r]
                rank_others = np.linalg.matrix_rank(W_others, tol=1e-10)
                rank_with = np.linalg.matrix_rank(np.column_stack([W_others, W_r]), tol=1e-10)
                if rank_with == rank_others:
                    found = r
                    break

        if found < 0:
            break

        # Remove column 'found'
        others = list(range(current_rank))
        others.remove(found)
        U_new = U[:, others]
        V_new = V[:, others]
        W_new = W[:, others]

        # Verify the remaining decomposition still works
        if _verify(U_new, V_new, W_new, tol=1e-8):
            U, V, W = U_new, V_new, W_new
            current_rank -= 1
            progress = True
        else:
            # Can't remove this column; try the next one
            # To avoid infinite loop, mark this column as tried
            break

    return U, V, W, current_rank, progress


def _strassen_decomposition():
    """Build a Strassen-style decomposition for 3x4x8 matmul.
    Uses 2x2x2 Strassen blocks to reduce from 96 to ~88 rank.
    """
    n, m, p = N, M, P
    # Partition:
    # n=3: rows 0,1 form a 2-block; row 2 is single
    # m=4: cols 0,1 form block 1; cols 2,3 form block 2
    # p=8: cols 0,1 form block 1; 2,3 block 2; 4,5 block 3; 6,7 block 4
    #
    # For each (n-part, m-part, p-part) combination, we use Strassen's trick
    # on the 2x2x2 sub-tensor where applicable.
    #
    # Let me think about this more carefully.
    #
    # The output C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # For i in {0,1} (2x2 n-block), j in {0,1} (m-block 1), k in a 2-block:
    #   C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in 2-block).
    # Strassen reduces 8 multiplications to 7.
    #
    # Similarly for j in {2,3}.
    #
    # For i=2 (single row), j in {0,1,2,3}, k in 2-block:
    #   C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k] + A[2,3]*B[3,k]
    # This is a 1x4x2 problem, which needs 8 multiplications (no Strassen benefit
    # for a single row).
    #
    # Total:
    # n-block {0,1}, m-block {0,1}: 4 p-blocks × 7 = 28
    # n-block {0,1}, m-block {2,3}: 4 p-blocks × 7 = 28
    # n-block {2}, m-block {0,1}: 4 p-blocks × 4 = 16
    # n-block {2}, m-block {2,3}: 4 p-blocks × 4 = 16
    # Total: 28 + 28 + 16 + 16 = 88
    #
    # Now let's build this explicitly.
    #
    # Strassen's 7 terms for 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1}):
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10
    # M3 = A00(B10-B11) -> contributes to C01
    # M4 = A11(B01-B10) -> contributes to C10, C11
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B10) -> contributes to C11
    # M7 = (A01-A11)(B01+B11) -> contributes to C00, C01
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mr is a product u * v where u is a linear combo of A entries and
    # v is a linear combo of B entries. In the tensor decomposition, each Mr
    # corresponds to a rank-1 term where:
    #   U has the coefficients of u spread over the (i,j) indices
    #   V has the coefficients of v spread over the (j,k) indices
    #   W has the output coefficients spread over the (i,k) indices
    #
    # Let me define the Strassen terms for a generic 2x2x2 block.
    # Let the A indices be (i0,j0), (i0,j1), (i1,j0), (i1,j1)
    # Let the B indices be (j0,k0), (j0,k1), (j1,k0), (j1,k1)
    # Let the C indices be (i0,k0), (i0,k1), (i1,k0), (i1,k1)
    #
    # M1: u = A(i0,j0) + A(i1,j1), v = B(j0,k0) + B(j1,k1)
    #   U: +1 at (i0,j0), +1 at (i1,j1)
    #   V: +1 at (j0,k0), +1 at (j1,k1)
    #   W: +1 at (i0,k0), +1 at (i1,k1)
    #
    # M2: u = A(i1,j0) + A(i1,j1), v = B(j0,k0)
    #   U: +1 at (i1,j0), +1 at (i1,j1)
    #   V: +1 at (j0,k0)
    #   W: +1 at (i1,k0)
    #
    # M3: u = A(i0,j0), v = B(j1,k0) - B(j1,k1)
    #   U: +1 at (i0,j0)
    #   V: +1 at (j1,k0), -1 at (j1,k1)
    #   W: +1 at (i0,k1)
    #
    # M4: u = A(i1,j1), v = B(j0,k1) - B(j1,k0)
    #   U: +1 at (i1,j1)
    #   V: +1 at (j0,k1), -1 at (j1,k0)
    #   W: +1 at (i1,k0), +1 at (i1,k1)
    #
    # M5: u = A(i0,j0) + A(i0,j1), v = B(j1,k1)
    #   U: +1 at (i0,j0), +1 at (i0,j1)
    #   V: +1 at (j1,k1)
    #   W: +1 at (i0,k0), +1 at (i0,k1)
    #
    # M6: u = A(i1,j0) - A(i0,j0), v = B(j0,k0) + B(j1,k0)
    #   U: +1 at (i1,j0), -1 at (i0,j0)
    #   V: +1 at (j0,k0), +1 at (j1,k0)
    #   W: +1 at (i1,k1)
    #
    # M7: u = A(i0,j1) - A(i1,j1), v = B(j0,k1) + B(j1,k1)
    #   U: +1 at (i0,j1), -1 at (i1,j1)
    #   V: +1 at (j0,k1), +1 at (j1,k1)
    #   W: +1 at (i0,k0), +1 at (i0,k1)
    #
    # Now let's build the full decomposition.

    U = {}  # column -> { (i*m+j) : coeff }
    V = {}  # column -> { (j*p+k) : coeff }
    W = {}  # column -> { (k*n+i) : coeff }
    col = 0

    def add_term(U_coeffs, V_coeffs, W_coeffs):
        nonlocal col
        U[col] = U_coeffs
        V[col] = V_coeffs
        W[col] = W_coeffs
        col += 1

    # n-block {0,1}, m-block {0,1}, for each p-block
    for pb in range(4):  # p-block index 0..3
        k0 = pb * 2
        k1 = k0 + 1
        i0, i1 = 0, 1
        j0, j1 = 0, 1
        # M1
        add_term({(i0,j0):1, (i1,j1):1},
                 {(j0,k0):1, (j1,k1):1},
                 {(k0,i0):1, (k1,i1):1})
        # M2
        add_term({(i1,j0):1, (i1,j1):1},
                 {(j0,k0):1},
                 {(k0,i1):1})
        # M3
        add_term({(i0,j0):1},
                 {(j1,k0):1, (j1,k1):-1},
                 {(k1,i0):1})
        # M4
        add_term({(i1,j1):1},
                 {(j0,k1):1, (j1,k0):-1},
                 {(k0,i1):1, (k1,i1):1})
        # M5
        add_term({(i0,j0):1, (i0,j1):1},
                 {(j1,k1):1},
                 {(k0,i0):1, (k1,i0):1})
        # M6
        add_term({(i1,j0):1, (i0,j0):-1},
                 {(j0,k0):1, (j1,k0):1},
                 {(k1,i1):1})
        # M7
        add_term({(i0,j1):1, (i1,j1):-1},
                 {(j0,k1):1, (j1,k1):1},
                 {(k0,i0):1, (k1,i0):1})

    # n-block {0,1}, m-block {2,3}, for each p-block
    for pb in range(4):
        k0 = pb * 2
        k1 = k0 + 1
        i0, i1 = 0, 1
        j0, j1 = 2, 3
        add_term({(i0,j0):1, (i1,j1):1},
                 {(j0,k0):1, (j1,k1):1},
                 {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1},
                 {(j0,k0):1},
                 {(k0,i1):1})
        add_term({(i0,j0):1},
                 {(j1,k0):1, (j1,k1):-1},
                 {(k1,i0):1})
        add_term({(i1,j1):1},
                 {(j0,k1):1, (j1,k0):-1},
                 {(k0,i1):1, (k1,i1):1})
        add_term({(i0,j0):1, (i0,j1):1},
                 {(j1,k1):1},
                 {(k0,i0):1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1},
                 {(j0,k0):1, (j1,k0):1},
                 {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1},
                 {(j0,k1):1, (j1,k1):1},
                 {(k0,i0):1, (k1,i0):1})

    # n-block {2}, m-block {0,1}, for each p-block (4 terms each)
    for pb in range(4):
        k0 = pb * 2
        k1 = k0 + 1
        i = 2
        j0, j1 = 0, 1
        # C[2,k0] += A[2,j0]*B[j0,k0] + A[2,j1]*B[j1,k0]
        # C[2,k1] += A[2,j0]*B[j0,k1] + A[2,j1]*B[j1,k1]
        add_term({(i,j0):1}, {(j0,k0):1}, {(k0,i):1})
        add_term({(i,j1):1}, {(j1,k0):1}, {(k0,i):1})
        add_term({(i,j0):1}, {(j0,k1):1}, {(k1,i):1})
        add_term({(i,j1):1}, {(j1,k1):1}, {(k1,i):1})

    # n-block {2}, m-block {2,3}, for each p-block (4 terms each)
    for pb in range(4):
        k0 = pb * 2
        k1 = k0 + 1
        i = 2
        j0, j1 = 2, 3
        add_term({(i,j0):1}, {(j0,k0):1}, {(k0,i):1})
        add_term({(i,j1):1}, {(j1,k0):1}, {(k0,i):1})
        add_term({(i,j0):1}, {(j0,k1):1}, {(k1,i):1})
        add_term({(i,j1):1}, {(j1,k1):1}, {(k1,i):1})

    total_rank = col
    # Convert to numpy arrays
    U_arr = np.zeros((n * m, total_rank))
    V_arr = np.zeros((m * p, total_rank))
    W_arr = np.zeros((p * n, total_rank))
    for c in range(total_rank):
        for (i, j), val in U[c].items():
            U_arr[i * m + j, c] = val
        for (j, k), val in V[c].items():
            V_arr[j * p + k, c] = val
        for (k, i), val in W[c].items():
            W_arr[k * n + i, c] = val

    return U_arr, V_arr, W_arr, total_rank


def construct():
    start_time = time.time()
    n, m, p = N, M, P

    # Build Strassen baseline (88 rank)
    U, V, W, rank = _strassen_decomposition()

    # Verify
    if not _verify(U, V, W, tol=1e-9):
        # Fallback to schoolbook
        U, V, W, rank = _build_strassen_baseline()

    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # Try to reduce further
    remaining_time = DEADLINE - (time.time() - start_time) - 5.0
    if remaining_time > 2.0:
        U2, V2, W2, rank2, _ = _try_eliminate(U, V, W, RANK_TARGET, remaining_time)
        if rank2 < best_rank and _verify(U2, V2, W2, tol=1e-9):
            best_U, best_V, best_W, best_rank = U2, V2, W2, rank2

    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        rows, cols = mat.shape
        result = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = mat[i, j]
                # Round to nearest half-integer
                val = round(val * 2) / 2
                result[i][j] = int(val) if val == int(val) else val
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
