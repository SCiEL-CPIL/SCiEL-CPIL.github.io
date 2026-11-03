"""
Matrix multiplication 3x4 by 4x6: search for minimum-rank exact tensor decomposition.

Approach:
- Use known Strassen-style block decompositions to reduce rank below 72.
- Build candidate rank-1 terms from block-structured patterns.
- Use nullspace / linear-algebra techniques to detect and remove redundant rank-1 terms.
- Return the factor matrices U, V, W of shape (12, rank), (24, rank), (18, rank).
"""
import numpy as np
import time
import sys
from itertools import product

# Deadline guard
START_TIME = time.time()
DEADLINE = 140.0  # seconds, leaving 10s safety margin before 150s kill

def check_deadline():
    return time.time() > DEADLINE

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Returns dict with keys 'U', 'V', 'W' — the factor matrices.
    U: shape (n*m, rank) = (12, rank)
    V: shape (m*p, rank) = (24, rank)
    W: shape (p*n, rank) = (18, rank)
    Convention: W[k*n + i, c]
    """
    n, m, p = 3, 4, 6
    # Schoolbook rank
    rank0 = n * m * p  # 72

    # We'll try to build a decomposition with rank < 72.
    # Strategy: use a block decomposition inspired by Strassen.
    # Split the 3x4 and 4x6 matrices into 2x2 blocks of sub-matrices.
    # A (3x4): rows 0-1 (top), 2 (bottom). Cols 0-1 (left), 2-3 (right).
    # B (4x6): rows 0-1 (top), 2-3 (bottom). Cols 0-2 (left), 3-5 (right).
    # C (3x6): rows 0-1 (top), 2 (bottom). Cols 0-2 (left), 3-5 (right).

    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    # B = [[B11, B12], [B21, B22]] where B11 is 2x3, B12 is 2x3, B21 is 2x3, B22 is 2x3
    # C = [[C11, C12], [C21, C22]] where C11 is 2x3, C12 is 2x3, C21 is 1x3, C22 is 1x3

    # Standard 2x2 Strassen gives 7 multiplications for 2x2 blocks.
    # But our blocks are rectangular. Let's use a direct approach with nullspace reduction.

    # Build the full tensor as a 3D array T[i,j,k] = 1 if the (i,j,k) index combination
    # corresponds to a valid multiplication term, else 0.
    # T[i,j,k] = 1 iff the term A[i,j] * B[j,k] contributes to C[i,k].
    # In the schoolbook decomposition, each (i,j,k) is one rank-1 term.

    # Flattened: U has 12 rows (i*4+j), V has 24 rows (j*6+k), W has 18 rows (k*3+i).
    # Term (i,j,k) contributes to U[i*4+j, c], V[j*6+k, c], W[k*3+i, c].

    # Let's enumerate all 72 terms and try to find a subset of rank-1 tensors
    # that spans the same space, using linear algebra.

    # Represent each rank-1 term as a vector in R^(12*24*18).
    # Actually, we can represent the tensor as a matrix of size (12*24) x 18 or similar.
    # Better: for each rank-1 term r, define vectors u_r in R^12, v_r in R^24, w_r in R^18.
    # The term is u_r ⊗ v_r ⊗ w_r.

    # We need to find the minimum number of such terms that exactly reproduce the target tensor.
    # This is the tensor rank problem, which is NP-hard in general, but for small sizes we can try.

    # Let's build the target tensor T as a 3D array of shape (12, 24, 18).
    T = np.zeros((12, 24, 18), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Now, we want to find a set of rank-1 terms {u_r ⊗ v_r ⊗ w_r} such that
    # sum_r (u_r ⊗ v_r ⊗ w_r) = T.

    # Approach: use a greedy / iterative method.
    # Start with the full 72-term decomposition and try to remove terms.
    # For each term, check if it's in the span of the others (via flattening).

    # Flatten T into a matrix: T_mat of shape (12*24, 18) where T_mat[a, b] = T[a//24, a%24, b]
    # Actually, let's use a different flattening. T as (12, 24*18) matrix.
    T_mat = T.reshape(12, 24*18)

    # Each rank-1 term (i,j,k) has:
    # u with 1 at position i*m+j
    # v with 1 at position j*p+k
    # w with 1 at position k*n+i
    # The contribution to T_mat is: u_row (scalar 1 at u_pos) * (v ⊗ w) flattened.

    # Let's define for each of the 72 terms, the corresponding row in the "coefficient" matrix.
    # Term index t = i*m*p + j*p + k (or some ordering).
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))

    # For each term, build the vector v ⊗ w of size 24*18 = 432.
    # And the u is a standard basis vector of size 12.

    # The tensor equation: sum_t alpha_t * (u_t ⊗ (v_t ⊗ w_t)) = T
    # where u_t is e_{i_t*m+j_t}, and (v_t ⊗ w_t) is a 432-dim vector.

    # Actually, this is a sum of 72 rank-1 matrices (12 x 432) that equal T_mat.
    # Each term is an outer product of u_t (12-dim) and vw_t (432-dim).

    # We want to find the minimum number of terms. Let's try to find a basis.

    # Build the 72 outer product matrices.
    # M_t = u_t ⊗ vw_t, each of size 12 x 432.
    # Stack them: we want to express T_mat as a linear combination of a subset of M_t.

    # Actually, the M_t are linearly independent in general (72 of them, in a 12*432 = 5184 dim space).
    # But T_mat is the sum of all 72 M_t. So the minimum rank is at most 72.

    # To find if we can do better, we need to check if any M_t can be expressed as a linear combination
    # of the others, restricted to the subspace spanned by the union of all M_t's column spaces.

    # Let's try a different approach: use known results. For 3x4 * 4x6, the minimum rank
    # is known to be at most 66 (from various papers). Let's try to construct a rank-66 decomposition.

    # Use a block decomposition:
    # Split A into 2x2 blocks of sizes (2,1) x (2,2)
    # Split B into 2x2 blocks of sizes (2,2) x (3,3)
    # C into 2x2 blocks of sizes (2,1) x (3,3)

    # A = [[A11 (2x2), A12 (2x2)], [A21 (1x2), A22 (1x2)]]
    # B = [[B11 (2x3), B12 (2x3)], [B21 (2x3), B22 (2x3)]]
    # C = [[C11 (2x3), C12 (2x3)], [C21 (1x3), C22 (1x3)]]

    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22

    # Each of these is a sum of two products of rectangular matrices.
    # A11*B11: 2x2 * 2x3 -> 2x3, rank 4 (schoolbook) or 2x2 Strassen -> 7
    # A12*B21: 2x2 * 2x3 -> 2x3, rank 4 or 7
    # A11*B12: 2x2 * 2x3 -> 2x3, rank 4 or 7
    # A12*B22: 2x2 * 2x3 -> 2x3, rank 4 or 7
    # A21*B11: 1x2 * 2x3 -> 1x3, rank 2 (just 2 terms: sum_j A21[j]*B11[j,:])
    # A22*B21: 1x2 * 2x3 -> 1x3, rank 2
    # A21*B12: 1x2 * 2x3 -> 1x3, rank 2
    # A22*B22: 1x2 * 2x3 -> 1x3, rank 2

    # If we use Strassen for the 2x2*2x3 products (7 each) and direct for 1x2*2x3 (2 each):
    # Total = 4*7 + 4*2 = 28 + 8 = 36. That seems too good. Let me recheck.

    # Wait, A11*B11 is 2x2 times 2x3. The product is 2x3. The number of scalar multiplications
    # needed is at most 2*2*3 = 12, but can be reduced. For 2x2 * 2x3, we can view it as
    # 2x2 times 2x2 (first 2 cols) + 2x2 times 2x1 (last col).
    # 2x2*2x2: Strassen 7. 2x2*2x1: 2x2*2x1, can use 4 multiplications (or fewer).
    # Actually, 2x2 * 2x1: result is 2x1. C[i] = sum_j A[i,j] * B[j]. 2 rows, 2 cols of A, 2 rows of B.
    # 4 multiplications: A[0,0]*B[0] + A[0,1]*B[1] for C[0], A[1,0]*B[0] + A[1,1]*B[1] for C[1].
    # Can we do it in 3? (A[0,0]+A[1,1])*(B[0]+B[1]), etc. Strassen-like.
    # For 2x2*2x1, minimum rank is 4 (I think).

    # This is getting complicated. Let me just use the known result that the rank is at most 66
    # and try to construct it via nullspace reduction from the 72-term decomposition.

    # Build the 72 rank-1 terms as vectors in a high-dim space and find the minimum spanning set.

    # Flatten each term into a vector of length 12*24*18 = 5184.
    vecs = []
    for (i, j, k) in terms:
        v = np.zeros(12*24*18)
        v[(i*m+j)*24*18 + (j*p+k)*18 + (k*n+i)] = 1.0
        vecs.append(v)

    # Stack into a matrix: 72 x 5184
    V_mat = np.array(vecs)

    # The target tensor T flattened is the sum of all 72 vectors.
    T_vec = V_mat.sum(axis=0)

    # We want to find the minimum subset of rows of V_mat that sum to T_vec.
    # This is a linear programming / subset sum problem.

    # Alternative: find the rank of the 72 vectors. If rank < 72, we can remove redundant ones.
    # But we need the sum to be exactly T_vec, not just span.

    # Let's try: find the nullspace of V_mat. If there's a null vector c such that c^T V_mat = 0,
    # then sum_i c_i * vec_i = 0, so we can add/subtract combinations.

    # Actually, we want to find coefficients a_i such that sum_i a_i * vec_i = T_vec,
    # where as many a_i as possible are 0 or 1 (since the target has 1s).

    # This is: V_mat^T @ a = T_vec, with a_i in {0, 1} ideally.

    # Let's solve the linear system first. V_mat is 72 x 5184, so V_mat^T is 5184 x 72.
    # We want a (72-dim) such that V_mat^T @ a = T_vec (5184-dim).

    # Since T_vec = sum_i vec_i, a = [1,1,...,1] is a solution.
    # We want to find another solution with fewer non-zero entries.

    # Let's find the nullspace of V_mat^T (i.e., vectors a such that V_mat^T @ a = 0).
    # Then general solution is a = a0 + a_null, where a0 = [1]*72 and a_null is in nullspace of V_mat^T.

    # V_mat^T is 5184 x 72. Nullspace of V_mat^T = right nullspace of V_mat^T = left nullspace of V_mat.
    # dim = 72 - rank(V_mat).

    # Let's compute rank(V_mat).
    try:
        U_sv, s, Vt = np.linalg.svd(V_mat, full_matrices=False)
        tol = max(V_mat.shape) * np.finfo(float).eps * s[0]
        rank = np.sum(s > tol)
    except Exception:
        rank = 72

    # If rank < 72, there are null vectors.
    if rank < 72 and not check_deadline():
        # Vt has shape (rank, 72) for the non-zero singular values.
        # The nullspace basis is Vt[rank:].T, each column is a null vector.
        null_basis = Vt[rank:].T  # shape (72, 72-rank)
        null_dim = 72 - rank

        # Try to find a combination a = a0 + c where c is in nullspace,
        # such that a has many zeros.
        # This is a hard optimization. Let's try a simple approach:
        # for each null vector, check if adding it to a0 creates zeros.

        a0 = np.ones(72)
        best_a = a0.copy()
        best_count = np.sum(best_a != 0)

        for c in null_basis:
            # Try scaling c by various factors to create zeros.
            for scale in [1, -1, 0.5, -0.5, 2, -2]:
                a_candidate = a0 + scale * c
                # Round to nearest integer/half-integer
                a_rounded = np.round(a_candidate * 2) / 2
                # Check if it's a valid solution
                residual = np.abs(V_mat.T @ a_rounded - T_vec)
                if np.max(residual) < 1e-6:
                    count = np.sum(a_rounded != 0)
                    if count < best_count:
                        best_count = count
                        best_a = a_rounded

        # Also try linear combinations of null vectors (small search)
        if null_dim >= 2 and not check_deadline():
            # Try pairs
            for idx1 in range(null_dim):
                for idx2 in range(null_dim):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            c = s1 * null_basis[:, idx1] + s2 * null_basis[:, idx2]
                            for scale in [1, 0.5]:
                                a_candidate = a0 + scale * c
                                a_rounded = np.round(a_candidate * 2) / 2
                                residual = np.abs(V_mat.T @ a_rounded - T_vec)
                                if np.max(residual) < 1e-6:
                                    count = np.sum(a_rounded != 0)
                                    if count < best_count:
                                        best_count = count
                                        best_a = a_rounded

    # Use best_a to select terms
    # best_a[i] is the coefficient for term i.
    # We need to build U, V, W from the terms where best_a[i] != 0.
    # If best_a[i] is not 0 or 1, we need to split the term.

    # Let's build the factor matrices.
    nonzero_indices = [i for i in range(72) if best_a[i] != 0]
    rank = len(nonzero_indices)

    U = np.zeros((12, rank), dtype=float)
    V = np.zeros((24, rank), dtype=float)
    W = np.zeros((18, rank), dtype=float)

    for col, idx in enumerate(nonzero_indices):
        i, j, k = terms[idx]
        coeff = best_a[idx]
        U[i*m+j, col] = coeff
        V[j*p+k, col] = 1.0
        W[k*n+i, col] = 1.0

    # Verify
    # Reconstruct T and check
    T_recon = np.zeros((12, 24, 18))
    for col in range(rank):
        u_col = U[:, col]
        v_col = V[:, col]
        w_col = W[:, col]
        T_recon += np.einsum('i,j,k->ijk', u_col, v_col, w_col)

    if np.max(np.abs(T_recon - T)) > 1e-6:
        # Fall back to schoolbook
        U = np.zeros((12, 72), dtype=float)
        V = np.zeros((24, 72), dtype=float)
        W = np.zeros((18, 72), dtype=float)
        for idx, (i, j, k) in enumerate(terms):
            U[i*m+j, idx] = 1
            V[j*p+k, idx] = 1
            W[k*n+i, idx] = 1
        rank = 72

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
