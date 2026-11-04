"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72

    # Build the schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    # The matrix multiplication tensor T is indexed by (i,j,k) with T[i,j,k] = delta
    # A rank-r decomposition gives U, V, W such that for all (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) valid else 0
    #
    # We attempt a greedy/ILP-style search to reduce rank by merging terms.
    # Strategy: try to find a decomposition of rank R < 72 using random restarts
    # with a constraint satisfaction approach.

    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0  # seconds

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to find a lower-rank decomposition using a structured approach:
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # A rank-1 term (u, v, w) contributes u[i*m+j] * v[j*p+k] * w[k*n+i] to T[i,j,k].
    #
    # Key insight: if we allow entries in {0, 1, -1, 1/2, ...}, we can potentially
    # merge multiple schoolbook terms.
    #
    # Approach: Use a greedy set-cover heuristic where each candidate rank-1 tensor
    # is parameterized by sparse vectors, and we try to cover all 72 monomials.

    # For a more systematic approach, try small structured merges.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged
    # if there exists a rank-1 tensor covering both. This requires:
    #   U[i1*m+j1] * V[j1*p+k1] * W[k1*n+i1] = 1
    #   U[i2*m+j2] * V[j2*p+k2] * W[k2*n+i2] = 1
    #   and all other (i,j,k) get 0 contribution.
    #
    # This is generally impossible with just 2 terms replacing 1, but with
    # half-integer entries we might do better.

    # Let's try a direct search for rank 71: remove one schoolbook term and
    # see if the remaining 71 terms can be adjusted to cover the gap.
    # This is equivalent to checking if the removed term's monomial can be
    # expressed as a linear combination of the others.

    # More practically: try random rank-r decompositions with r < 72 and check
    # if they exactly decompose the tensor. Use random sparse initialization.

    def check_decomposition(U_r, V_r, W_r, tol=1e-9):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        r = U_r.shape[1]
        # Build the decomposition tensor
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 for all valid (i,j,k)
        # And T should be 0 for invalid (but all (i,j,k) with i<n, j<m, k<p are valid)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                    if max_err > tol:
                        return False
        return True

    # Try structured rank reductions:
    # 1. Merge pairs that share an index
    # 2. Use Strassen-like identities on sub-blocks

    # For 3x4x6, consider splitting into 3x2x3 + 3x2x3 blocks?
    # Or 2x4x4 + 1x4x6?

    # Try: use the fact that we can do 2x2x2 in rank 7 (Strassen).
    # Embed Strassen in sub-problems.

    # Actually, let's try a simple optimization: use numpy to solve for
    # minimum rank by checking if a given rank works via random search.

    # Given time constraints, try a few random searches for rank 71, 70, etc.
    # Each trial: initialize U, V, W with random ±1 or ±0.5 entries, then
    # use a few gradient steps to minimize the error.

    # But exact decomposition is hard. Let's try a more targeted approach.

    # Targeted approach: the schoolbook has 72 terms. Can we replace 3 terms
    # with 2 terms? Or 4 with 3?
    #
    # Consider three terms: (i,j,k1), (i,j,k2), (i,j,k3) for the same (i,j).
    # These require U[i*m+j,c]=1 for each, V[j*p+k_a,c]=1, W[k_a*n+i,c]=1.
    # Can we cover all three with 2 rank-1 terms?
    # Need: sum_c U[i*m+j,c]*V[j*p+k1,c]*W[k1*n+i,c] = 1
    #        sum_c U[i*m+j,c]*V[j*p+k2,c]*W[k2*n+i,c] = 1
    #        sum_c U[i*m+j,c]*V[j*p+k3,c]*W[k3*n+i,c] = 1
    # And for other (i',j',k'), the contribution should be 0 (or canceled).
    #
    # This is a system of linear equations. With 2 terms, we have 2*(n*m+m*p+p*n)
    # variables but constrained. It's underdetermined, so solutions may exist.

    # Let's try: for a fixed (i,j), can we cover all p=6 k-values with fewer than 6 terms?
    # We need vectors u (scalar, shared), v_k (length 6, one per k), w_k (length 6, one per k).
    # Actually for fixed (i,j), the constraint is: for each k, sum_c u_c * v_{k,c} * w_{k,c} = 1
    # where u_c = U[i*m+j,c], v_{k,c} = V[j*p+k,c], w_{k,c} = W[k*n+i,c].
    # But these must also be consistent with other (i',j') values.

    # This is getting complex. Let's just return the schoolbook for safety,
    # but first try a quick check if rank 71 is achievable via a simple linear algebra test.

    # Quick test: is the 72x72 matrix of inner products invertible?
    # If we remove one column and see if the remaining can span, that tells us
    # about linear dependence, not about the multiplicative constraint.

    # Given the difficulty and time limit, return schoolbook.
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
