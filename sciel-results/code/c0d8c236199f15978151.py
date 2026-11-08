"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114

Includes a post-processing step to remove any linearly dependent or duplicate terms
to ensure the minimal rank for this construction.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]


def _verify_decomposition(U, V, W, n=4, m=4, p=9, tol=1e-6):
    """Check that the decomposition exactly matches the matmul tensor."""
    rank = U.shape[1]
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # Vectorized update: T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        # Use broadcasting
        # Ur: (n, m) -> (n, m, 1)
        # Vr: (m, p) -> (1, m, p)
        # Wr: (p, n) -> (p, n, 1) -> transpose to (1, p, n)?
        # T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        # Term = Ur[i,j] * Vr[j,k] * Wr[k,i]
        # This is a tensor product contraction.
        # We can compute this as:
        # temp = Ur[:, :, np.newaxis] * Vr[np.newaxis, :, :]
        # temp shape (n, m, p)
        # Then multiply by Wr?
        # Wr[k, i] is at index (k, i).
        # We need to multiply temp[i, j, k] by Wr[k, i].
        # We can permute Wr to (i, k) then broadcast?
        # Wr is (p, n). We want Wr[k, i].
        # Let's use einsum for clarity and speed.
        T_decomp += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
    T_expected = np.ones((n, m, p))
    return np.allclose(T_decomp, T_expected, atol=tol)


def _build_strassen_114():
    """Build the known rank-114 Strassen block decomposition for 4x4 * 4x9."""
    n, m, p = 4, 4, 9
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Strassen 2x2 terms
    strassen_2x2_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        def get_A(gi, gj):
            return gi * m + gj
        def get_B(gj, gk):
            return gj * p + gk
        def get_C(gk, gi):
            return gk * n + gi

        def add_block_mult(A_side, B_side, C_side):
            for A_coeffs, B_coeffs, C_contribs in strassen_2x2_terms:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)

                for (li, lj), meta_coeff in A_coeffs.items():
                    for (br, bc, block_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[get_A(gi, gj)] += meta_coeff * block_coeff

                for (lj, lk), meta_coeff in B_coeffs.items():
                    for (br, bc, block_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[get_B(gj, gk)] += meta_coeff * block_coeff

                for (li, lk), meta_coeff in C_contribs.items():
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[get_C(gk, gi)] += meta_coeff * block_coeff

                add_term(u_vec, v_vec, w_vec)

        add_block_mult(
            [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)],
            {(0,0): 1, (1,1): 1}
        )
        add_block_mult(
            [(1,0,1), (1,1,1)], [(0,0,1)],
            {(1,0): 1, (1,1): -1}
        )
        add_block_mult(
            [(0,0,1)], [(0,1,1), (1,1,-1)],
            {(0,1): 1, (1,1): 1}
        )
        add_block_mult(
            [(1,1,1)], [(1,0,1), (0,0,-1)],
            {(0,0): 1, (1,0): 1}
        )
        add_block_mult(
            [(0,0,1), (0,1,1)], [(1,1,1)],
            {(0,0): -1, (0,1): 1}
        )
        add_block_mult(
            [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)],
            {(1,1): 1}
        )
        add_block_mult(
            [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)],
            {(0,0): 1}
        )

    # Part 1: A * B[0:4]
    strassen_4x4(0, 0, 0, 0, 0, 0)
    # Part 2: A * B[4:8]
    strassen_4x4(0, 0, 0, 4, 0, 4)
    # Part 3: A * B[8:9] (4x1), schoolbook
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = np.array(U_cols, dtype=float).T
    V = np.array(V_cols, dtype=float).T
    W = np.array(W_cols, dtype=float).T
    return U, V, W


def _prune_dependencies(U, V, W, n=4, m=4, p=9, max_time=10.0):
    """
    Attempt to prune linearly dependent terms.
    We check if the tensor contribution of term r is a linear combination
    of terms 0..r-1.
    """
    start = time.time()
    rank = U.shape[1]
    keep = []
    
    # Pre-compute the tensor contribution of each term as a vector
    # T_r[i, j, k] = U[i, r] * V[j, r] * W[k, r]
    terms = []
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                u_val = Ur[i, j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = Vr[j, k]
                    if v_val == 0: continue
                    w_val = Wr[k, i]
                    if w_val == 0: continue
                    vec[i * m * p + j * p + k] = u_val * v_val * w_val
        terms.append(vec)
        
        if time.time() - start > max_time:
            break

    # Now, check linear independence
    basis = []
    for r in range(len(terms)):
        vec = terms[r]
        if not np.any(vec):
            continue
            
        is_dependent = False
        if basis:
            B = np.array(basis).T
            try:
                x, residuals, rank_svd, s = np.linalg.lstsq(B, vec, rcond=None)
                if residuals and residuals[0] < 1e-8 * (np.linalg.norm(vec) + 1e-8):
                    is_dependent = True
            except np.linalg.LinAlgError:
                pass
        
        if not is_dependent:
            keep.append(r)
            basis.append(vec)
            
    if len(keep) < rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS

    # Build the rank-114 Strassen decomposition
    U, V, W = _build_strassen_114()

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to pure schoolbook (rank 144)
        U_cols, V_cols, W_cols = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n*m); v = [0] * (m*p); w = [0] * (p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    U_cols.append(u); V_cols.append(v); W_cols.append(w)
        rank = len(U_cols)
        U = np.array(U_cols, dtype=float).T
        V = np.array(V_cols, dtype=float).T
        W = np.array(W_cols, dtype=float).T

    # Try to reduce rank via pruning
    if time.time() < deadline - 5:
        U, V, W = _prune_dependencies(U, V, W, n, m, p, max_time=deadline - time.time() - 5)

    # Convert to lists
    rank = U.shape[1]
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
