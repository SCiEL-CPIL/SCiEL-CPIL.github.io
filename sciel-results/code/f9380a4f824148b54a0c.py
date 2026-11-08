<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j2*p+k, r] * Wa[k*n+i, r]
                    expected = 1.0 if j == j2 else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def construct():
    start = time.time()
    deadline = start + 115  # safety margin

    n, m, p = DIMENSIONS

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def make_U(indices):
        u = [0.0] * (n * m)
        for idx, val in indices:
            u[idx] = val
        return u

    def make_V(indices):
        v = [0.0] * (m * p)
        for idx, val in indices:
            v[idx] = val
        return v

    def make_W(indices):
        w = [0.0] * (p * n)
        for idx, val in indices:
            w[idx] = val
        return w

    def strassen_2x2(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cr0, cr1, cc0, cc1):
        """
        Strassen for A[ar0:ar1+1, ac0:ac1+1] @ B[br0:br1+1, bc0:bc1+1]
        contributing to C[cr0:cr1+1, cc0:cc1+1]
        """
        ua11 = u_idx(ar0, ac0)
        ua12 = u_idx(ar0, ac1)
        ua21 = u_idx(ar1, ac0)
        ua22 = u_idx(ar1, ac1)

        vb11 = v_idx(br0, bc0)
        vb12 = v_idx(br0, bc1)
        vb21 = v_idx(br1, bc0)
        vb22 = v_idx(br1, bc1)

        wc11 = w_idx(cc0, cr0)
        wc12 = w_idx(cc1, cr0)
        wc21 = w_idx(cc0, cr1)
        wc22 = w_idx(cc1, cr1)

        # M1 = (a11+a22)(b11+b22) -> C11 += 1, C22 += 1
        U_list.append(make_U([(ua11, 1), (ua22, 1)]))
        V_list.append(make_V([(vb11, 1), (vb22, 1)]))
        W_list.append(make_W([(wc11, 1), (wc22, 1)]))

        # M2 = (a21+a22)b11 -> C21 += 1, C22 -= 1
        U_list.append(make_U([(ua21, 1), (ua22, 1)]))
        V_list.append(make_V([(vb11, 1)]))
        W_list.append(make_W([(wc21, 1), (wc22, -1)]))

        # M3 = a11(b21-b22) -> C12 += 1, C22 += 1
        U_list.append(make_U([(ua11, 1)]))
        V_list.append(make_V([(vb21, 1), (vb22, -1)]))
        W_list.append(make_W([(wc12, 1), (wc22, 1)]))

        # M4 = a22(b12-b11) -> C11 += 1, C21 += 1
        U_list.append(make_U([(ua22, 1)]))
        V_list.append(make_V([(vb12, 1), (vb11, -1)]))
        W_list.append(make_W([(wc11, 1), (wc21, 1)]))

        # M5 = (a11+a12)b22 -> C11 -= 1, C12 += 1
        U_list.append(make_U([(ua11, 1), (ua12, 1)]))
        V_list.append(make_V([(vb22, 1)]))
        W_list.append(make_W([(wc11, -1), (wc12, 1)]))

        # M6 = (a21-a11)(b11+b12) -> C22 += 1
        U_list.append(make_U([(ua21, 1), (ua11, -1)]))
        V_list.append(make_V([(vb11, 1), (vb12, 1)]))
        W_list.append(make_W([(wc22, 1)]))

        # M7 = (a12-a22)(b21+b22) -> C11 += 1, C22 += 1
        U_list.append(make_U([(ua12, 1), (ua22, -1)]))
        V_list.append(make_V([(vb21, 1), (vb22, 1)]))
        W_list.append(make_W([(wc11, 1), (wc22, 1)]))

    # Strategy: Build a decomposition that minimizes rank by exploiting the
    # structure of 3x3 @ 3x7.
    #
    # Key insight: For the "tail" columns of B (columns 3,4,5,6), we can use
    # a rank-9 decomposition for 3x3 @ 3x4 by sharing terms across columns.
    # For each (i,j) pair, one term handles ALL 4 tail columns simultaneously.
    #
    # For the first 3 columns, we use Strassen on the 2x2 block + schoolbook
    # for the remaining parts.
    #
    # Total rank: 7 (Strassen 2x2@2x2) + 4 (schoolbook 2x2@2x1) + 7 (A[0:2,2]*B[2,:])
    #            + 21 (A[2,:]*B[:,0:3]) + 9 (3x3@3x4 tail)
    #            = 48
    #
    # Wait, let me recount more carefully.
    #
    # Actually, let me use a cleaner decomposition:
    #
    # Part 1: A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2] : Strassen, rank 7
    # Part 2: A[0:2, 0:2] @ B[0:2, 2:3] -> C[0:2, 2:3] : schoolbook 2x2@2x1, rank 4
    # Part 3: A[0:2, 0:2] @ B[0:2, 3:7] -> C[0:2, 3:7] : 2x2@2x4
    #         For each (i,j) in 2x2, one term handles all 4 columns: rank 4
    # Part 4: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] : 2x1 @ 1x2, rank 4
    # Part 5: A[0:2, 2] * B[2, 2:3] -> C[0:2, 2:3] : 2x1 @ 1x1, rank 2
    # Part 6: A[0:2, 2] * B[2, 3:7] -> C[0:2, 3:7] : 2x1 @ 1x4, rank 2 (one per row)
    # Part 7: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] : 1x2 @ 2x2, rank 4
    # Part 8: A[2, 0:2] * B[0:2, 2:3] -> C[2, 2:3] : 1x2 @ 2x1, rank 2
    # Part 9: A[2, 0:2] * B[0:2, 3:7] -> C[2, 3:7] : 1x2 @ 2x4, rank 2 (one per col of A[2,:])
    # Part 10: A[2, 2] * B[2, 0:2] -> C[2, 0:2] : 1x1 @ 1x2, rank 2
    # Part 11: A[2, 2] * B[2, 2:3] -> C[2, 2:3] : 1x1 @ 1x1, rank 1
    # Part 12: A[2, 2] * B[2, 3:7] -> C[2, 3:7] : 1x1 @ 1x4, rank 1
    #
    # Total: 7+4+4+4+2+2+4+2+2+2+1+1 = 35
    #
    # Hmm, but I need to be more careful. Let me just build it and verify.

    U_list = []
    V_list = []
    W_list = []

    # Part 1: A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2] : Strassen, rank 7
    strassen_2x2(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)

    # Part 2: A[0:2, 0:2] @ B[0:2, 2:3] -> C[0:2, 2:3] : schoolbook 2x2@2x1, rank 4
    for i in [0, 1]:
        for j in [0, 1]:
            U_list.append(make_U([(u_idx(i, j), 1)]))
            V_list.append(make_V([(v_idx(j, 2), 1)]))
            W_list.append(make_W([(w_idx(2, i), 1)]))

    # Part 3: A[0:2, 0:2] @ B[0:2, 3:7] -> C[0:2, 3:7] : 2x2@2x4, rank 4
    for i in [0, 1]:
        for j in [0, 1]:
            v_indices = [(v_idx(j, k), 1) for k in range(3, 7)]
            w_indices = [(w_idx(k, i), 1) for k in range(3, 7)]
            U_list.append(make_U([(u_idx(i, j), 1)]))
            V_list.append(make_V(v_indices))
            W_list.append(make_W(w_indices))

    # Part 4: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] : 2x1 @ 1x2, rank 4
    for i in [0, 1]:
        for k in [0, 1]:
            U_list.append(make_U([(u_idx(i, 2), 1)]))
            V_list.append(make_V([(v_idx(2, k), 1)]))
            W_list.append(make_W([(w_idx(k, i), 1)]))

    # Part 5: A[0:2, 2] * B[2, 2:3] -> C[0:2, 2:3] : 2x1 @ 1x1, rank 2
    for i in [0, 1]:
        U_list.append(make_U([(u_idx(i, 2), 1)]))
        V_list.append(make_V([(v_idx(2, 2), 1)]))
        W_list.append(make_W([(w_idx(2, i), 1)]))

    # Part 6: A[0:2, 2] * B[2, 3:7] -> C[0:2, 3:7] : 2x1 @ 1x4, rank 2
    for i in [0, 1]:
        v_indices = [(v_idx(2, k), 1) for k in range(3, 7)]
        w_indices = [(w_idx(k, i), 1) for k in range(3, 7)]
        U_list.append(make_U([(u_idx(i, 2), 1)]))
        V_list.append(make_V(v_indices))
        W_list.append(make_W(w_indices))

    # Part 7: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] : 1x2 @ 2x2, rank 4
    for j in [0, 1]:
        for k in [0, 1]:
            U_list.append(make_U([(u_idx(2, j), 1)]))
            V_list.append(make_V([(v_idx(j, k), 1)]))
            W_list.append(make_W([(w_idx(k, 2), 1)]))

    # Part 8: A[2, 0:2] * B[0:2, 2:3] -> C[2, 2:3] : 1x2 @ 2x1, rank 2
    for j in [0, 1]:
        U_list.append(make_U([(u_idx(2, j), 1)]))
        V_list.append(make_V([(v_idx(j, 2), 1)]))
        W_list.append(make_W([(w_idx(2, 2), 1)]))

    # Part 9: A[2, 0:2] * B[0:2, 3:7] -> C[2, 3:7] : 1x2 @ 2x4, rank 2
    for j in [0, 1]:
        v_indices = [(v_idx(j, k), 1) for k in range(3, 7)]
        w_indices = [(w_idx(k, 2), 1) for k in range(3, 7)]
        U_list.append(make_U([(u_idx(2, j), 1)]))
        V_list.append(make_V(v_indices))
        W_list.append(make_W(w_indices))

    # Part 10: A[2, 2] * B[2, 0:2] -> C[2, 0:2] : 1x1 @ 1x2, rank 2
    for k in [0, 1]:
        U_list.append(make_U([(u_idx(2, 2), 1)]))
        V_list.append(make_V([(v_idx(2, k), 1)]))
        W_list.append(make_W([(w_idx(k, 2), 1)]))

    # Part 11: A[2, 2] * B[2, 2:3] -> C[2, 2:3] : 1x1 @ 1x1, rank 1
    U_list.append(make_U([(u_idx(2, 2), 1)]))
    V_list.append(make_V([(v_idx(2, 2), 1)]))
    W_list.append(make_W([(w_idx(2, 2), 1)]))

    # Part 12: A[2, 2] * B[2, 3:7] -> C[2, 3:7] : 1x1 @ 1x4, rank 1
    v_indices = [(v_idx(2, k), 1) for k in range(3, 7)]
    w_indices = [(w_idx(k, 2), 1) for k in range(3, 7)]
    U_list.append(make_U([(u_idx(2, 2), 1)]))
    V_list.append(make_V(v_indices))
    W_list.append(make_W(w_indices))

    rank = len(U_list)

    if _validate(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Tensor decomposition for 3x3 * 3x7 matrix multiplication using rank-reduction search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _validate_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized validation of tensor decomposition."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Build the multiplication tensor T[i*m+j, j2*p+k, k*n+i]
    # For each r, compute outer product and check
    for r in range(rank):
        u_r = Ua[:, r]  # shape (n*m,)
        v_r = Va[:, r]  # shape (m*p,)
        w_r = Wa[:, r]  # shape (p*n,)
        # Reshape to 2D
        u_mat = u_r.reshape(n, m)
        v_mat = v_r.reshape(m, p)
        w_mat = w_r.reshape(p, n)
        # The contribution to C[i,k] from this term is sum_j u[i,j]*v[j,k]*w[k,i]
        # Actually the term contributes u[i,j] * v[j,k] * w[k,i] to C[i,k]
        # We need: for all i,k: sum_j u[i,j]*v[j,k]*w[k,i] = delta_{j==j2} pattern
        # Let me think again...
        # T_{(i,j),(j',k),(k,i)} = 1 if j==j' else 0
        # So sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = 1 if j==j' else 0
        pass
    # Direct check
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[jp*p+k, r] * Wa[k*n+i, r]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _validate(U, V, W, n, m, p, tol=1e-9):
    return _validate_fast(U, V, W, n, m, p, tol)


def _build_schoolbook(n, m, p):
    """Build the trivial schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_strassen_based(n, m, p):
    """Build a Strassen-inspired decomposition for 3x3 * 3x7."""
    U_list = []
    V_list = []
    W_list = []

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def make_U(indices):
        u = [0.0] * (n * m)
        for idx, val in indices:
            u[idx] = val
        return u

    def make_V(indices):
        v = [0.0] * (m * p)
        for idx, val in indices:
            v[idx] = val
        return v

    def make_W(indices):
        w = [0.0] * (p * n)
        for idx, val in indices:
            w[idx] = val
        return w

    # Use Strassen for the 2x2x2 core block
    # A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    def strassen_2x2(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cr0, cr1, cc0, cc1):
        ua11 = u_idx(ar0, ac0)
        ua12 = u_idx(ar0, ac1)
        ua21 = u_idx(ar1, ac0)
        ua22 = u_idx(ar1, ac1)

        vb11 = v_idx(br0, bc0)
        vb12 = v_idx(br0, bc1)
        vb21 = v_idx(br1, bc0)
        vb22 = v_idx(br1, bc1)

        wc11 = w_idx(cc0, cr0)
        wc12 = w_idx(cc1, cr0)
        wc21 = w_idx(cc0, cr1)
        wc22 = w_idx(cc1, cr1)

        # M1 = (a11+a22)(b11+b22)
        U_list.append(make_U([(ua11, 1), (ua22, 1)]))
        V_list.append(make_V([(vb11, 1), (vb22, 1)]))
        W_list.append(make_W([(wc11, 1), (wc22, 1)]))

        # M2 = (a21+a22)b11
        U_list.append(make_U([(ua21, 1), (ua22, 1)]))
        V_list.append(make_V([(vb11, 1)]))
        W_list.append(make_W([(wc21, 1), (wc22, -1)]))

        # M3 = a11(b21-b22)
        U_list.append(make_U([(ua11, 1)]))
        V_list.append(make_V([(vb21, 1), (vb22, -1)]))
        W_list.append(make_W([(wc12, 1), (wc22, 1)]))

        # M4 = a22(b12-b11)
        U_list.append(make_U([(ua22, 1)]))
        V_list.append(make_V([(vb12, 1), (vb11, -1)]))
        W_list.append(make_W([(wc11, 1), (wc21, 1)]))

        # M5 = (a11+a12)b22
        U_list.append(make_U([(ua11, 1), (ua12, 1)]))
        V_list.append(make_V([(vb22, 1)]))
        W_list.append(make_W([(wc11, -1), (wc12, 1)]))

        # M6 = (a21-a11)(b11+b12)
        U_list.append(make_U([(ua21, 1), (ua11, -1)]))
        V_list.append(make_V([(vb11, 1), (vb12, 1)]))
        W_list.append(make_W([(wc22, 1)]))

        # M7 = (a12-a22)(b21+b22)
        U_list.append(make_U([(ua12, 1), (ua22, -1)]))
        V_list.append(make_V([(vb21, 1), (vb22, 1)]))
        W_list.append(make_W([(wc11, 1), (wc22, 1)]))

    strassen_2x2(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)

    # Remaining terms: schoolbook for all other (i,j,j',k) combinations
    # where j' != j and not already covered
    # Actually, let's just add schoolbook terms for everything not in the Strassen block
    # Strassen covers: i in {0,1}, j in {0,1}, j' in {0,1}, k in {0,1}
    # We need: all (i,j,j',k) where NOT (i in {0,1} AND j in {0,1} AND j' in {0,1} AND k in {0,1})
    
    covered = set()
    for i in [0, 1]:
        for j in [0, 1]:
            for jp in [0, 1]:
                for k in [0, 1]:
                    if j == jp:
                        covered.add((i, j, jp, k))
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    if j != jp:
                        continue  # These must be 0, so no term needed
                    if (i, j, jp, k) in covered:
                        continue
                    U_list.append(make_U([(u_idx(i, j), 1)]))
                    V_list.append(make_V([(v_idx(jp, k), 1)]))
                    W_list.append(make_W([(w_idx(k, i), 1)]))

    return {"U": U_list, "V": V_list, "W": W_list}


def _try_merge_terms(decomp, n, m, p):
    """Try to merge pairs of terms that have overlapping support."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0])
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Try merging term r1 and r2 by creating a new term that is their sum
    # and checking if we can eliminate one of them
    best_decomp = decomp
    best_rank = rank
    
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            # Check if merging is possible:
            # The sum U[r1]+U[r2], V[r1]+V[r2], W[r1]+W[r2] should still
            # contribute correctly when we remove both and add the merged term
            # This is only valid if the cross terms cancel out
            # i.e., U[r1]*V[r2]*W[r1] + U[r2]*V[r1]*W[r2] = 0 (as a tensor)
            # This is complex, so let's just try random linear combinations
            
            # Try: new_term = U[r1] + U[r2], V[r1] + V[r2], W[r1] + W[r2]
            # and remove r1 and r2, add new term
            # Check if this preserves the decomposition
            new_U = Ua.copy()
            new_V = Va.copy()
            new_W = Wa.copy()
            
            # Set r1 and r2 to zero
            new_U[:, r1] = 0
            new_V[:, r1] = 0
            new_W[:, r1] = 0
            new_U[:, r2] = 0
            new_V[:, r2] = 0
            new_W[:, r2] = 0
            
            # Add merged term at position r1
            new_U[:, r1] = Ua[:, r1] + Ua[:, r2]
            new_V[:, r1] = Va[:, r1] + Va[:, r2]
            new_W[:, r1] = Wa[:, r1] + Wa[:, r2]
            
            # Validate
            if _validate(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
                # Remove the zero column r2
                new_U = np.delete(new_U, r2, axis=1)
                new_V = np.delete(new_V, r2, axis=1)
                new_W = np.delete(new_W, r2, axis=1)
                if new_U.shape[1] < best_rank:
                    best_rank = new_U.shape[1]
                    best_decomp = {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
    
    return best_decomp


def construct():
    start = time.time()
    deadline = start + 110  # safety margin

    n, m, p = DIMENSIONS

    # Start with Strassen-based decomposition
    decomp = _build_strassen_based(n, m, p)
    
    if not _validate(decomp["U"], decomp["V"], decomp["W"], n, m, p):
        # Fallback to schoolbook
        decomp = _build_schoolbook(n, m, p)
    
    best_decomp = decomp
    best_rank = len(decomp["U"][0])
    
    # Try to improve by merging terms
    if time.time() < deadline - 5:
        improved = _try_merge_terms(decomp, n, m, p)
        if len(improved["U"][0]) < best_rank:
            best_decomp = improved
            best_rank = len(improved["U"][0])
    
    # Try another round of merging
    if time.time() < deadline - 5 and best_rank > 30:
        improved = _try_merge_terms(best_decomp, n, m, p)
        if len(improved["U"][0]) < best_rank:
            best_decomp = improved
            best_rank = len(improved["U"][0])
    
    if time.time() > deadline - 5:
        return best_decomp
    
    return best_decomp
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
