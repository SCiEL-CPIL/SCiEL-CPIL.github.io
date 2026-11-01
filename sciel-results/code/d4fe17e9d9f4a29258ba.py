"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank from 50,
    then applies local search (gradient descent on the tensor factorization
    residual) to attempt further rank reduction via removing dependent terms
    and re-optimizing the remaining factors.
    
    The decomposition uses the trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # n=2, m=5, p=5

    # Build the full tensor T[a, b, c] where:
    #   a = i*m + j  (row of A, col of A)
    #   b = j*p + k  (row of B, col of B)
    #   c = k*n + i  (output, trace convention: k*n + i)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def build_decomposition_from_terms(terms):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) terms."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r] = val

        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err

    def try_reduce_rank(U, V, W, max_reduction=10, steps_per_trial=80):
        """Try to reduce rank by removing dependent terms and re-optimizing."""
        R = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = R

        # Compute norms to identify potentially removable terms
        norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
        sorted_indices = np.argsort(norms)

        for reduction in range(1, max_reduction + 1):
            if time.time() > deadline - 15:
                break

            new_R = R - reduction
            if new_R < 1:
                break

            # Keep the largest new_R terms
            keep_indices = sorted_indices[-new_R:]
            keep_indices = np.sort(keep_indices)

            U_new = U[:, keep_indices].copy()
            V_new = V[:, keep_indices].copy()
            W_new = W[:, keep_indices].copy()

            # Local optimization with the reduced rank
            lr = 0.003
            momentum = 0.9
            vU = np.zeros_like(U_new)
            vV = np.zeros_like(V_new)
            vW = np.zeros_like(W_new)

            for step in range(steps_per_trial):
                if time.time() > deadline - 15:
                    break

                Rec = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                diff = T - Rec

                gU = np.einsum('abc,br,cr->ar', diff, V_new, W_new)
                gV = np.einsum('abc,ar,cr->br', diff, U_new, W_new)
                gW = np.einsum('abc,ar,br->cr', diff, U_new, V_new)

                vU = momentum * vU + (1 - momentum) * gU
                vV = momentum * vV + (1 - momentum) * gV
                vW = momentum * vW + (1 - momentum) * gW

                U_new -= lr * vU
                V_new -= lr * vV
                W_new -= lr * vW

            # Verify the reduced decomposition
            valid, err = verify_decomposition(U_new, V_new, W_new)
            if valid:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_R
            else:
                break

        return best_U, best_V, best_W, best_rank

    # Construct Strassen-based decomposition
    terms = []

    def add_strassen(i_start, j_start, k_start):
        """Add 7 Strassen terms for a 2x2x2 block at (i_start, j_start, k_start)."""
        # M1 = (A11+A22)(B11+B22)
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))

        # M2 = (A21+A22)B11
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))

        # M3 = A11(B21-B11)
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))

        # M4 = A22(B22-B12)
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))

        # M5 = (A11+A12)B22
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))

        # M6 = (A21-A11)(B11+B12)
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))

        # M7 = (A12-A22)(B21+B22)
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook_term(i, j, k):
        """Add a single schoolbook term."""
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))

    # Decomposition plan for 2x5x5:
    # i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    #
    # Use Strassen where we have 2x2x2 blocks (i_start must be 0 since n=2):
    # j in {0,1}, k in {0,1}: Strassen(0, 0, 0) -> 7 terms
    # j in {0,1}, k in {2,3}: Strassen(0, 0, 2) -> 7 terms
    # j in {0,1}, k in {4}:    schoolbook for i=0,1 -> 2 terms
    # j in {2,3}, k in {0,1}: Strassen(0, 2, 0) -> 7 terms
    # j in {2,3}, k in {2,3}: Strassen(0, 2, 2) -> 7 terms
    # j in {2,3}, k in {4}:    schoolbook for i=0,1 -> 2 terms
    # j in {4},    k in {0,1,2,3,4}: schoolbook for i=0,1 -> 10 terms
    #
    # Total: 7+7+2+7+7+2+10 = 42 terms

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    for i in range(n):
        add_schoolbook_term(i, 0, 4)
        add_schoolbook_term(i, 1, 4)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    for i in range(n):
        add_schoolbook_term(i, 2, 4)
        add_schoolbook_term(i, 3, 4)
    for i in range(n):
        for k in range(p):
            add_schoolbook_term(i, 4, k)

    # Build the decomposition
    U, V, W = build_decomposition_from_terms(terms)
    initial_rank = len(terms)

    # Verify the Strassen decomposition
    valid, err = verify_decomposition(U, V, W)
    if not valid:
        # Fallback to schoolbook if Strassen is incorrect
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_schoolbook_term(i, j, k)
        U, V, W = build_decomposition_from_terms(terms)
        initial_rank = len(terms)

    # Try to reduce rank via local search
    if time.time() < deadline - 20:
        U, V, W, best_rank = try_reduce_rank(U, V, W, max_reduction=8, steps_per_trial=80)
    else:
        best_rank = initial_rank

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
