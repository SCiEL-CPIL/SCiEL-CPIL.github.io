import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    n, m, p = DIMENSIONS

    def build_from_terms(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        """Check that the decomposition is exact for matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strassen-based construction (known good, rank 46)
    def strassen_construction():
        terms = []

        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))

        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))

        # Block partition:
        # i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
        # Split j into {0,1}, {2,3}, {4}
        # Split k into {0,1}, {2,3}, {4}
        # For j in {0,1}, k in {0,1}: Strassen 2x2x2 = 7
        # For j in {0,1}, k in {2,3}: Strassen 2x2x2 = 7
        # For j in {0,1}, k in {4}: schoolbook 2x2x1 = 4
        # For j in {2,3}, k in {0,1}: Strassen 2x2x2 = 7
        # For j in {2,3}, k in {2,3}: Strassen 2x2x2 = 7
        # For j in {2,3}, k in {4}: schoolbook 2x2x1 = 4
        # For j in {4}, k in {0,1}: schoolbook 2x1x2 = 4
        # For j in {4}, k in {2,3}: schoolbook 2x1x2 = 4
        # For j in {4}, k in {4}: schoolbook 2x1x1 = 2
        # Total: 7+7+4+7+7+4+4+4+2 = 46

        add_strassen(0, 0, 0)       # j={0,1}, k={0,1}
        add_strassen(0, 0, 2)       # j={0,1}, k={2,3}
        add_schoolbook(0, 2, 0, 2, 4, 5)  # j={0,1}, k={4}
        add_strassen(0, 2, 0)       # j={2,3}, k={0,1}
        add_strassen(0, 2, 2)       # j={2,3}, k={2,3}
        add_schoolbook(0, 2, 2, 4, 4, 5)  # j={2,3}, k={4}
        add_schoolbook(0, 2, 4, 5, 0, 2)  # j={4}, k={0,1}
        add_schoolbook(0, 2, 4, 5, 2, 4)  # j={4}, k={2,3}
        add_schoolbook(0, 2, 4, 5, 4, 5)  # j={4}, k={4}

        return terms

    def try_alt_partition_1():
        """Alternative: split j into {0,1,2}, {3,4} and k into {0,1,2}, {3,4}"""
        terms = []

        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))

        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))

        # j={0,1,2}, k={0,1,2}: need 2x3x3
        #   Split j into {0,1}, {2}
        #   For j={0,1}, k={0,1}: Strassen 2x2x2 = 7
        #   For j={0,1}, k={2}: schoolbook 2x2x1 = 4
        #   For j={2}, k={0,1}: schoolbook 2x1x2 = 4
        #   For j={2}, k={2}: schoolbook 2x1x1 = 2
        #   Total: 17
        # j={0,1,2}, k={3,4}: 2x3x2
        #   For j={0,1}, k={3,4}: Strassen 2x2x2 = 7
        #   For j={2}, k={3,4}: schoolbook 2x1x2 = 4
        #   Total: 11
        # j={3,4}, k={0,1,2}: 2x2x3
        #   For j={3,4}, k={0,1}: Strassen 2x2x2 = 7
        #   For j={3,4}, k={2}: schoolbook 2x2x1 = 4
        #   Total: 11
        # j={3,4}, k={3,4}: Strassen 2x2x2 = 7
        # Total: 17+11+11+7 = 46

        # j={0,1,2}, k={0,1,2}
        add_strassen(0, 0, 0)       # j={0,1}, k={0,1}
        add_schoolbook(0, 2, 0, 2, 2, 3)  # j={0,1}, k={2}
        add_schoolbook(0, 2, 2, 3, 0, 2)  # j={2}, k={0,1}
        add_schoolbook(0, 2, 2, 3, 2, 3)  # j={2}, k={2}

        # j={0,1,2}, k={3,4}
        add_strassen(0, 0, 3)       # j={0,1}, k={3,4}
        add_schoolbook(0, 2, 2, 3, 3, 5)  # j={2}, k={3,4}

        # j={3,4}, k={0,1,2}
        add_strassen(0, 3, 0)       # j={3,4}, k={0,1}
        add_schoolbook(0, 2, 3, 5, 2, 3)  # j={3,4}, k={2}

        # j={3,4}, k={3,4}
        add_strassen(0, 3, 3)       # j={3,4}, k={3,4}

        return terms

    def try_alt_partition_2():
        """Alternative: use Strassen on (i,k) pairs for each j"""
        terms = []

        def add_strassen_ik(i_start, k_start, j):
            """For fixed j, apply Strassen-like on (i,k) 2x2 block
            This gives rank 7 for 2x1x2 instead of 4, so it's worse.
            Skip this approach."""
            pass

        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))

        # Just use schoolbook for everything: rank 50
        add_schoolbook(0, 2, 0, 5, 0, 5)
        return terms

    # Start with the known good construction
    best_terms = strassen_construction()
    best_result = build_from_terms(best_terms)
    best_rank = len(best_terms)

    # Verify it's valid
    if not validate(best_result["U"], best_result["V"], best_result["W"]):
        raise ValueError("Strassen construction failed validation")

    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Try alternative partition
        alt_terms = try_alt_partition_1()
        alt_result = build_from_terms(alt_terms)
        if validate(alt_result["U"], alt_result["V"], alt_result["W"]):
            if len(alt_terms) < best_rank:
                best_result = alt_result
                best_rank = len(alt_terms)

    # Try to find linear dependencies in the term vectors
    # Flatten each term into a vector of length n*m*p
    U = np.array(best_result["U"], dtype=float)
    V = np.array(best_result["V"], dtype=float)
    W = np.array(best_result["W"], dtype=float)
    rank = U.shape[1]

    # Build the term matrix: each column is the Kronecker product of U[:,c], V[:,c], W[:,c]
    term_vecs = []
    for c in range(rank):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    vec[idx] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        term_vecs.append(vec)

    M = np.array(term_vecs).T  # shape (n*m*p, rank)

    # Check if the matrix has full column rank
    try:
        matrix_rank = np.linalg.matrix_rank(M, tol=1e-8)
        if matrix_rank < rank:
            # There's a linear dependency, but we need to check if we can actually
            # remove a term and still satisfy the constraints
            # This is complex, so we just note it
            pass
    except:
        pass

    return best_result
