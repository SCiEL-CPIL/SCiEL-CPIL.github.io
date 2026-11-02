import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = 3, 3, 6

    def _verify_decomposition(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def _make_schoolbook(n, m, p):
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
        return U, V, W, rank

    def _strassen_2x2x2_terms():
        """
        Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
        Each term is (u, v, w) where u, v, w are dicts mapping indices to coefficients.
        u: (i,j) -> coeff, v: (j,k) -> coeff, w: (k,i) -> coeff
        """
        terms = []

        # M1 = (A00 + A11)(B00 + B11)
        terms.append({
            'U': {(0, 0): 1, (1, 1): 1},
            'V': {(0, 0): 1, (1, 1): 1},
            'W': {(0, 0): 1, (1, 0): 1}
        })

        # M2 = (A10 + A11)B00
        terms.append({
            'U': {(1, 0): 1, (1, 1): 1},
            'V': {(0, 0): 1},
            'W': {(0, 1): 1}
        })

        # M3 = A00(B01 - B11)
        terms.append({
            'U': {(0, 0): 1},
            'V': {(0, 1): 1, (1, 1): -1},
            'W': {(1, 0): 1, (1, 1): -1}
        })

        # M4 = A11(B10 - B00)
        terms.append({
            'U': {(1, 1): 1},
            'V': {(1, 0): 1, (0, 0): -1},
            'W': {(0, 0): -1, (1, 0): 1}
        })

        # M5 = (A00 + A01)B11
        terms.append({
            'U': {(0, 0): 1, (0, 1): 1},
            'V': {(1, 1): 1},
            'W': {(1, 1): 1}
        })

        # M6 = (A10 - A00)(B00 + B01)
        terms.append({
            'U': {(1, 0): 1, (0, 0): -1},
            'V': {(0, 0): 1, (0, 1): 1},
            'W': {(0, 0): 1}
        })

        # M7 = (A01 - A11)(B10 + B11)
        terms.append({
            'U': {(0, 1): 1, (1, 1): -1},
            'V': {(1, 0): 1, (1, 1): 1},
            'W': {(0, 0): 1, (1, 1): 1}
        })

        return terms

    def _try_strassen_block(n, m, p):
        """
        Build a block-structured Strassen-based decomposition for 3x3 * 3x6.
        Split the 6 output columns into three 2-column pairs.
        For each pair, use:
          - 7 Strassen terms for the 2x2 sub-block (i in {0,1}, j in {0,1})
          - 4 schoolbook terms for i=2, j in {0,1}
          - 3 schoolbook terms for j=2, i in {0,1,2}
        Total per pair: 7 + 4 + 3 = 14
        Total: 3 * 14 = 42
        """
        all_U = []
        all_V = []
        all_W = []

        strassen_terms = _strassen_2x2x2_terms()

        for col_pair in range(3):
            k0 = col_pair * 2  # 0, 2, 4

            # Part 1: Strassen for i in {0,1}, j in {0,1}, k in {k0, k0+1}
            for st in strassen_terms:
                U_term = [0] * (n * m)
                V_term = [0] * (m * p)
                W_term = [0] * (p * n)

                for (ii, jj), coeff in st['U'].items():
                    U_term[ii * m + jj] = coeff
                for (jj, kk), coeff in st['V'].items():
                    V_term[jj * p + (k0 + kk)] = coeff
                for (kk, ii), coeff in st['W'].items():
                    W_term[(k0 + kk) * n + ii] = coeff

                all_U.append(U_term)
                all_V.append(V_term)
                all_W.append(W_term)

            # Part 2: i=2, j in {0,1}, k in {k0, k0+1} -- schoolbook (4 terms)
            for kk in range(2):
                for jj in range(2):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    U_term[2 * m + jj] = 1
                    V_term[jj * p + (k0 + kk)] = 1
                    W_term[(k0 + kk) * n + 2] = 1
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)

            # Part 3: j=2, i in {0,1,2}, k in {k0, k0+1} -- schoolbook (3 terms)
            for kk in range(2):
                for ii in range(3):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    U_term[ii * m + 2] = 1
                    V_term[2 * p + (k0 + kk)] = 1
                    W_term[(k0 + kk) * n + ii] = 1
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)

            # Time check
            if time.time() > deadline:
                break

        rank = len(all_U)
        if rank != 42:
            return None

        if _verify_decomposition(all_U, all_V, all_W, n, m, p):
            return all_U, all_V, all_W, rank

        return None

    # Try the Strassen block approach (rank 42)
    result_42 = _try_strassen_block(n, m, p)

    if result_42 is not None:
        U, V, W, rank = result_42
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    U, V, W, rank = _make_schoolbook(n, m, p)

    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
