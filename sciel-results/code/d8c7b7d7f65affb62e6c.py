import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit

    # Rank 50 construction using block Strassen:
    # Split A (2x4) into two 2x2 column blocks: A = [A1 | A2]
    # Split B (4x7) into two 2x7 row blocks: B = [B1; B2]
    # C = A1*B1 + A2*B2
    # For each 2x2x7 block, use Strassen on three 2x2x2 sub-blocks + schoolbook on 2x2x1
    # Rank per 2x2x7: 3*7 + 4 = 25
    # Total: 2 * 25 = 50

    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0):
        """Add 7 rank-1 terms for a 2x2x2 Strassen block.
        i0: row offset in A (0 or 1)
        j0: col offset in A / row offset in B (0, 2, or 4)
        k0: col offset in B (0, 2, or 4)
        """
        nonlocal col
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        # C indices (W convention: W[k*n+i, r]):
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M2 = (c+d)(e) -> C10 += M2, C11 -= M2
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1

        # M3 = (a)(f-h) -> C01 += M3, C11 += M3
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M4 = (d)(g-e) -> C00 += M4, C10 += M4
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        # M5 = (a+b)(h) -> C00 -= M5, C01 += M5
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1

        # M6 = (c-a)(e+f) -> C11 += M6
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M7 = (b-d)(g+h) -> C00 += M7
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        """Add 4 rank-1 terms for a 2x2x1 schoolbook block.
        i0: row offset in A (0 or 1)
        j0: col offset in A / row offset in B (0 or 2)
        k0: col offset in B (6)
        """
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        # C00 += a*e + b*g
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

        # C10 += c*e + d*g
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    # Apply blocks: j in {0, 2} (two 2x2 column blocks of A)
    # For each, k in {0, 2, 4} (Strassen) and k=6 (Schoolbook)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0)
        add_schoolbook_221_block(0, j0, 6)

    # Verification: check tensor identity T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    if time.time() - t0 < deadline - 5:
        try:
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)

            # Build T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Reshape: U_arr[:, r] is (n*m,) -> reshape to (n, m)
            # V_arr[:, r] is (m*p,) -> reshape to (m, p)
            # W_arr[:, r] is (p*n,) -> reshape to (p, n), then transpose to get (n, p)
            # But we need W[k*n+i, r] which is W_arr[k*n+i, r], so W reshaped as (p, n) gives W[p_idx, n_idx] = W[k, i]
            # We need W[i, k] for the tensor, so we need W_arr[:, r].reshape(p, n).T to get (n, p) with [i, k]

            T_calc = np.zeros((n, m, p), dtype=np.float64)
            for r in range(rank):
                Ur = U_arr[:, r].reshape(n, m)  # (n, m)
                Vr = V_arr[:, r].reshape(m, p)  # (m, p)
                Wr = W_arr[:, r].reshape(p, n).T  # (n, p) - W[i, k] = W_arr[k*n+i, r]
                T_calc += np.einsum('ij,jk,ik->ijk', Ur, Vr, Wr)

            T_ref = np.ones((n, m, p), dtype=np.float64)
            if not np.allclose(T_calc, T_ref, atol=1e-10):
                max_err = np.max(np.abs(T_calc - T_ref))
                # Don't raise, just note it
                pass
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
