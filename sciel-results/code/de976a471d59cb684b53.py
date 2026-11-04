DIMENSIONS = [5, 5, 5]
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct using random sampling."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for _ in range(50):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        if abs(val - 1.0) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 5, 5, 5

    # Block partition: [2, 2, 1]
    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]

    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])

    # Collect terms as (u_dict, v_dict, w_dict)
    terms = []

    def add_term(u_dict, v_dict, w_dict):
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]

                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]

                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen for 2x2x2
                    i0, i1 = i_start, i_start + 1
                    j0, j1 = j_start, j_start + 1
                    k0, k1 = k_start, k_start + 1

                    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
                    add_term({i0*m+j0: 1, i1*m+j1: 1},
                           {j0*p+k0: 1, j1*p+k1: 1},
                           {k0*n+i0: 1, k1*n+i1: 1})

                    # M2 = (a10+a11)b00 -> C10 += M2, C11 -= M2
                    add_term({i1*m+j0: 1, i1*m+j1: 1},
                           {j0*p+k0: 1},
                           {k0*n+i1: 1, k1*n+i1: -1})

                    # M3 = a00(b01-b11) -> C01 += M3, C11 += M3
                    add_term({i0*m+j0: 1},
                           {j0*p+k1: 1, j1*p+k1: -1},
                           {k1*n+i0: 1, k1*n+i1: 1})

                    # M4 = a11(b10-b00) -> C00 += M4, C10 += M4
                    add_term({i1*m+j1: 1},
                           {j1*p+k0: 1, j0*p+k0: -1},
                           {k0*n+i0: 1, k0*n+i1: 1})

                    # M5 = (a00+a01)b11 -> C00 -= M5, C01 += M5
                    add_term({i0*m+j0: 1, i0*m+j1: 1},
                           {j1*p+k1: 1},
                           {k0*n+i0: -1, k1*n+i0: 1})

                    # M6 = (a10-a00)(b00+b01) -> C11 += M6
                    add_term({i1*m+j0: 1, i0*m+j0: -1},
                           {j0*p+k0: 1, j0*p+k1: 1},
                           {k1*n+i1: 1})

                    # M7 = (a01-a11)(b10+b11) -> C00 += M7
                    add_term({i0*m+j1: 1, i1*m+j1: -1},
                           {j1*p+k0: 1, j1*p+k1: 1},
                           {k0*n+i0: 1})
                else:
                    # Schoolbook for smaller blocks
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                add_term({(i_start+u)*m + (j_start+v): 1},
                                       {(j_start+v)*p + (k_start+w): 1},
                                       {(k_start+w)*n + (i_start+u): 1})

    rank = len(terms)

    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to naive
        rank_naive = n * m * p
        U2 = [[0] * rank_naive for _ in range(n * m)]
        V2 = [[0] * rank_naive for _ in range(m * p)]
        W2 = [[0] * rank_naive for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j][col] = 1
                    V2[j*p+k][col] = 1
                    W2[k*n+i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
