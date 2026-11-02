import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    def decompose(n, m, p):
        """
        Returns a list of (U_dict, V_dict, W_dict)
        U_dict: {i*m+j: coeff}
        V_dict: {j*p+k: coeff}
        W_dict: {k*n+i: coeff}
        """
        cols = []

        def add_strassen_2x2x2(i0, j0, k0):
            """
            Adds 7 Strassen terms for a 2x2x2 block at offsets (i0, j0, k0).
            """
            # M1 = (A11+A22)(B11+B22)
            cols.append((
                {i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
            ))

            # M2 = (A21+A22)B11
            cols.append((
                {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                {j0*p+k0: 1},
                {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1}
            ))

            # M3 = A11(B12-B22)
            cols.append((
                {i0*m+j0: 1},
                {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
                {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
            ))

            # M4 = A22(B21-B11)
            cols.append((
                {(i0+1)*m+(j0+1): 1},
                {(j0+1)*p+k0: 1, j0*p+k0: -1},
                {k0*n+i0: 1, k0*n+(i0+1): 1}
            ))

            # M5 = (A11+A12)B22
            cols.append((
                {i0*m+j0: 1, i0*m+(j0+1): 1},
                {(j0+1)*p+(k0+1): 1},
                {k0*n+i0: -1, (k0+1)*n+i0: 1}
            ))

            # M6 = (A21-A11)(B11+B12)
            cols.append((
                {(i0+1)*m+j0: 1, i0*m+j0: -1},
                {j0*p+k0: 1, j0*p+(k0+1): 1},
                {(k0+1)*n+(i0+1): 1}
            ))

            # M7 = (A12-A22)(B21+B22)
            cols.append((
                {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                {k0*n+i0: 1}
            ))

        def recurse(n_curr, m_curr, p_curr, i0, j0, k0):
            if time.time() > deadline:
                return

            # Base cases
            if n_curr == 1 and m_curr == 1 and p_curr == 1:
                u_idx = i0 * m + j0
                v_idx = j0 * p + k0
                w_idx = k0 * n + i0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

            # Check if we can apply Strassen 2x2x2
            if n_curr >= 2 and m_curr >= 2 and p_curr >= 2:
                add_strassen_2x2x2(i0, j0, k0)
                
                # Recurse on remainders
                if n_curr > 2:
                    recurse(n_curr - 2, m_curr, p_curr, i0 + 2, j0, k0)
                if m_curr > 2:
                    recurse(2, m_curr - 2, p_curr, i0, j0 + 2, k0)
                if p_curr > 2:
                    recurse(2, 2, p_curr - 2, i0, j0, k0 + 2)
                return

            # Fallback: split into smaller blocks
            if n_curr > 1:
                recurse(1, m_curr, p_curr, i0, j0, k0)
                recurse(n_curr - 1, m_curr, p_curr, i0 + 1, j0, k0)
                return
            elif m_curr > 1:
                recurse(n_curr, 1, p_curr, i0, j0, k0)
                recurse(n_curr, m_curr - 1, p_curr, i0, j0 + 1, k0)
                return
            elif p_curr > 1:
                recurse(n_curr, m_curr, 1, i0, j0, k0)
                recurse(n_curr, m_curr, p_curr - 1, i0, j0, k0 + 1)
                return
            else:
                # 1x1x1
                u_idx = i0 * m + j0
                v_idx = j0 * p + k0
                w_idx = k0 * n + i0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

        recurse(n, m, p, 0, 0, 0)
        return cols

    cols = decompose(n, m, p)
    
    rank = len(cols)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(cols):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
