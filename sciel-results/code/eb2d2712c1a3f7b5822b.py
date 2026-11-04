"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    n, m, p = DIMENSIONS
    
    # Partition sizes
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    # Offsets
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_local, v_local, w_local, i_off, j_off, k_off, n_b, m_b, p_b):
        """
        u_local: list of length n_b*m_b
        v_local: list of length m_b*p_b
        w_local: list of length p_b*n_b
        """
        # Map local indices to global indices
        # U index: i*m + j
        # Local u index: i_l * m_b + j_l
        # Global u index: (i_off + i_l) * m + (j_off + j_l)
        
        u_global = [0.0] * (n * m)
        v_global = [0.0] * (m * p)
        w_global = [0.0] * (p * n)

        for idx, val in enumerate(u_local):
            if val != 0:
                i_l = idx // m_b
                j_l = idx % m_b
                u_idx = (i_off + i_l) * m + (j_off + j_l)
                u_global[u_idx] = val

        for idx, val in enumerate(v_local):
            if val != 0:
                j_l = idx // p_b
                k_l = idx % p_b
                v_idx = (j_off + j_l) * p + (k_off + k_l)
                v_global[v_idx] = val

        for idx, val in enumerate(w_local):
            if val != 0:
                k_l = idx // n_b
                i_l = idx % n_b
                w_idx = (k_off + k_l) * n + (i_off + i_l)
                w_global[w_idx] = val

        U_cols.append(u_global)
        V_cols.append(v_global)
        W_cols.append(w_global)

    # Strassen 2x2x2 decomposition
    def get_strassen_2x2x2():
        terms = []
        # M1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        # M3
        terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]))
        # M4
        terms.append(([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 0, 0]))
        # M5
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]))
        # M6
        terms.append([(-1, 1, 0, 0), (1, 1, 0, 0), (0, 0, 0, 1)]) # Wait, M6 = (A10-A00)(B00+B10)
        # Correction for M6:
        # u: A10=1, A00=-1 -> [ -1, 0, 1, 0 ]
        # v: B00=1, B10=1 -> [ 1, 0, 1, 0 ]
        # w: C11 gets M6 -> [ 0, 0, 0, 1 ]
        terms[5] = ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1])
        
        # M7
        # u: A01=1, A11=-1 -> [ 0, 1, 0, -1 ]
        # v: B01=1, B11=1 -> [ 0, 1, 0, 1 ]
        # w: C00 gets M7 -> [ 1, 0, 0, 0 ]
        terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]))
        
        return terms

    # Schoolbook decomposition for n_b x m_b x p_b
    def get_schoolbook(n_b, m_b, p_b):
        terms = []
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u = [0.0] * (n_b * m_b)
                    v = [0.0] * (m_b * p_b)
                    w = [0.0] * (p_b * n_b)
                    
                    u[i * m_b + j] = 1.0
                    v[j * p_b + k] = 1.0
                    w[k * n_b + i] = 1.0
                    
                    terms.append((u, v, w))
        return terms

    for i_idx, n_b in enumerate(n_parts):
        if time.time() > deadline: break
        for j_idx, m_b in enumerate(m_parts):
            if time.time() > deadline: break
            for k_idx, p_b in enumerate(p_parts):
                if time.time() > deadline: break
                
                i_off = n_offsets[i_idx]
                j_off = m_offsets[j_idx]
                k_off = p_offsets[k_idx]

                if n_b == 2 and m_b == 2 and p_b == 2:
                    terms = get_strassen_2x2x2()
                else:
                    terms = get_schoolbook(n_b, m_b, p_b)

                for u_l, v_l, w_l in terms:
                    add_term(u_l, v_l, w_l, i_off, j_off, k_off, n_b, m_b, p_b)

    rank = len(U_cols)
    
    # Convert to lists of lists for output
    U = [col for col in U_cols]
    V = [col for col in V_cols]
    W = [col for col in W_cols]
    
    # Transpose to match required shape (n*m, rank)
    # U_cols is a list of vectors of length n*m.
    # We need U[row][col].
    U_out = [[U[c][r] for c in range(rank)] for r in range(n*m)]
    V_out = [[V[c][r] for c in range(rank)] for r in range(m*p)]
    W_out = [[W[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
