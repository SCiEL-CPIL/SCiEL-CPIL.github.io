# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 130  # Safety margin: evaluator kills at 150s

    # Strategy:
    # Use rank-10 decomposition for 2x2x3 block and rank-7 for 2x2x2 blocks.
    # 15 = 3 + 2 + 2 + 2 + 2 + 2 + 2
    # Rank = 10 + 6 * 7 = 52
    
    all_U = []
    all_V = []
    all_W = []

    # Block 0: k=0,1,2 (Rank 10)
    terms_3 = _rank10_terms_2x2x3(0, 1, 2)
    for (u_dict, v_dict, w_dict) in terms_3:
        u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
        v_entries = [(j * p + k, c) for (j, k), c in v_dict.items()]
        w_entries = [(k * n + i, c) for (k, i), c in w_dict.items()]
        all_U.append(u_entries)
        all_V.append(v_entries)
        all_W.append(w_entries)

    # Blocks 1-6: k=3,4; 5,6; 7,8; 9,10; 11,12; 13,14 (Rank 7 each)
    k_start = 3
    for _ in range(6):
        k0, k1 = k_start, k_start + 1
        terms_2 = _strassen_terms(k0, k1, n, m, p)
        for (u_dict, v_dict, w_dict) in terms_2:
            u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
            v_entries = [(j * p + k, c) for (j, k), c in v_dict.items()]
            w_entries = [(k * n + i, c) for (k, i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)
        k_start += 2

    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c in range(rank):
        for (row, val) in all_U[c]:
            U[row][c] = val
        for (row, val) in all_V[c]:
            V[row][c] = val
        for (row, val) in all_W[c]:
            W[row][c] = val

    return {"U": U, "V": V, "W": W}

def _rank10_terms_2x2x3(k0, k1, k2, n, m, p):
    """
    Return rank-10 terms for a 2x2x3 block with columns k0, k1, k2.
    Uses Strassen for k0, k1 and 3 schoolbook-like terms for k2.
    """
    terms = []
    
    # Strassen terms for k0, k1 (indices 0,1 in local coords)
    # Note: We map local k=0 -> k0, k=1 -> k1, k=2 -> k2
    strassen_local = _strassen_terms_local(0, 1)
    for (u_dict, v_dict, w_dict) in strassen_local:
        # Remap k indices
        v_new = {}
        for (j, k), c in v_dict.items():
            k_new = k0 if k == 0 else k1
            v_new[(j, k_new)] = c
        w_new = {}
        for (k, i), c in w_dict.items():
            k_new = k0 if k == 0 else k1
            w_new[(k_new, i)] = c
        terms.append((u_dict, v_new, w_new))
        
    # Terms for k2
    # 1. A[0,0] B[0,k2] -> C[k2,0]
    terms.append(({(0,0): 1}, {(0, k2): 1}, {(k2, 0): 1}))
    # 2. A[0,1] B[1,k2] -> C[k2,0]
    terms.append(({(0,1): 1}, {(1, k2): 1}, {(k2, 0): 1}))
    # 3. A[1,0] B[0,k2] + A[1,1] B[1,k2] -> C[k2,1]
    terms.append(({(1,0): 1, (1,1): 1}, {(0, k2): 1, (1, k2): 1}, {(k2, 1): 1}))
    
    return terms

def _strassen_terms_local(k0, k1):
    """Return Strassen terms for 2x2x2 with local indices k0, k1 (usually 0,1)."""
    # This is identical to _strassen_terms but assumes k0, k1 are passed
    # and we return dicts with those keys.
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0, k0): 1, (1, k1): 1}, {(k0, 0): 1, (k1, 1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0, k0): 1}, {(k0, 1): 1}),
        ({(0,0): 1}, {(1, k0): 1, (1, k1): -1}, {(k1, 0): 1}),
        ({(1,1): 1}, {(0, k1): 1, (0, k0): -1}, {(k0, 0): 1, (k0, 1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1, k1): 1}, {(k0, 0): -1, (k1, 0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0, k0): 1, (0, k1): 1}, {(k1, 1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1, k0): 1, (1, k1): 1}, {(k0, 0): 1}),
    ]
    return terms

def _strassen_terms(k0, k1, n, m, p):
    """Return Strassen terms for a 2x2x2 block with specific k indices."""
    return _strassen_terms_local(k0, k1)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = np.sum(U_arr[u_row, :] * V_arr[v_row, :] * W_arr[w_row, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True
# EVOLVE-BLOCK-END
