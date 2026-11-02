import numpy as np

# EVOLVE-BLOCK-START

# The 7 Strassen linear combinations for 2x2 matrix multiplication.
# Each combo is a list of (row, col, sign) entries.
_A_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (0, 0, -1)],
    [(0, 1, 1), (1, 1, -1)]
]

_B_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(0, 1, 1), (1, 1, -1)],
    [(1, 0, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (1, 1, 1)]
]

_C_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 1, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1)]
]

def _generate_strassen_4x4():
    """
    Generates U, V, W for 4x4 Strassen decomposition directly.
    Uses the recursive Kronecker product structure:
    T_4 = T_2 (x) T_2, giving rank 7*7 = 49.
    
    Indexing convention:
    - U[i*n + k, r] for A element (i,k)
    - V[j*n + l, r] for B element (j,l)
    - W[k*n + i, r] for C element (k,i) per trace convention
    """
    n = 4
    rank = 49
    U = np.zeros((n * n, rank), dtype=int)
    V = np.zeros((n * n, rank), dtype=int)
    W = np.zeros((n * n, rank), dtype=int)
    
    for t_outer in range(7):
        for t_inner in range(7):
            r_idx = t_outer * 7 + t_inner
            
            for (br, bc, sign_a) in _A_COMBOS[t_outer]:
                for (ir, ik, sign_inner_a) in _A_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    U[i_global * n + k_global, r_idx] += sign_a * sign_inner_a
            
            for (br, bc, sign_b) in _B_COMBOS[t_outer]:
                for (ir, ik, sign_inner_b) in _B_COMBOS[t_inner]:
                    j_global = br * 2 + ir
                    l_global = bc * 2 + ik
                    V[j_global * n + l_global, r_idx] += sign_b * sign_inner_b
            
            for (br, bc, sign_c) in _C_COMBOS[t_outer]:
                for (ir, ik, sign_inner_c) in _C_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    j_global = bc * 2 + ik
                    W[j_global * n + i_global, r_idx] += sign_c * sign_inner_c

    return U, V, W

def construct():
    n, m, p = 4, 4, 4
    U, V, W = _generate_strassen_4x4()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
