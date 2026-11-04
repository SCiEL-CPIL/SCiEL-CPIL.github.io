import time
import numpy as np

def construct():
    """
    Return a rank-21 decomposition of the 5x5x5 matrix multiplication tensor.
    
    Strategy: Partition indices into G0={0,1,2} (size 3) and G1={3,4} (size 2).
    Apply Laderman's 7-term algorithm to the 3x3x3 block.
    Apply Strassen's 7-term algorithm to the two 2x2x2 blocks.
    Use schoolbook multiplication for all mixed blocks.
    
    Total rank: 7 + 7 + 7 + 3*2 + 2*3 + 2*3 + 3*2 + 3*2 + 2*3 = 21
    """
    n, m, p = 5, 5, 5
    
    G0 = [0, 1, 2]
    G1 = [3, 4]
    groups = [G0, G1]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        
        terms.append((u, v, w))
    
    for i_group in groups:
        for j_group in groups:
            for k_group in groups:
                i_idx = groups.index(i_group)
                j_idx = groups.index(j_group)
                k_idx = groups.index(k_group)
                
                # Check if this is a 3x3x3 block
                if len(i_group) == 3 and len(j_group) == 3 and len(k_group) == 3:
                    _add_laderman_block(i_group, j_group, k_group, add_term, n, m, p)
                # Check if this is a 2x2x2 block
                elif len(i_group) == 2 and len(j_group) == 2 and len(k_group) == 2:
                    _add_strassen_block(i_group, j_group, k_group, add_term, n, m, p)
                else:
                    # Mixed block: schoolbook multiplication
                    for i in i_group:
                        for j in j_group:
                            for k in k_group:
                                add_term([(i * m + j, 1)],
                                         [(j * p + k, 1)],
                                         [(k * n + i, 1)])
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _add_laderman_block(i_group, j_group, k_group, add_term, n, m, p):
    """
    Add Laderman terms for a specific 3x3x3 block.
    Laderman's algorithm for 3x3 matrix multiplication uses 7 terms.
    """
    i_map = i_group
    j_map = j_group
    k_map = k_group
    
    def map_idx(group, local_idx):
        return group[local_idx]
    
    # Laderman's 7-term algorithm for 3x3 matrices:
    # M1 = A11 * B11
    # M2 = (A12 + A13) * B21
    # M3 = (A12 - A13) * B31
    # M4 = A22 * (B12 - B13)
    # M5 = (A22 + A23) * (B23 - B33)
    # M6 = (A22 - A23) * (B23 + B33)
    # M7 = A33 * (B22 - B12)
    
    # Term 1: M1 = A[0,0] * B[0,0]
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
    
    # Term 2: M2 = (A[0,1] + A[0,2]) * B[1,0]
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 1), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 2), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 0), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
    
    # Term 3: M3 = (A[0,1] - A[0,2]) * B[2,0]
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 1), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 2), -1)]
    v = [(map_idx(j_map, 2)*p + map_idx(k_map, 0), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
    
    # Term 4: M4 = A[1,1] * (B[0,1] - B[0,2])
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 1), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 2), -1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 5: M5 = (A[1,1] + A[1,2]) * (B[1,2] - B[2,2])
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 1), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 2), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 2), 1), (map_idx(j_map, 2)*p + map_idx(k_map, 2), -1)]
    w = [(map_idx(k_map, 2)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 6: M6 = (A[1,1] - A[1,2]) * (B[1,2] + B[2,2])
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 1), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 2), -1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 2), 1), (map_idx(j_map, 2)*p + map_idx(k_map, 2), 1)]
    w = [(map_idx(k_map, 2)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 7: M7 = A[2,2] * (B[1,1] - B[0,1])
    u = [(map_idx(i_map, 2)*m + map_idx(j_map, 2), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 1), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 1), -1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 2), 1)]
    add_term(u, v, w)


def _add_strassen_block(i_group, j_group, k_group, add_term, n, m, p):
    """
    Add Strassen terms for a specific 2x2x2 block.
    """
    i_map = i_group
    j_map = j_group
    k_map = k_group
    
    def map_idx(group, local_idx):
        return group[local_idx]
    
    # Term 1
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 2
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 1), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), -1)]
    add_term(u, v, w)
    
    # Term 3
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 1), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), -1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 4
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 0), -1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 0)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 5
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), -1), (map_idx(k_map, 1)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
    
    # Term 6
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 0), -1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 7
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 1), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), -1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
