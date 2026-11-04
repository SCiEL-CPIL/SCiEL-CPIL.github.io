import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_rank(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    def init_vecs():
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        return u, v, w

    # Strassen terms for 2x2x2
    # Local indices: A(i,j), B(j,k), C(i,k)
    # i,j,k in 0..1
    
    strassen_terms = [
        # P1 = (a00+a11)(b00+b11)
        {
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        },
        # P2 = (a10+a11)b00
        {
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(1,0): 1, (1,1): -1}
        },
        # P3 = (a00-a10)b10
        {
            'u': {(0,0): 1, (1,0): -1},
            'v': {(1,0): 1},
            'w': {(0,1): 1, (1,1): 1}
        },
        # P4 = a11(b11-b00)
        {
            'u': {(1,1): 1},
            'v': {(1,1): 1, (0,0): -1},
            'w': {(0,0): 1, (1,0): 1}
        },
        # P5 = (a00+a01)b11
        {
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,0): -1, (0,1): 1}
        },
        # P6 = (a11-a01)(b00+b01)
        {
            'u': {(1,1): 1, (0,1): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(1,1): 1}
        },
        # P7 = (a01-a11)(b10+b11)
        {
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(0,0): 1}
        }
    ]
    
    def add_strassen(a_j_off, b_j_off, b_k_off):
        # a_j_off: offset for A columns (0 or 2)
        # b_j_off: offset for B rows (0 or 2)
        # b_k_off: offset for B columns (0, 2, 4)
        
        for term in strassen_terms:
            u, v, w = init_vecs()
            # Fill U
            for (i, j), coeff in term['u'].items():
                u[i*m + (a_j_off + j)] = coeff
            # Fill V
            for (j, k), coeff in term['v'].items():
                v[(b_j_off + j)*p + (b_k_off + k)] = coeff
            # Fill W
            for (i, k), coeff in term['w'].items():
                w[(b_k_off + k)*n + i] = coeff
            add_rank(u, v, w)

    def add_trivial_2x2x1(a_j_off, b_j_off, b_k_off):
        # 2x2x1 trivial rank 4
        # A is 2x2, B is 2x1, C is 2x1
        # C_i0 = A_i0 B_00 + A_i1 B_10
        
        terms = [
            # A_00 B_00 -> C_00
            {'u': {(0,0): 1}, 'v': {(0,0): 1}, 'w': {(0,0): 1}},
            # A_01 B_10 -> C_00
            {'u': {(0,1): 1}, 'v': {(1,0): 1}, 'w': {(0,0): 1}},
            # A_10 B_00 -> C_10
            {'u': {(1,0): 1}, 'v': {(0,0): 1}, 'w': {(1,0): 1}},
            # A_11 B_10 -> C_10
            {'u': {(1,1): 1}, 'v': {(1,0): 1}, 'w': {(1,0): 1}},
        ]
        
        for term in terms:
            u, v, w = init_vecs()
            for (i, j), coeff in term['u'].items():
                u[i*m + (a_j_off + j)] = coeff
            for (j, k), coeff in term['v'].items():
                v[(b_j_off + j)*p + (b_k_off + k)] = coeff
            for (i, k), coeff in term['w'].items():
                w[(b_k_off + k)*n + i] = coeff
            add_rank(u, v, w)

    # Block 1: A_L (j=0,1) with B_T (j=0,1)
    # B_T columns k=0,1 (Strassen), k=2,3 (Strassen), k=4 (Trivial)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_trivial_2x2x1(0, 0, 4)
    
    # Block 2: A_R (j=2,3) with B_B (j=2,3)
    # B_B columns k=0,1 (Strassen), k=2,3 (Strassen), k=4 (Trivial)
    add_strassen(2, 2, 0)
    add_strassen(2, 2, 2)
    add_trivial_2x2x1(2, 2, 4)
    
    rank = len(U_list)
    
    # Convert to list of rows format required by interface
    U_out = [[0.0]*rank for _ in range(n*m)]
    V_out = [[0.0]*rank for _ in range(m*p)]
    W_out = [[0.0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U_out[i][r] = U_list[r][i]
        for j in range(m*p):
            V_out[j][r] = V_list[r][j]
        for k in range(p*n):
            W_out[k][r] = W_list[r][k]
            
    # Convert to integers if possible to satisfy "integer/half-integer" constraint
    # The coefficients are -1, 0, 1, so they are integers.
    
    return {"U": U_out, "V": V_out, "W": W_out}
