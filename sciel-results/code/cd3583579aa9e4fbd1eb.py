import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _build_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    Indices:
    U: 2x2 (4 rows)
    V: 2x2 (4 rows)
    W: 2x2 (4 rows)
    """
    # Strassen's 7 terms
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Term 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    W[1][0] = -1; W[2][0] = -1 # Wait, C11 gets M1, C22 gets M1.
    # C11 (0) = M1 ...
    # C22 (3) = M1 ...
    # So W[0][0]=1, W[3][0]=1.
    # What about the other signs?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Reset W
    W = [[0]*7 for _ in range(4)]
    
    # T1
    W[0][0] = 1
    W[3][0] = 1
    
    # T2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    W[3][1] = -1
    
    # T3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # T4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # T5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # T6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # T7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication.
    Rank 4.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)] # 2x1 B matrix has 2 entries
    W = [[0]*4 for _ in range(2)] # 2x1 C matrix has 2 entries
    
    # C1 = A11 B11 + A12 B21
    # C2 = A21 B11 + A22 B21
    
    # T1: A11 B11 -> C1
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # T2: A12 B21 -> C1
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # T3: A21 B11 -> C2
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # T4: A22 B21 -> C2
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W

def _build_223():
    """
    Builds 2x2x3 decomposition from 222 (Strassen) and 221 (Schoolbook).
    Rank 11.
    """
    U1, V1, W1 = _build_strassen_222()
    U2, V2, W2 = _build_schoolbook_221()
    
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)] # 2x3 B has 6 entries
    W = [[0]*11 for _ in range(6)] # 2x3 C has 6 entries
    
    # Map 222 to k=0,1
    for a in range(4):
        for r in range(7):
            U[a][r] = U1[a][r]
            
    v_map = [0, 1, 3, 4]
    for b in range(4):
        for r in range(7):
            V[v_map[b]][r] = V1[b][r]
            
    w_map = [0, 1, 3, 4]
    for c in range(4):
        for r in range(7):
            W[w_map[c]][r] = W1[c][r]
            
    # Map 221 to k=2
    for a in range(4):
        for r in range(4):
            U[a][r + 7] = U2[a][r]
            
    v_map2 = [2, 5]
    for b in range(2):
        for r in range(4):
            V[v_map2[b]][r + 7] = V2[b][r]
            
    w_map2 = [2, 5]
    for c in range(2):
        for r in range(4):
            W[w_map2[c]][r + 7] = W2[c][r]
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    def make_vec(dim, indices_vals):
        vec = [0] * dim
        for idx, val in indices_vals:
            vec[idx] = val
        return vec

    # Pre-compute 223 decomposition
    U223, V223, W223 = _build_223()
    
    # 1. M_1,4,6 (i=2)
    for j in range(4):
        for k in range(6):
            u = make_vec(U_dim, [(8+j, 1)])
            v = make_vec(V_dim, [(j*6+k, 1)])
            w = make_vec(W_dim, [(k*3+2, 1)])
            add_term(u, v, w)
            
    # 2. M_2,4,6 (i=0,1)
    for k_block in range(2):
        k_offset = k_block * 3
        for j_block in range(2):
            j_offset = j_block * 2
            
            for r in range(11):
                u_vec = make_vec(U_dim, [])
                v_vec = make_vec(V_dim, [])
                w_vec = make_vec(W_dim, [])
                
                u_map = [
                    0*4 + j_offset,
                    0*4 + j_offset + 1,
                    1*4 + j_offset,
                    1*4 + j_offset + 1
                ]
                for a in range(4):
                    if U223[a][r] != 0:
                        u_vec[u_map[a]] = U223[a][r]
                        
                v_map = [
                    j_offset * 6 + k_offset + 0,
                    j_offset * 6 + k_offset + 1,
                    j_offset * 6 + k_offset + 2,
                    (j_offset+1) * 6 + k_offset + 0,
                    (j_offset+1) * 6 + k_offset + 1,
                    (j_offset+1) * 6 + k_offset + 2
                ]
                for b in range(6):
                    if V223[b][r] != 0:
                        v_vec[v_map[b]] = V223[b][r]
                        
                w_map = [
                    (k_offset+0)*3 + 0,
                    (k_offset+1)*3 + 0,
                    (k_offset+2)*3 + 0,
                    (k_offset+0)*3 + 1,
                    (k_offset+1)*3 + 1,
                    (k_offset+2)*3 + 1
                ]
                for c in range(6):
                    if W223[c][r] != 0:
                        w_vec[w_map[c]] = W223[c][r]
                        
                add_term(u_vec, v_vec, w_vec)
                
    rank = len(terms)
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    for r, (u, v, w) in enumerate(terms):
        for a in range(U_dim):
            if u[a] != 0:
                U[a][r] = u[a]
        for b in range(V_dim):
            if v[b] != 0:
                V[b][r] = v[b]
        for c in range(W_dim):
            if w[c] != 0:
                W[c][r] = w[c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
