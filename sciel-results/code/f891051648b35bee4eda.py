"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    """
    terms = []
    # P1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    # P2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    # P3
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    # P4
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    terms.append((u4, v4, w4))
    # P5
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    terms.append((u5, v5, w5))
    # P6
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    # P7
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    return terms

def _construct_2x3x9_strassen():
    """
    Construct decomposition for 2x3 * 3x9 using Strassen on 2x2 blocks.
    Rank 30: 4*7 (Strassen) + 1 (2x2x1) + 1 (2x1x9)
    """
    n, m, p = 2, 3, 9
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Part 1: A_{j<2} * B_{j<2}
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    k_single = [8]
    strassen_terms = _strassen_2x2x2_terms()
    
    for (k1, k2) in k_pairs:
        for (u_dict, v_dict, w_dict) in strassen_terms:
            U_term = {}
            for (li, lj), coeff in u_dict.items():
                U_term[(li, lj)] = coeff
            V_term = {}
            for (lj, lk), coeff in v_dict.items():
                V_term[(lj, k1 + lk)] = coeff
            W_term = {}
            for (li, lk), coeff in w_dict.items():
                W_term[(li, k1 + lk)] = coeff
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
            
    # 2x2x1 block: rank 1
    # T[i,j,k]=1 for i in {0,1}, j in {0,1}, k=8
    # u[i,j]=1, v[j,8]=1, w[8,i]=1
    k = k_single[0]
    U_term = {}
    for i in range(2):
        for j in range(2):
            U_term[(i, j)] = 1
    V_term = {}
    for j in range(2):
        V_term[(j, k)] = 1
    W_term = {}
    for i in range(2):
        W_term[(k, i)] = 1
    U_terms.append(U_term)
    V_terms.append(V_term)
    W_terms.append(W_term)
    
    # 2x1x9 block: rank 1
    # T[i,2,k]=1 for i in {0,1}, k in {0..8}
    # u[i,2]=1, v[2,k]=1, w[k,i]=1
    U_term = {}
    for i in range(2):
        U_term[(i, 2)] = 1
    V_term = {}
    for k in range(9):
        V_term[(2, k)] = 1
    W_term = {}
    for i in range(2):
        for k in range(9):
            W_term[(k, i)] = 1
    U_terms.append(U_term)
    V_terms.append(V_term)
    W_terms.append(W_term)
    
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), coeff in U_terms[col].items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_terms[col].items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in W_terms[col].items():
            W[k * n + i][col] = coeff
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    T_computed = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(n*m)])
        v_vec = np.array([V[i][c] for i in range(m*p)])
        w_vec = np.array([W[i][c] for i in range(p*n)])
        T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    return np.max(np.abs(T_computed - T_expected)) < 1e-10

def _rank1_decomposition(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    try:
        U, V, W = _construct_2x3x9_strassen()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            U, V, W = _rank1_decomposition(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _rank1_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
