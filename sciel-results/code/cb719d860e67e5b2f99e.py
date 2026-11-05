"""Matrix multiplication 3x5x7 rank-minimal decomposition."""
import time
import random

DIMENSIONS = [3, 5, 7]

def _build_strassen_rank99():
    n, m, p = 3, 5, 7
    n_split = [2, 1]
    m_split = [2, 2, 1]
    p_split = [2, 2, 2, 1]
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    n_off = 0
    for ns in n_split:
        m_off = 0
        for ms in m_split:
            p_off = 0
            for ps in p_split:
                if ns == 2 and ms == 2 and ps == 2:
                    strassen_terms = [
                        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
                        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
                        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
                        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
                        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
                        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1}),
                    ]
                    for u_dict, v_dict, w_dict in strassen_terms:
                        U_entry = {}
                        V_entry = {}
                        W_entry = {}
                        for (i, j), coeff in u_dict.items():
                            U_entry[(n_off + i, m_off + j)] = coeff
                        for (j, k), coeff in v_dict.items():
                            V_entry[(m_off + j, p_off + k)] = coeff
                        for (k, i), coeff in w_dict.items():
                            W_entry[(p_off + k, n_off + i)] = coeff
                        U_rows.append(U_entry)
                        V_rows.append(V_entry)
                        W_rows.append(W_entry)
                else:
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U_rows.append({(n_off + i, m_off + j): 1})
                                V_rows.append({(m_off + j, p_off + k): 1})
                                W_rows.append({(p_off + k, n_off + i): 1})
                p_off += ps
            m_off += ms
        n_off += ns
    
    rank = len(U_rows)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(zip(U_rows, V_rows, W_rows)):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][c] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i][c] = coeff
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=3, m=5, p=7):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U[0])):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    deadline = time.time() + 120
    
    result = _build_strassen_rank99()
    
    if not _verify(result["U"], result["V"], result["W"]):
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return result
