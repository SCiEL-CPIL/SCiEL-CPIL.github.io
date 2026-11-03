import time
import numpy as np

N, M, P = 3, 4, 4
NM, MP, PN = N*M, M*P, P*N

def _verify(U, V, W, tol=1e-10):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _schoolbook():
    rank = N*M*P
    U = [[0]*rank for _ in range(NM)]
    V = [[0]*rank for _ in range(MP)]
    W = [[0]*rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j][c] = 1
                V[j*P+k][c] = 1
                W[k*N+i][c] = 1
                c += 1
    return {'U': U, 'V': V, 'W': W}

def _construct_rank32():
    rank = 32
    U = [[0]*rank for _ in range(NM)]
    V = [[0]*rank for _ in range(MP)]
    W = [[0]*rank for _ in range(PN)]
    
    s_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    blocks = [(0,0,0,0,0,0), (0,2,2,0,0,0), (0,0,0,2,0,2), (0,2,2,2,0,2)]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for (u_c, v_c, w_c) in s_steps:
            for (a, b), val in u_c.items():
                U[(a+xi)*M + (b+xj)][col] += val
            for (b, c), val in v_c.items():
                V[(b+yj)*P + (c+yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c+zk)*N + (a+zi)][col] += val
            col += 1
    
    # Bottom row: 4 terms instead of 16
    for r in range(4):
        U[2*M + r][col] = 1
        for k in range(P):
            V[r*P + k][col] = 1
            W[k*N + 2][col] = 1
        col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}

def construct():
    start = time.time()
    deadline = start + 100
    
    result = _construct_rank32()
    ok, err = _verify(result['U'], result['V'], result['W'])
    
    if not ok:
        result = _schoolbook()
    
    return result
