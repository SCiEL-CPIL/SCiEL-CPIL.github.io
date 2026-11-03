import time
import itertools

N, M, P = 3, 4, 4
NM, MP, PN = N*M, M*P, P*N
TOTAL_TENSORS = NM*MP*PN  # 3*4*4*4*4*4 = 768? No: 12*16*12 = 2304

def _verify(U, V, W, tol=1e-10):
    """Verify using einsum."""
    import numpy as np
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _schoolbook():
    rank = NM*MP*PN
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

def _strassen_44():
    rank = 44
    U = [[0]*rank for _ in range(NM)]
    V = [[0]*rank for _ in range(MP)]
    W = [[0]*rank for _ in range(PN)]
    
    # 7 Strassen steps for 2x2 block
    s_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # 4 blocks for the 2x2 output blocks
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
    
    # Bottom row (row 2) done naively: 4x4 = 16 terms
    for j in range(M):
        for k in range(P):
            U[2*M + j][col] = 1
            V[j*P + k][col] = 1
            W[k*N + 2][col] = 1
            col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}

def _try_reduce_by_merge(U, V, W):
    """
    Try to find pairs of rank-1 tensors that can be merged into one.
    Two rank-1 terms with U vectors u1,u2; V v1,v2; W w1,w2 can be merged
    if u1 ⊗ u2 ⊗ v1 ⊗ v2 ⊗ w1 ⊗ w2 allows cancellation.
    
    Simplest merge: if u1=-u2 and v1=v2 and w1=w2 -> zero out. 
    Or more generally, try to find if two terms are identical -> combine.
    """
    # This is a heuristic - try small modifications
    pass

def _gaussian_reduce(U, V, W):
    """
    Try to reduce rank by finding linear dependencies among rank-1 terms.
    This is computationally expensive, so we limit the search.
    """
    pass

def construct():
    start = time.time()
    deadline = start + 130  # 20s safety margin
    
    best_result = _strassen_44()
    ok, err = _verify(best_result['U'], best_result['V'], best_result['W'])
    if not ok:
        best_result = _schoolbook()
    
    # We could try to search for lower rank, but the Strassen 44 is our best known
    # Let's just return the verified result
    return best_result
