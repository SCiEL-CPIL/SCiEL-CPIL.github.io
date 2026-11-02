import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i
    
    def add_strassen_block(k0, col_offset):
        prods = [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ]
        for t, (a_coeff, b_coeff) in enumerate(prods):
            col = col_offset + t
            for (i, j), c in a_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in b_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
        c_combos = [
            (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
            (0, k0+1,   [(2,1),(4,1)]),
            (1, k0,     [(1,1),(3,1)]),
            (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col_offset + t] += coeff
    
    def add_single_block(k, col_offset):
        prods = [
            ((0,0),(0,k)),
            ((0,1),(1,k)),
            ((1,0),(0,k)),
            ((1,1),(1,k)),
        ]
        for t, ((i,j),(jj,kk)) in enumerate(prods):
            col = col_offset + t
            U[A_idx(i, j), col] += 1
            V[B_idx(jj, kk), col] += 1
        W[C_idx(0, k), col_offset+0] += 1
        W[C_idx(0, k), col_offset+1] += 1
        W[C_idx(1, k), col_offset+2] += 1
        W[C_idx(1, k), col_offset+3] += 1
    
    # 6 Strassen blocks for k0 = 0,2,4,6,8,10
    col = 0
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
    # 1 single block for k=12
    add_single_block(12, col)
    col += 4
    assert col == R, f"col={col}, R={R}"
    
    # Check deadline
    if time.time() > deadline - 10:
        # Return the solution without full verification
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the decomposition using tensor reconstruction
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Check target entries are 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[A_idx(i,j), B_idx(j,k), C_idx(i,k)] - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    # Check non-target entries are 0
    if valid:
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    a, b = divmod(i, m)
                    b2, k2 = divmod(j, p)
                    k3, a2 = divmod(k, n)
                    if a == a2 and b == b2 and k2 == k3:
                        continue
                    if abs(T[i, j, k]) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
    
    if not valid:
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        for idx in range(R0):
            i, j = divmod(idx, m * p)
            k = idx % p
            U0[i * m + j, idx] = 1.0
            V0[j * p + k, idx] = 1.0
            W0[k * n + i, idx] = 1.0
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
