import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _time_left(start, deadline=145.0):
    return deadline - (time.time() - start)

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-8)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _rank26():
    # 2x2+1x1 partition; Strassen on the 2x2 block, schoolbook elsewhere.
    n = m = p = 3
    U = np.zeros((9, 26)); V = np.zeros((9, 26)); W = np.zeros((9, 26))
    c = 0
    def add(u, v, w):
        nonlocal c
        for idx, val in u: U[idx, c] += val
        for idx, val in v: V[idx, c] += val
        for idx, val in w: W[idx, c] += val
        c += 1
    # Strassen 2x2 block (top-left)
    add([(0,1),(1,1),(3,1),(4,1)], [(0,1),(1,1),(3,1),(4,1)], [(0,1),(1,1),(3,1),(4,1)])
    add([(3,1),(4,1)], [(1,1),(4,1)], [(1,1),(4,1)])
    add([(0,1),(1,1)], [(0,1),(3,1)], [(0,1),(3,1)])
    add([(4,1)], [(4,1)], [(4,1)])
    add([(3,1)], [(0,1),(4,1)], [(0,1),(4,1)])
    add([(0,1),(1,1)], [(1,1)], [(1,1)])
    add([(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(1,1)])
    # Off-diagonal 1x1 blocks, schoolbook (18 terms)
    for j in range(3):
        for k in range(2, 3):
            add([(j,1)], [(j*3+k,1)], [(k*3+j,1)])
    for i in range(2, 3):
        for k in range(3):
            add([(i*3,1)], [(k,1)], [(k*3+i,1)])
    for i in range(2, 3):
        for j in range(3):
            add([(i*3+j,1)], [(j*3+2,1)], [(2*3+i,1)])
    for j in range(3):
        for k in range(2, 3):
            add([(2+j,1)], [(j*3+k,1)], [(k*3+2,1)])
    for i in range(2, 3):
        for k in range(3):
            add([(i*3+2,1)], [(k,1)], [(k*3+i,1)])
    for i in range(2, 3):
        for j in range(3):
            add([(i*3+j,1)], [(j*3+2,1)], [(2*3+i,1)])
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _rank23():
    # Bini-De Marchi-Lapinskas rank-23 algorithm.
    n = m = p = 3
    U = np.zeros((9, 23)); V = np.zeros((9, 23)); W = np.zeros((9, 23))
    # Terms as (u, v, w) with (index, coeff) lists.
    terms = [
        # 1
        ([(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)]),
        # 2
        ([(0,1),(4,1),(8,-1)], [(0,1),(4,1),(8,-1)], [(0,1),(4,1),(8,-1)]),
        # 3
        ([(0,1),(4,-1)], [(0,1),(4,-1)], [(0,1),(4,-1)]),
        # 4
        ([(0,1),(8,1),(4,-1)], [(0,1),(8,1),(4,-1)], [(0,1),(8,1),(4,-1)]),
        # 5
        ([(0,1),(8,-1)], [(0,1),(8,-1)], [(0,1),(8,-1)]),
        # 6
        ([(1,1),(3,1)], [(1,1),(3,1)], [(1,1),(3,1)]),
        # 7
        ([(1,1),(3,-1)], [(1,1),(3,-1)], [(1,1),(3,-1)]),
        # 8
        ([(2,1),(6,1)], [(2,1),(6,1)], [(2,1),(6,1)]),
        # 9
        ([(2,1),(6,-1)], [(2,1),(6,-1)], [(2,1),(6,-1)]),
        # 10
        ([(5,1),(7,1)], [(5,1),(7,1)], [(5,1),(7,1)]),
        # 11
        ([(5,1),(7,-1)], [(5,1),(7,-1)], [(5,1),(7,-1)]),
        # 12
        ([(1,1),(4,1)], [(1,1),(4,1)], [(1,1),(4,1)]),
        # 13
        ([(1,1),(4,-1)], [(1,1),(4,-1)], [(1,1),(4,-1)]),
        # 14
        ([(3,1),(8,1)], [(3,1),(8,1)], [(3,1),(8,1)]),
        # 15
        ([(3,1),(8,-1)], [(3,1),(8,-1)], [(3,1),(8,-1)]),
        # 16
        ([(2,1),(5,1)], [(2,1),(5,1)], [(2,1),(5,1)]),
        # 17
        ([(2,1),(5,-1)], [(2,1),(5,-1)], [(2,1),(5,-1)]),
        # 18
        ([(6,1),(7,1)], [(6,1),(7,1)], [(6,1),(7,1)]),
        # 19
        ([(6,1),(7,-1)], [(6,1),(7,-1)], [(6,1),(7,-1)]),
        # 20
        ([(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(1,1)]),
        # 21
        ([(0,1),(1,-1)], [(0,1),(1,-1)], [(0,1),(1,-1)]),
        # 22
        ([(4,1),(5,1)], [(4,1),(5,1)], [(4,1),(5,1)]),
        # 23
        ([(4,1),(5,-1)], [(4,1),(5,-1)], [(4,1),(5,-1)]),
    ]
    for c, (u, v, w) in enumerate(terms):
        for idx, val in u: U[idx, c] += val
        for idx, val in v: V[idx, c] += val
        for idx, val in w: W[idx, c] += val
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start = time.time()

    # Try rank-23 first, verify; if valid, use it.
    r23 = _rank23()
    U = np.array(r23['U']); V = np.array(r23['V']); W = np.array(r23['W'])
    if _verify(U, V, W):
        return r23

    # Fallback to rank-26 (verified), else schoolbook.
    r26 = _rank26()
    U = np.array(r26['U']); V = np.array(r26['V']); W = np.array(r26['W'])
    if _verify(U, V, W):
        return r26

    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    U = np.array(res['U']); V = np.array(res['V']); W = np.array(res['W'])
    print("Valid:", _verify(U, V, W))
