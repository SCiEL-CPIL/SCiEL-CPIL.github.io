import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    try:
        U, V, W = _strassen_4x4(n, m, p)
        # Verify correctness with random test
        if time.time() < deadline - 20:
            _verify(U, V, W, n, m, p, deadline)
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass

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

def _verify(U, V, W, n, m, p, deadline):
    """Verify the decomposition with a random test."""
    rng = np.random.RandomState(42)
    A = rng.rand(n, m)
    B = rng.rand(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    C_computed = np.zeros((n, p))
    
    for r in range(U_arr.shape[1]):
        if time.time() > deadline - 5:
            break
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        
        uA = u_col * A
        vB = v_col * B
        
        term = uA @ vB
        
        w_mat = W_arr[:, r].reshape(p, n)
        C_computed += term * w_mat.T
        
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        raise ValueError("Decomposition verification failed")

def _strassen_4x4(n, m, p):
    """
    Direct combinatorial construction of rank-49 Strassen decomposition for 4x4.
    Uses Kronecker product structure: 49 = 7 (outer) * 7 (inner)
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Inner Strassen 2x2 patterns
    # Each: u_terms [(r,c,sign)], v_terms, w_terms
    inner_defs = [
        # t=0: M0 = (X00+X11)(Y00+Y11) -> Z00+=1, Z11+=1
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        # t=1: M1 = (X10+X11)Y00 -> Z10+=1, Z11-=1
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1),(1,1,-1)]},
        # t=2: M2 = X00(Y01-Y11) -> Z01+=1, Z11+=1
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(0,1,1),(1,1,1)]},
        # t=3: M3 = X11(Y10-Y00) -> Z00+=1, Z10+=1
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,0,1)]},
        # t=4: M4 = (X00+X01)Y11 -> Z01+=1, Z00-=1
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1),(0,0,-1)]},
        # t=5: M5 = (X10-X00)(Y00+Y01) -> Z11+=1
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        # t=6: M6 = (X01-X11)(Y10+Y11) -> Z00+=1
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1)]},
    ]

    # Outer Strassen for 4x4 (2x2 blocks of 2x2)
    outer_defs = [
        # s=0: M0 = (A00+A11)(B00+B11) -> C00+=M0, C11+=M0
        {'a': [(0,0,1),(1,1,1)], 'b': [(0,0,1),(1,1,1)], 'c': [(0,0,1),(1,1,1)]},
        # s=1: M1 = (A10+A11)B00 -> C10+=M1, C11-=M1
        {'a': [(1,0,1),(1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1),(1,1,-1)]},
        # s=2: M2 = A00(B01-B11) -> C01+=M2, C11+=M2
        {'a': [(0,0,1)], 'b': [(0,1,1),(1,1,-1)], 'c': [(0,1,1),(1,1,1)]},
        # s=3: M3 = A11(B10-B00) -> C00+=M3, C10+=M3
        {'a': [(1,1,1)], 'b': [(1,0,1),(0,0,-1)], 'c': [(0,0,1),(1,0,1)]},
        # s=4: M4 = (A00+A01)B11 -> C01+=M4, C00-=M4
        {'a': [(0,0,1),(0,1,1)], 'b': [(1,1,1)], 'c': [(0,1,1),(0,0,-1)]},
        # s=5: M5 = (A10-A00)(B00+B01) -> C11+=M5
        {'a': [(1,0,1),(0,0,-1)], 'b': [(0,0,1),(0,1,1)], 'c': [(1,1,1)]},
        # s=6: M6 = (A01-A11)(B10+B11) -> C00+=M6
        {'a': [(0,1,1),(1,1,-1)], 'b': [(1,0,1),(1,1,1)], 'c': [(0,0,1)]},
    ]

    # Pre-compute global indices for each (s,t) pair
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            outer = outer_defs[s]
            inner = inner_defs[t]

            # U: A element at global (row, col) where row = 2*P+ir, col = 2*Q+jc
            for (P, Q, sign_a) in outer['a']:
                for (ir, jc, sign_x) in inner['u']:
                    coeff = sign_a * sign_x
                    u_idx = (2 * P + ir) * 4 + (2 * Q + jc)
                    U[u_idx][r] += coeff

            # V: B element at global (row, col) where row = 2*P+ir, col = 2*Q+jc
            for (P, Q, sign_b) in outer['b']:
                for (ir, jc, sign_y) in inner['v']:
                    coeff = sign_b * sign_y
                    v_idx = (2 * P + ir) * 4 + (2 * Q + jc)
                    V[v_idx][r] += coeff

            # W: C element at global (row, col) where row = 2*P+ir, col = 2*Q+jc
            # W index follows trace convention: W[k*n+i, r] = col*4 + row
            for (P, Q, sign_c) in outer['c']:
                for (ir, jc, sign_z) in inner['w']:
                    coeff = sign_c * sign_z
                    w_idx = (2 * Q + jc) * 4 + (2 * P + ir)
                    W[w_idx][r] += coeff

    return U, V, W
