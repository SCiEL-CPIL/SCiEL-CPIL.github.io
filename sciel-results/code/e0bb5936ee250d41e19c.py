import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 120.0
_START = time.time()


def _time_left():
    return _DEADLINE_S - (time.time() - _START)


def _verify(U, V, W, num_tests=5):
    n, m, p = DIMENSIONS
    R = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for _ in range(num_tests):
        A = np.random.randint(-10, 11, (3, 3)).astype(np.float64)
        B = np.random.randint(-10, 11, (3, 3)).astype(np.float64)
        C_expected = A @ B
        A_flat = A.flatten()
        B_flat = B.flatten()
        U_proj = U_arr.T @ A_flat
        V_proj = V_arr.T @ B_flat
        C_computed = np.zeros((3, 3), dtype=np.float64)
        for r in range(R):
            p_r = U_proj[r] * V_proj[r]
            if abs(p_r) > 1e-15:
                w_col = W_arr[:, r]
                w_matrix = w_col.reshape(3, 3)
                C_computed += p_r * w_matrix.T
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _rank26_construct():
    """
    Rank-26 decomposition: Strassen on 2x2 block + scalar 1x1 + cross terms.
    A indices: 0=a00,1=a01,2=a02,3=a10,4=a11,5=a12,6=a20,7=a21,8=a22
    B indices: 0=b00,1=b01,2=b02,3=b10,4=b11,5=b12,6=b20,7=b21,8=b22
    W indices: W[k*3+i] where C[i,k]
    """
    terms = []

    def add(u, v, w):
        terms.append((u, v, w))

    # --- Strassen 2x2x2 on top-left block ---
    # M1 = (A00+A11)(B00+B11)
    add({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    # M2 = A10*(B00+B11)
    add({3: 1}, {0: 1, 4: 1}, {3: 1, 4: 1})
    # M3 = A00*(B11-B00)
    add({0: 1}, {4: 1, 0: -1}, {1: 1})
    # M4 = A11*(B10-B00)
    add({4: 1}, {3: 1, 0: -1}, {3: 1})
    # M5 = (A00+A01)*B10
    add({0: 1, 1: 1}, {3: 1}, {0: 1, 4: 1})
    # M6 = (A10-A00)*(B00+B01)
    add({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1})
    # M7 = (A01-A11)*(B10+B11)
    add({1: 1, 4: -1}, {3: 1, 4: 1}, {1: 1})

    # --- 1x1 block ---
    # a22*b22 -> c22
    add({8: 1}, {8: 1}, {8: 1})

    # --- Cross terms: row 2 of A * cols of B (a20,a21 with b*) ---
    # a20*b00 -> c20
    add({6: 1}, {0: 1}, {2: 1})
    # a20*b01 -> c21
    add({6: 1}, {1: 1}, {5: 1})
    # a20*b02 -> c22
    add({6: 1}, {2: 1}, {8: 1})
    # a21*b00 -> c20
    add({7: 1}, {0: 1}, {2: 1})
    # a21*b01 -> c21
    add({7: 1}, {1: 1}, {5: 1})
    # a21*b02 -> c22
    add({7: 1}, {2: 1}, {8: 1})
    # a20*b10 -> c20
    add({6: 1}, {3: 1}, {2: 1})
    # a20*b11 -> c21
    add({6: 1}, {4: 1}, {5: 1})
    # a20*b12 -> c22
    add({6: 1}, {5: 1}, {8: 1})
    # a21*b10 -> c20
    add({7: 1}, {3: 1}, {2: 1})
    # a21*b11 -> c21
    add({7: 1}, {4: 1}, {5: 1})
    # a21*b12 -> c22
    add({7: 1}, {5: 1}, {8: 1})

    # --- Cross terms: cols 1,2 of A * row 2 of B (a*,b20,b21) ---
    # a02*b20 -> c02
    add({2: 1}, {6: 1}, {6: 1})
    # a02*b21 -> c02
    add({2: 1}, {7: 1}, {6: 1})
    # a12*b20 -> c12
    add({5: 1}, {6: 1}, {7: 1})
    # a12*b21 -> c12
    add({5: 1}, {7: 1}, {7: 1})

    # --- Cross terms: rows 0,1 of A * col 2 of B (a*,b*2) already handled above? ---
    # Actually need: a00*b02, a01*b12, a10*b02, a11*b12 for c02,c12
    # Wait, let me recount. The Strassen handles c00,c01,c10,c11.
    # We need c02, c12, c20, c21, c22.
    # c22 = a20*b02 + a21*b12 + a22*b22 (a22*b22 done above)
    #   a20*b02 and a21*b12 done above.
    # c20 = a20*b00 + a21*b10 + a22*b20
    #   a20*b00, a21*b10 done above. Need a22*b20.
    # c21 = a20*b01 + a21*b11 + a22*b21
    #   a20*b01, a21*b11 done above. Need a22*b21.
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22

    # a22*b20 -> c20
    add({8: 1}, {6: 1}, {2: 1})
    # a22*b21 -> c21
    add({8: 1}, {7: 1}, {5: 1})

    # a00*b02 -> c02
    add({0: 1}, {2: 1}, {6: 1})
    # a01*b12 -> c02
    add({1: 1}, {5: 1}, {6: 1})
    # a02*b22 -> c02
    add({2: 1}, {8: 1}, {6: 1})

    # a10*b02 -> c12
    add({3: 1}, {2: 1}, {7: 1})
    # a11*b12 -> c12
    add({4: 1}, {5: 1}, {7: 1})
    # a12*b22 -> c12
    add({5: 1}, {8: 1}, {7: 1})

    rank = len(terms)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]

    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    best = _schoolbook()
    best_rank = 27

    if _time_left() > 5.0:
        cand26 = _rank26_construct()
        if _verify(cand26["U"], cand26["V"], cand26["W"], num_tests=5):
            best = cand26
            best_rank = 26

    return best
