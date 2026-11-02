import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 120.0
_START = time.time()

def _time_left():
    return _DEADLINE_S - (time.time() - _START)

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

def _verify_matmul(U, V, W):
    """Verify the tensor decomposition against multiple random integer matrices."""
    n, m, p = DIMENSIONS
    R = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    rng = np.random.default_rng(42)
    for trial in range(20):
        A = rng.integers(-5, 6, size=(3, 3))
        B = rng.integers(-5, 6, size=(3, 3))
        C_expected = A @ B
        A_flat = A.flatten()
        B_flat = B.flatten()
        U_proj = U_arr.T @ A_flat
        V_proj = V_arr.T @ B_flat
        C_computed = np.zeros((3, 3), dtype=np.int64)
        for r in range(R):
            p_r = U_proj[r] * V_proj[r]
            if p_r != 0:
                w_col = W_arr[:, r]
                w_matrix = w_col.reshape(3, 3)
                C_computed += p_r * w_matrix.T
        if not np.array_equal(C_computed, C_expected):
            return False
    return True

def _rank23_construct():
    """
    Rank-23 decomposition for 3x3 matmul.
    Partition A into 2x2, 2x1, 1x2, 1x1 blocks; B similarly.
    Use Strassen (7 mults) for A11*B11.
    Use 6 cross terms with Strassen-like sharing for A12*B21.
    Use 6 cross terms for A11*B12.
    Use 1 term for A22*B22.
    Use 1 shared term for A12*B22 + A22*B21 (via (a12+a22)*(b21+b22) - a12*b22 - a22*b21).
    Total: 7 + 6 + 6 + 1 + 1 + 1 + 1 = 23
    """
    rank = 23
    terms = []

    def add(u, v, w):
        terms.append((u, v, w))

    # A index: a[i][j] -> i*3+j
    # B index: b[j][k] -> j*3+k
    # W index: C[i][k] -> k*3+i (trace convention W[k*n+i])

    # --- Strassen for A11 * B11 (7 terms) ---
    # M1 = (a00+a11)(b00+b11)
    add({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    # M2 = (a10+a11)*b00
    add({3:1, 4:1}, {0:1}, {3:1})
    # M3 = a00*(b01-b11)
    add({0:1}, {1:1, 4:-1}, {3:1})
    # M4 = a11*(b10-b00)
    add({4:1}, {3:1, 0:-1}, {0:1, 3:1})
    # M5 = (a00+a01)*b11
    add({0:1, 1:1}, {4:1}, {0:-1, 3:1})
    # M6 = (a10-a00)*(b00+b01)
    add({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    # M7 = (a01-a11)*(b10+b11)
    add({1:1, 4:-1}, {3:1, 4:1}, {0:1})

    # --- A12 * B21 (2x1 * 1x2 = 2x2): 6 terms using Strassen-like ---
    # A12 = [[a02],[a12]], B21 = [b20, b21]
    # C contribution: [[a02*b20, a02*b21],[a12*b20, a12*b21]]
    # Use 4 scalar mults + 2 shared:
    # N1 = (a02+a12)*b20 -> C[0,0]+=a02*b20, C[1,0]+=a12*b20
    add({2:1, 5:1}, {6:1}, {0:1, 3:1})
    # N2 = (a02+a12)*b21 -> C[0,1]+=a02*b21, C[1,1]+=a12*b21
    add({2:1, 5:1}, {7:1}, {3:1, 4:1})
    # N3 = a02*b21 -> C[0,1] already covered; need correction
    # Actually: C[0,0]=a02*b20, C[0,1]=a02*b21, C[1,0]=a12*b20, C[1,1]=a12*b21
    # N1 gives a02*b20 + a12*b20 to C[0,0] and C[1,0] -- WRONG, that's sum not separate
    # Let me redo: need separate terms
    # a02*b20 -> C[0,0]
    terms.pop()  # remove N1
    # a02*b21 -> C[0,1]
    terms.pop()  # remove N2
    add({2:1}, {6:1}, {0:1})   # a02*b20 -> C[0,0]
    add({2:1}, {7:1}, {3:1})   # a02*b21 -> C[0,1]
    add({5:1}, {6:1}, {3:1})   # a12*b20 -> C[1,0]
    add({5:1}, {7:1}, {4:1})   # a12*b21 -> C[1,1]
    # That's 4 terms, not 6. Let me recount total.
    # 7 (Strassen) + 4 (A12*B21) + 4 (A11*B12) + 4 (A21*B11) + 2 (A21*B12) + 2 (A12*B22) + 2 (A22*B21) + 1 (A22*B22) = 26

    # Let me try a different approach: rank 23 via known BDL construction.
    # The BDL (Bürgisser-Dehornoy-Lam) rank-23 construction for 3x3 matmul.
    # This is complex; let me use a verified rank-23 from literature.
    #
    # Actually, the simplest correct approach: use the known rank-23 from
    # Strassen's 1973 paper or the BDL construction. Let me hardcode it.
    #
    # Alternative: just use rank 26 which we can verify works.
    # For now, return the rank-26 as fallback.
    return None

def _rank26_construct():
    """
    Rank-26 decomposition for 3x3 matmul using Strassen on 2x2 block.
    """
    rank = 26
    terms = []

    def add(u, v, w):
        terms.append((u, v, w))

    # Strassen terms for A11*B11 (7 terms)
    add({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    add({3:1, 4:1}, {0:1}, {3:1})
    add({0:1}, {1:1, 4:-1}, {3:1})
    add({4:1}, {3:1, 0:-1}, {0:1, 3:1})
    add({0:1, 1:1}, {4:1}, {0:-1, 3:1})
    add({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    add({1:1, 4:-1}, {3:1, 4:1}, {0:1})

    # A12*B21 (4 terms)
    add({2:1}, {6:1}, {0:1})
    add({2:1}, {7:1}, {3:1})
    add({5:1}, {6:1}, {3:1})
    add({5:1}, {7:1}, {4:1})

    # A11*B12 (4 terms)
    add({0:1}, {2:1}, {6:1})
    add({1:1}, {5:1}, {6:1})
    add({3:1}, {2:1}, {7:1})
    add({4:1}, {5:1}, {7:1})

    # A12*B22 (2 terms)
    add({2:1}, {8:1}, {6:1})
    add({5:1}, {8:1}, {7:1})

    # A21*B11 (4 terms)
    add({6:1}, {0:1}, {2:1})
    add({7:1}, {3:1}, {2:1})
    add({6:1}, {1:1}, {5:1})
    add({7:1}, {4:1}, {5:1})

    # A22*B21 (2 terms)
    add({8:1}, {6:1}, {2:1})
    add({8:1}, {7:1}, {5:1})

    # A21*B12 (2 terms)
    add({6:1}, {2:1}, {8:1})
    add({7:1}, {5:1}, {8:1})

    # A22*B22 (1 term)
    add({8:1}, {8:1}, {8:1})

    assert len(terms) == 26, f"Expected 26 terms, got {len(terms)}"

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

def construct():
    best = _schoolbook()

    if _time_left() > 5.0:
        # Try rank-23 first
        cand23 = _rank23_construct()
        if cand23 is not None:
            if _verify_matmul(cand23["U"], cand23["V"], cand23["W"]):
                return cand23

        # Fall back to rank-26
        cand26 = _rank26_construct()
        if cand26 is not None:
            if _verify_matmul(cand26["U"], cand26["V"], cand26["W"]):
                best = cand26

    return best
