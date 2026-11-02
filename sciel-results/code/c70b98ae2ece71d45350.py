"""Rank-minimal exact tensor decomposition for 2x2 * 2x16 matrix multiply."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

_START = time.time()

def _build_uvp(n, m, p):
    # 8 independent 2x2x2 blocks -> rank 7 per block -> total 56
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen 7 products for a 2x2 block
    # A order in U rows: i*m + j -> 0:A00,1:A01,2:A10,3:A11
    U_factors = np.array([
        [1, 0, 0, 1],   # M1 = (A00+A11) * (...)
        [0, 0, 1, 1],   # M2 = (A10+A11) * (...)
        [1, 0, 0, 0],   # M3 = A00 * (...)
        [0, 0, 0, 1],   # M4 = A11 * (...)
        [1, 1, 0, 0],   # M5 = (A00+A01) * (...)
        [-1, 0, 1, 0],  # M6 = (A10-A00) * (...)
        [0, 1, 0, -1],  # M7 = (A01-A11) * (...)
    ])
    # B within a 2x2 block: [B00, B01, B10, B11]
    B_factors = np.array([
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 1, -1],  # M3: B10-B11
        [-1, 1, 0, 0],  # M4: B01-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ])
    # C within a 2x2 block: [C00, C01, C10, C11]
    C_factors = np.array([
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1],  # M2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0],   # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],  # M5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],   # M6: C11 += 1
        [1, 0, 0, 0],   # M7: C00 += 1
    ])

    num_blocks = p // 2  # 8
    for j in range(num_blocks):
        col_start = 2 * j
        for k in range(7):
            c = j * 7 + k
            # U
            U[0, c] = U_factors[k, 0]
            U[1, c] = U_factors[k, 1]
            U[2, c] = U_factors[k, 2]
            U[3, c] = U_factors[k, 3]
            # V: B_00 -> V[0*p+col_start], B_01 -> V[0*p+col_start+1]
            #    B_10 -> V[1*p+col_start], B_11 -> V[1*p+col_start+1]
            V[0 * p + col_start, c] += B_factors[k, 0]
            V[0 * p + col_start + 1, c] += B_factors[k, 1]
            V[1 * p + col_start, c] += B_factors[k, 2]
            V[1 * p + col_start + 1, c] += B_factors[k, 3]
            # W: C_00 -> W[col_start*n+0], C_01 -> W[(col_start+1)*n+0]
            #    C_10 -> W[col_start*n+1], C_11 -> W[(col_start+1)*n+1]
            W[col_start * n + 0, c] += C_factors[k, 0]
            W[(col_start + 1) * n + 0, c] += C_factors[k, 1]
            W[col_start * n + 1, c] += C_factors[k, 2]
            W[(col_start + 1) * n + 1, c] += C_factors[k, 3]
    return U, V, W


def _verify(n, m, p, U, V, W):
    # Build the multiplication tensor T[i,j,k] = 1 iff via j, i.e. T[i,j,k] = 1
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for the correct (i,j,k)
    # Actually the multiplication tensor M[i,j,k] = 1 for all (i,j,k) in range
    # (since C[i,k] = sum_j A[i,j] B[j,k])
    U_t = U.T  # (rank, n*m)
    V_t = V.T  # (rank, m*p)
    W_t = W.T  # (rank, p*n)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j, :]
                vj = V[j * p + k, :]
                wk = W[k * n + i, :]
                val = np.dot(np.dot(ui, vj), wk)
                expected = 1.0
                if abs(val - expected) > 1e-9:
                    raise AssertionError(f"Mismatch at ({i},{j},{k}): got {val}, expected {expected}")


def construct():
    deadline = _START + 120
    n, m, p = DIMENSIONS

    def now():
        return time.time()

    # Build the rank-56 decomposition
    U, V, W = _build_uvp(n, m, p)

    # Verify correctness (fast, exact with floats on small integer data)
    _verify(n, m, p, U, V, W)

    # Convert to Python lists of numbers (integers where possible)
    def to_list(mat):
        out = []
        for row in mat:
            r = []
            for x in row:
                xf = float(x)
                # snap near-integers
                xi = int(round(xf))
                if abs(xf - xi) < 1e-12:
                    r.append(xi)
                else:
                    r.append(xf)
            out.append(r)
        return out

    result = {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    return result
