import time
import numpy as np

# --- deadline guard (evaluator kills at 150 s wall-clock) ---
_START = time.time()
_DEADLINE = _START + 140.0

def _time_left():
    return _DEADLINE - time.time()


def _strassen_2x2():
    """Rank-7 Strassen decomposition of 2x2x2 matrix multiplication.

    Row index convention (matches the evaluator's official convention):
      A: U[2*i + k]  ->  A[i, k]
      B: V[2*k + j]  ->  B[k, j]
      C: W[2*j + i]  ->  C[j, i]   (trace convention W[k*n+i, r])
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # Term 2: M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # Term 3: M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # Term 4: M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # Term 5: M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # Term 6: M6 = (A10-A00)(B00+B01) -> C10 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1

    # Term 7: M7 = (A01-A11)(B10+B11) -> C01 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1

    return U, V, W


def _verify(U, V, W, n, m, p, trials=6):
    """Return max |tensor(C) - U V^T W^T| over random integer trials."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # tensor[r, i, k] = sum_s U[i,k,s] * V[k,j,s] * W[j,i,s]
    T = np.einsum('iks,kjs,jis->rij', U, V, W)
    worst = 0.0
    for _ in range(trials):
        A = np.random.randint(-2, 3, size=(n, m)).astype(float)
        B = np.random.randint(-2, 3, size=(m, p)).astype(float)
        C = A @ B
        # flatten: a[2*i+k], b[2*k+j], c[2*j+i]
        af = np.zeros(4)
        bf = np.zeros(4)
        for i in range(n):
            for k in range(m):
                af[2 * i + k] = A[i, k]
        for k in range(m):
            for j in range(p):
                bf[2 * k + j] = B[k, j]
        cf = np.zeros(4)
        for j in range(p):
            for i in range(n):
                cf[2 * j + i] = C[j, i]
        pred = (af * V) @ W  # (af*V)[j,s]=sum_k B[k,j]V[k,j,s]; then @W gives sum_s (..)*W[j,i,s]
        worst = max(worst, float(np.max(np.abs(pred - cf))))
    return worst


def _rank49_decomposition():
    """Kronecker product of the rank-7 Strassen factor with itself."""
    U2, V2, W2 = _strassen_2x2()

    U2_t = U2.reshape(2, 2, 7)  # [i, k, s]
    V2_t = V2.reshape(2, 2, 7)  # [k, j, s]
    W2_t = W2.reshape(2, 2, 7)  # [j, i, s]

    # tensor product: index (i,k,j) -> (i1,i2,k1,k2,j1,j2), (s1,s2)
    U4 = np.einsum('iks,lpt->ilktsp', U2_t, U2_t).reshape(4, 4, 49)
    V4 = np.einsum('kjt,lur->kljtrt', V2_t, V2_t).reshape(4, 4, 49)
    W4 = np.einsum('jit,klv->jiktvl', W2_t, W2_t).reshape(4, 4, 49)

    U = U4.reshape(16, 49)
    V = V4.reshape(16, 49)
    W = W4.reshape(16, 49)
    return U, V, W


def _to_integer(U, V, W):
    """Convert to integer entries if every value is integral (within tolerance)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    ok = (
        np.allclose(U, np.round(U), atol=1e-9)
        and np.allclose(V, np.round(V), atol=1e-9)
        and np.allclose(W, np.round(W), atol=1e-9)
    )
    if ok:
        U = np.round(U).astype(int)
        V = np.round(V).astype(int)
        W = np.round(W).astype(int)
        return U.tolist(), V.tolist(), W.tolist()
    # fall back to floats (still exact for integer inputs)
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = 4, 4, 4
    U, V, W = _rank49_decomposition()

    # Verify the decomposition is exact (cheap: a handful of random integer trials).
    if _time_left() > 5.0:
        err = _verify(U, V, W, n, m, p, trials=8)
        if not np.isfinite(err) or err > 1e-6:
            # fallback: schoolbook rank-64 (always exact)
            Us = np.zeros((16, 64))
            Vs = np.zeros((16, 64))
            Ws = np.zeros((16, 64))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        r = (i * m + k) * p + j
                        Us[2 * i + k, r] = 1
                        Vs[2 * k + j, r] = 1
                        Ws[2 * j + i, r] = 1
            U, V, W = Us, Vs, Ws

    U_list, V_list, W_list = _to_integer(U, V, W)
    return {"U": U_list, "V": V_list, "W": W_list}
