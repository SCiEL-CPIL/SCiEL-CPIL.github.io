import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Partition: columns [0,1], [2,3], [4,5] use Strassen (rank 7 each), column [6] uses schoolbook (rank 4).
    Total rank: 7 + 7 + 7 + 4 = 25.
    """
    n, m, p = 2, 2, 7
    rank = 25

    if _time_left() < 5.0:
        # Emergency schoolbook fallback
        return _schoolbook()

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_strassen(col_start, k0, k1):
        """
        Add Strassen rank-7 terms for 2x2 * 2x2 block.
        A is 2x2, B columns are k0, k1, C columns are k0, k1.
        U index: i*m + k for A[i,k]
        V index: k*p + j for B[k,j]
        W index: j*n + i for C[i,j] (row i, col j)
        """
        # M1 = (A00+A11) * (B0k0 + B1k1)
        # C0k0 += M1, C1k1 += M1
        U[0, col_start] += 1   # A00
        U[3, col_start] += 1   # A11
        V[0*p + k0, col_start] += 1   # B0k0
        V[1*p + k1, col_start] += 1   # B1k1
        W[k0*n + 0, col_start] += 1   # C[0,k0]
        W[k1*n + 1, col_start] += 1   # C[1,k1]

        # M2 = (A10+A11) * B0k0
        # C1k0 += M2, C1k1 -= M2
        U[2, col_start+1] += 1   # A10
        U[3, col_start+1] += 1   # A11
        V[0*p + k0, col_start+1] += 1   # B0k0
        W[k0*n + 1, col_start+1] += 1   # C[1,k0]
        W[k1*n + 1, col_start+1] -= 1   # C[1,k1]

        # M3 = A00 * (B0k1 - B1k1)
        # C0k1 += M3, C1k1 += M3
        U[0, col_start+2] += 1   # A00
        V[0*p + k1, col_start+2] += 1   # B0k1
        V[1*p + k1, col_start+2] -= 1   # -B1k1
        W[k1*n + 0, col_start+2] += 1   # C[0,k1]
        W[k1*n + 1, col_start+2] += 1   # C[1,k1]

        # M4 = A11 * (B1k0 - B0k0)
        # C0k0 += M4, C1k0 += M4
        U[3, col_start+3] += 1   # A11
        V[1*p + k0, col_start+3] += 1   # B1k0
        V[0*p + k0, col_start+3] -= 1   # -B0k0
        W[k0*n + 0, col_start+3] += 1   # C[0,k0]
        W[k0*n + 1, col_start+3] += 1   # C[1,k0]

        # M5 = (A00+A01) * B1k1
        # C0k0 -= M5, C0k1 += M5
        U[0, col_start+4] += 1   # A00
        U[1, col_start+4] += 1   # A01
        V[1*p + k1, col_start+4] += 1   # B1k1
        W[k0*n + 0, col_start+4] -= 1   # C[0,k0]
        W[k1*n + 0, col_start+4] += 1   # C[0,k1]

        # M6 = (A10-A00) * (B0k0 + B0k1)
        # C1k1 += M6
        U[2, col_start+5] += 1   # A10
        U[0, col_start+5] -= 1   # -A00
        V[0*p + k0, col_start+5] += 1   # B0k0
        V[0*p + k1, col_start+5] += 1   # B0k1
        W[k1*n + 1, col_start+5] += 1   # C[1,k1]

        # M7 = (A01-A11) * (B1k0 + B1k1)
        # C0k0 += M7
        U[1, col_start+6] += 1   # A01
        U[3, col_start+6] -= 1   # -A11
        V[1*p + k0, col_start+6] += 1   # B1k0
        V[1*p + k1, col_start+6] += 1   # B1k1
        W[k0*n + 0, col_start+6] += 1   # C[0,k0]

    # Strassen for columns (0,1)
    add_strassen(0, 0, 1)
    # Strassen for columns (2,3)
    add_strassen(7, 2, 3)
    # Strassen for columns (4,5)
    add_strassen(14, 4, 5)

    # Schoolbook for column 6 (4 terms)
    cs = 21
    # A00 * B0,6 -> C[0,6]
    U[0, cs] += 1
    V[0*p + 6, cs] += 1
    W[6*n + 0, cs] += 1
    # A01 * B1,6 -> C[0,6]
    U[1, cs+1] += 1
    V[1*p + 6, cs+1] += 1
    W[6*n + 0, cs+1] += 1
    # A10 * B0,6 -> C[1,6]
    U[2, cs+2] += 1
    V[0*p + 6, cs+2] += 1
    W[6*n + 1, cs+2] += 1
    # A11 * B1,6 -> C[1,6]
    U[3, cs+3] += 1
    V[1*p + 6, cs+3] += 1
    W[6*n + 1, cs+3] += 1

    # Verify with random test
    if _time_left() > 10.0:
        rng = np.random.default_rng(42)
        A_test = rng.standard_normal((n, m))
        B_test = rng.standard_normal((m, p))
        C_expected = A_test @ B_test

        C_W = np.zeros(p * n)
        for r in range(rank):
            SA = np.dot(U[:, r], A_test.flatten())
            SB = np.dot(V[:, r], B_test.flatten())
            C_W += W[:, r] * (SA * SB)

        C_mat = np.zeros((n, p))
        for j in range(p):
            for i in range(n):
                C_mat[i, j] = C_W[j * n + i]

        if not np.allclose(C_mat, C_expected, atol=1e-10):
            return _schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook():
    """Fallback: rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, idx] = 1
                V[k * p + j, idx] = 1
                W[j * n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print(f"Rank: {U.shape[1]}")
    
    rng = np.random.default_rng(123)
    A = rng.standard_normal((2, 2))
    B = rng.standard_normal((2, 7))
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        C_W += W[:, r] * (SA * SB)
    
    C_mat = np.zeros((2, 7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k + i]
    
    if np.allclose(C_mat, C_expected, atol=1e-10):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print("C_mat:")
        print(C_mat)
        print("C_expected:")
        print(C_expected)
        print("Max error:", np.max(np.abs(C_mat - C_expected)))
