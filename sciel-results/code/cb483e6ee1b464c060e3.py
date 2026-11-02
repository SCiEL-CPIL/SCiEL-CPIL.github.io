"""Bootstrap SEARCH for exact tensor decomposition of 3x4x7 matrix multiplication."""
import time
import itertools
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as sla

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Return the best exact tensor decomposition found within a time budget.
    The interface is construct() -> {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start = time.time()
    deadline = start + 140.0  # safety margin below 150s kill

    def _elapsed():
        return time.time() - start

    def _time_left(frac=0.7):
        return deadline - _elapsed() - 10.0  # 10s safety margin

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 84

    def _schoolbook():
        """Return the classical rank-nmp decomposition."""
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _strassen_embedded():
        """
        Construct a decomposition of rank 83 by replacing the 2x2x2 schoolbook block
        (i in {0,1}, j in {0,1}, k in {0,1}) with Strassen's rank-7 decomposition.
        """
        R = 83
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        # Strassen terms for 2x2x2 block
        # Indices: i, j in {0,1}, k in {0,1}
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00, A11
        U[0*m + 0][0] = 1
        U[1*m + 1][0] = 1
        # V: B00, B11
        V[0*p + 0][0] = 1
        V[1*p + 1][0] = 1
        # W: C00 += M1, C11 += M1
        W[0*n + 0][0] = 1
        W[1*n + 1][0] = 1
        
        # Term 2: M2 = (A10+A11)B00
        # U: A10, A11
        U[1*m + 0][1] = 1
        U[1*m + 1][1] = 1
        # V: B00
        V[0*p + 0][1] = 1
        # W: C10 += M2, C11 += M2
        W[0*n + 1][1] = 1
        W[1*n + 1][1] = 1
        
        # Term 3: M3 = A00(B10-B11)
        # U: A00
        U[0*m + 0][2] = 1
        # V: B10, B11
        V[1*p + 0][2] = 1
        V[1*p + 1][2] = -1
        # W: C01 += M3, C11 -= M3
        W[1*n + 0][2] = 1
        W[1*n + 1][2] = -1
        
        # Term 4: M4 = A11(B01-B00)
        # U: A11
        U[1*m + 1][3] = 1
        # V: B01, B00
        V[0*p + 1][3] = 1
        V[0*p + 0][3] = -1
        # W: C00 -= M4, C01 += M4
        W[0*n + 0][3] = -1
        W[1*n + 0][3] = 1
        
        # Term 5: M5 = (A00+A10)B11
        # U: A00, A10
        U[0*m + 0][4] = 1
        U[1*m + 0][4] = 1
        # V: B11
        V[1*p + 1][4] = 1
        # W: C00 += M5, C10 += M5
        W[0*n + 0][4] = 1
        W[0*n + 1][4] = 1
        
        # Term 6: M6 = (A01-A00)(B00+B10)
        # U: A01, A00
        U[0*m + 1][5] = 1
        U[0*m + 0][5] = -1
        # V: B00, B10
        V[0*p + 0][5] = 1
        V[1*p + 0][5] = 1
        # W: C11 -= M6
        W[1*n + 1][5] = -1
        
        # Term 7: M7 = (A10-A11)(B01+B11)
        # U: A10, A11
        U[1*m + 0][6] = 1
        U[1*m + 1][6] = -1
        # V: B01, B11
        V[0*p + 1][6] = 1
        V[1*p + 1][6] = 1
        # W: C00 += M7
        W[0*n + 0][6] = 1
        
        # Fill remaining with schoolbook terms for the complement
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip the 2x2x2 block handled by Strassen
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Verify col matches R
        if col != R:
            # Fallback if logic error
            return _schoolbook()
            
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W):
        """Verify that the decomposition exactly reconstructs the matrix multiplication."""
        N = len(U)
        M = len(V)
        P = len(W)
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rng = np.random.RandomState(42)
        for trial in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            C_approx = np.zeros((n, p))
            A_flat = A.reshape(-1)
            B_flat = B.reshape(-1)
            for r in range(R):
                ua = Ua[:, r]
                va = Va[:, r]
                wa = Wa[:, r]
                u_dot_a = ua @ A_flat
                v_dot_b = va @ B_flat
                C_approx += (wa * (u_dot_a * v_dot_b)).reshape(p, n).T
            if not np.allclose(C_true, C_approx, atol=1e-10):
                return False
        return True

    # Try the Strassen-embedded decomposition
    best = _strassen_embedded()
    if _verify(best["U"], best["V"], best["W"]):
        return best
        
    # Fallback to schoolbook if verification fails (should not happen)
    return _schoolbook()

# EVOLVE-BLOCK-END
