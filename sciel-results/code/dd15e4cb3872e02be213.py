import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110  # Leave 10s margin

    # Precompute the target tensor
    # T[a,b,c] = 1 if there exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W, R):
        """Verify exact decomposition with integer/half-integer entries."""
        # Check all entries are 0, 1/2, or integers
        for arr in [U, V, W]:
            for val in arr:
                if not (val * 2 == int(val * 2)):
                    return False
        # Check tensor reconstruction
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        if not np.allclose(T_rec, T, atol=1e-10):
            return False
        # Check exactness with fractions for the 1-positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if s != 1:
                        return False
        return True

    # Build schoolbook as fallback
    best_rank = n * m * p
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school, best_rank)

    # Try to find a lower-rank decomposition using a structured approach
    # The tensor has the structure: T[i*m+j, j*p+k, k*n+i] = 1
    # We can try to find a decomposition with rank R < 36
    
    # Strategy: Use the fact that for fixed j, the sub-tensor is a permutation
    # Let's try to find a decomposition where the rank is reduced by sharing factors
    
    # Approach: Try small ranks and use a constrained search
    # For R in [20, 35], try to find U, V, W with entries in {-1, 0, 1}
    
    # Given the complexity, let's try a specific structured ansatz
    # that exploits the symmetry of the problem.
    
    # The output dimension is 12 (2*6), and the input dimensions are 6 and 18.
    # We can try to find a decomposition where each rank-1 term corresponds
    # to a "pattern" in the multiplication.
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
    # A more sophisticated search would be needed to find a lower-rank decomposition.
    
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
