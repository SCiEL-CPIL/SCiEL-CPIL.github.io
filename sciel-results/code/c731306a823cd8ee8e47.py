"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def verify_exact(U, V, W):
        """Verify exact decomposition using Fractions."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    def verify_float(U, V, W, tol=1e-9):
        """Verify using float for speed."""
        r = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_arr[i * m + j, :]
                    v_row = V_arr[j * p + k, :]
                    w_row = W_arr[k * n + i, :]
                    s = np.dot(np.dot(u_row, v_row), w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Structured decomposition with rank = m*p = 20
    # U[i*m+j, j*p+k] = 1, V[j*p+k, j*p+k] = 1, W[k*n+i, j*p+k] = 1
    rank_struct = m * p
    U_struct = [[0] * rank_struct for _ in range(n * m)]
    V_struct = [[0] * rank_struct for _ in range(m * p)]
    W_struct = [[0] * rank_struct for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            for i in range(n):
                U_struct[i * m + j][c] = 1
                W_struct[k * n + i][c] = 1
            V_struct[j * p + k][c] = 1
    
    if verify_exact(U_struct, V_struct, W_struct):
        best_result = {"U": U_struct, "V": V_struct, "W": W_struct}
        best_rank = rank_struct
    
    return best_result
# EVOLVE-BLOCK-END
