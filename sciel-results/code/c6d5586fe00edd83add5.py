"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # The tensor constraint: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # U: (n*m) x rank = 9 x rank
    # V: (m*p) x rank = 18 x rank
    # W: (p*n) x rank = 18 x rank
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly satisfies the tensor equation."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_round_and_verify(U, V, W, tol=1e-9):
        """Try rounding to integers or half-integers and verify."""
        # Try integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int, tol):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        # Try half-integer rounding
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        if verify(U_half, V_half, W_half, tol):
            return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
        return None
    
    # Fallback: schoolbook decomposition (rank = n*m*p = 54)
    # This is guaranteed to be valid and uses integer coefficients.
    # We skip optimization to avoid timeouts and floating-point issues.
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

# EVOLVE-BLOCK-END
