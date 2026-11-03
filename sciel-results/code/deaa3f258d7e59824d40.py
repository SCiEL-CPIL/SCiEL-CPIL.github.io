"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def round_to_half(x):
        return round(x * 2) / 2.0

    def try_find_decomposition(r, max_iter=100):
        # Initialize randomly
        np.random.seed(42) # For reproducibility in search
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        for _ in range(max_iter):
            # Solve for V given U, W
            A = np.zeros((n * m * p, m * p * r))
            b = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u = (j * p + k) * r + c
                            coeff = U[i * m + j, c] * W[k * n + i, c]
                            A[e, u] = coeff
            
            v_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V = v_flat.reshape(m * p, r)
            
            # Solve for U given V, W
            A = np.zeros((n * m * p, n * m * r))
            b = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u = (i * m + j) * r + c
                            coeff = V[j * p + k, c] * W[k * n + i, c]
                            A[e, u] = coeff
            
            u_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U = u_flat.reshape(n * m, r)
            
            # Solve for W given U, V
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u = (k * n + i) * r + c
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            A[e, u] = coeff
            
            w_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W = w_flat.reshape(p * n, r)
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        error += (val - 1.0) ** 2
            
            if error < 1e-9:
                # Round to half-integers
                U_rounded = [[round_to_half(val) for val in row] for row in U]
                V_rounded = [[round_to_half(val) for val in row] for row in V]
                W_rounded = [[round_to_half(val) for val in row] for row in W]
                
                # Verify rounded solution
                error_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_rounded[i * m + j][c] * V_rounded[j * p + k][c] * W_rounded[k * n + i][c]
                            error_rounded += (val - 1.0) ** 2
                            
                if error_rounded < 1e-9:
                    return U_rounded, V_rounded, W_rounded
                else:
                    # If rounding breaks it, return floats? The task asks for int/half-int.
                    # If we can't get exact half-ints, we might fail.
                    # But for small ranks, solutions are often rational.
                    pass
        return None

    # Try ranks from 24 to 36
    for r in range(24, 37):
        res = try_find_decomposition(r)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
