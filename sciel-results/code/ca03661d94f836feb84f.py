import numpy as np
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 84
    
    # The matrix multiplication tensor T has components T[i, j, k] where:
    # i in [0, n), j in [0, m), k in [0, p)
    # The decomposition requires:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    # This is equivalent to finding a rank-r decomposition of the tensor.
    # We'll try to find a low-rank decomposition by solving a system of equations.
    
    # Strategy: 
    # 1. Try ranks from a reasonable minimum up to schoolbook
    # 2. For each rank, use random initialization and iterative solving
    
    # Known lower bound: For matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # Actually, the tensor rank is at least the "slice rank" which is related to these products.
    # A better lower bound is the "border rank" but we'll use a simpler heuristic.
    
    # Let's try to find a decomposition with rank close to the theoretical minimum.
    # For 3x4x7, let's try ranks from 30 to 84.
    
    def check_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, n, m, p, num_trials=100):
        """Try to find a rank-r decomposition."""
        for trial in range(num_trials):
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            # Solve for W using least squares
            W = np.zeros((p * n, r))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Build the linear system for W[k*n+i, :]
                    # For each j, we have: sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    b = np.ones(m)
                    
                    try:
                        w, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                        if residuals and residuals[0] > 1e-10:
                            valid = False
                            break
                        W[k * n + i, :] = w
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Check if the solution is exact
                if check_decomposition(U, V, W, n, m, p, tol=1e-8):
                    # Try to round to integers or half-integers
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    if check_decomposition(U_round, V_round, W_round, n, m, p, tol=1e-9):
                        return U_round, V_round, W_round
                    
                    # If rounding doesn't work, return the float solution
                    # But the task requires integer/half-integer entries
                    # So we need to find a solution that rounds correctly
                    # Let's try to solve with constraints that force integer/half-integer solutions
                    # For now, if we have a valid float solution, we can try to find nearby integer solutions
                    
                    # Try rounding to nearest integer
                    U_int = np.round(U).astype(int)
                    V_int = np.round(V).astype(int)
                    W_int = np.round(W).astype(int)
                    
                    if check_decomposition(U_int.astype(float), V_int.astype(float), W_int.astype(float), n, m, p, tol=1e-9):
                        return U_int.astype(float), V_int.astype(float), W_int.astype(float)
        
        return None
    
    # Try ranks from a reasonable minimum
    # Theoretical lower bound for tensor rank of matrix multiplication is at least 
    # max(n*m, m*p, p*n) / something... actually it's more complex.
    # Let's try from 30 up to 84
    
    for r in range(30, rank_school + 1):
        result = try_rank(r, n, m, p, num_trials=50)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    rank = rank_school
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
