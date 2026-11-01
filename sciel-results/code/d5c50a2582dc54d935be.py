import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 7]
DEADLINE = 140  # seconds, leave 10s margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return time.time() - start_time < DEADLINE
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if not np.isclose(val, 1.0, atol=1e-8):
                        return False
        # Check zero entries
        for I in range(n*m):
            for J in range(m*p):
                for K in range(p*n):
                    # Decode indices
                    i1 = I // m
                    j1 = I % m
                    j2 = J // p
                    k2 = J % p
                    k3 = K // n
                    i3 = K % n
                    # This is a valid multiplication term if j1 == j2 and i1 == i3
                    if j1 == j2 and i1 == i3:
                        continue
                    val = sum(U[I][r] * V[J][r] * W[K][r] for r in range(rank))
                    if not np.isclose(val, 0.0, atol=1e-8):
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the standard rank-n*m*p decomposition."""
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
    
    def try_kronecker_approach(R_vec):
        """
        Try to find a low-rank decomposition by exploiting the block structure.
        We treat the multiplication as a bilinear map over polynomials in x.
        
        The idea: represent the 3-dimensional intermediate space using polynomial
        multiplication. If we can find a rank-R decomposition of the 2x3 by 3x1
        multiplication, we can extend it to 2x3 by 3x7 by considering each of the
        7 output columns independently.
        
        Actually, let's try a more direct approach: search for a rank-R decomposition
        of the full tensor directly, but use a smarter parameterization.
        """
        if not check_deadline():
            return None
        
        # The tensor has n*m = 6 "input" indices for U, m*p = 21 for V, p*n = 14 for W
        # Total equations: 6*21*14 = 1764, but only 2*3*7 = 42 are non-zero
        
        # Let's try to find a rank-R decomposition directly
        # Variables: U (6 x R), V (21 x R), W (14 x R)
        # Total variables: 41*R
        
        def residuals(x):
            U = x[:6*R].reshape(6, R)
            V = x[6*R:6*R+21*R].reshape(21, R)
            W = x[6*R+21*R:].reshape(14, R)
            
            res = []
            # Non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        res.append(val - 1)
            
            # Zero entries
            for I in range(n*m):
                for J in range(m*p):
                    for K in range(p*n):
                        i1, j1 = divmod(I, m)
                        j2, k2 = divmod(J, p)
                        k3, i3 = divmod(K, n)
                        if j1 == j2 and i1 == i3:
                            continue
                        val = np.sum(U[I] * V[J] * W[K])
                        res.append(val)
            
            return np.array(res)
        
        # Try multiple random initializations
        best_x = None
        best_err = np.inf
        
        num_tries = 20
        for _ in range(num_tries):
            if not check_deadline():
                break
            x0 = np.random.randn(41*R) * 0.3
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=500)
                err = sol.cost
                if err < best_err:
                    best_err = err
                    best_x = sol.x
            except Exception:
                pass
        
        if best_err < 1e-8:
            U = best_x[:6*R].reshape(6, R)
            V = best_x[6*R:6*R+21*R].reshape(21, R)
            W = best_x[6*R+21*R:].reshape(14, R)
            
            # Try to snap to integers or half-integers
            for scale in [2, 1]:
                U_snapped = np.round(U * scale) / scale
                V_snapped = np.round(V * scale) / scale
                W_snapped = np.round(W * scale) / scale
                
                if verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()):
                    return {
                        "U": U_snapped.tolist(),
                        "V": V_snapped.tolist(),
                        "W": W_snapped.tolist()
                    }
            
            # If snapping fails, try returning the float solution
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None
    
    # Try ranks from 42 downward, but be strategic about it
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But for matrix multiplication, the rank is typically higher
    
    # Try a few promising ranks
    for R in [42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 21]:
        if not check_deadline():
            break
        
        result = try_kronecker_approach(R)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook_decomposition()
