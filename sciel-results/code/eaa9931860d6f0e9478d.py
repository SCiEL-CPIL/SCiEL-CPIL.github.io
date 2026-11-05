import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find rank-49 decomposition for 2x5x5 matrix multiplication.
    Uses Alternating Least Squares (ALS) with multiple restarts and rational rounding.
    """
    if time.time() > _DEADLINE - 15:
        return None

    target_rank = 49
    
    try:
        best_error = float('inf')
        best_params = None
        
        max_restarts = 20 # Reduced restarts to ensure time safety, ALS is fast
        
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 5:
                break
            
            np.random.seed(1000 + restart)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # ALS Iterations
            for it in range(50): # Reduced iterations per restart to allow more restarts if needed
                if time.time() > _DEADLINE - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build A: (p, r)
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except np.linalg.LinAlgError:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        # System: sum_c V_jk,c * (U_ij,c * W_ki,c) = 1 for all i
                        # Let B_jk be matrix of size (n, r) where B_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        # Solve B_jk @ V_jk = ones(n)
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            y, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V[j * p + k, :] = y
                        except np.linalg.LinAlgError:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        # System: sum_c W_ki,c * (U_ij,c * V_jk,c) = 1 for all j
                        # Let C_ki be matrix of size (m, r) where C_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        # Solve C_ki @ W_ki = ones(m)
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            z, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = z
                        except np.linalg.LinAlgError:
                            pass
                
                # Check convergence / error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            error += (s - 1.0)**2
                
                if error < 1e-12:
                    break

            if error < best_error:
                best_error = error
                best_params = (U.copy(), V.copy(), W.copy())
            
            if best_error < 1e-10:
                break

        if best_error < 1e-6 and best_params is not None:
            U, V, W = best_params
            
            # Try rounding to rationals (integers or half-integers)
            def try_round(U, V, W, denom):
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                
                # Convert to list of lists of floats for verification
                # The verify function expects exact arithmetic. 
                # If we use floats, 1.0 might not be exact if denom is not power of 2? 
                # 1/2, 1/4, 1/8 are exact in binary.
                # But the verify function uses `!=` on sums of products.
                # If we pass floats, 0.5 * 0.5 * 0.5 = 0.125. Sum might be 1.0 exactly?
                # It's safer to convert to Fractions or check with high precision.
                # However, the task says "integer/half-integer entries".
                # So we only need to check denom=1 and denom=2.
                
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return U_r, V_r, W_r
                return None

            res = try_round(U, V, W, 1)
            if res:
                U_r, V_r, W_r = res
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

            res = try_round(U, V, W, 2)
            if res:
                U_r, V_r, W_r = res
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # If ALS converged to a very low error, maybe the solution is rational 
            # but not aligned to the grid? 
            # Try denominators 4 and 8 as well, just in case, though task says int/half-int.
            # The task says "integer/half-integer entries". So 1 and 2 are the only valid denominators.
            # If it doesn't round to 1 or 2, it's not a valid solution for this specific constraint.
            # However, sometimes numerical drift makes it look like 0.5 but it's 0.49999.
            # np.round handles this.
            
            # One more check: maybe the solution has integer entries but with different scaling?
            # No, the tensor entries are 1. The decomposition is unique up to scaling of columns.
            # If we scale column c by alpha, U_c *= alpha, V_c /= alpha, W_c /= alpha.
            # This doesn't change the product.
            # So if a rational solution exists, we can scale it to have integer or half-integer entries?
            # Not necessarily. If entries are 1/3, we can't make them integers or half-integers by scaling all three matrices simultaneously with a single scalar?
            # Wait, the scaling is per column.
            # If we have a solution with rational entries, we can clear denominators for each column.
            # But the constraint is that ALL entries must be integer or half-integer.
            # This is a strong constraint.
            
            # If best_error is very small but rounding fails, we might try to refine the solution
            # by solving the linear system exactly using the rounded values as initial guess?
            # No, if rounding fails, the exact solution is not on the grid.
            
            # Let's try one more thing: check if the solution is close to integers/half-integers
            # but with a global scale factor?
            # No, the scale is per column.
            
            # If we can't find a valid integer/half-integer solution, we return None.
            # The caller will fall back to schoolbook.
            
            # One potential improvement: if the error is small, but rounding to 1/2 fails,
            # maybe the true solution is 1/4? But the task forbids it.
            # So we must stick to 1 and 2.
            
            # If we fail, we return None.
            
        return None

    except Exception:
        return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find a better (lower rank) solution
    # Rank 50 is lower bound? No, rank must be at least max(n*m, m*p, p*n) = 25.
    # Strassen-like algorithms might achieve less than n*m*p = 50.
    # For 2x5x5, is rank < 50 possible?
    # The lower bound is max(n*m, m*p, p*n) = 25.
    # Known results for small matrices:
    # 2x2x2: rank 7
    # 2x3x3: rank ?
    # 2x5x5: likely close to 50, but maybe 49 or 48?
    
    # We will try rank 49 first. If that fails, we can try 48, etc.
    # But 49 is the most likely candidate for improvement.
    
    if time.time() < _DEADLINE - 20:
        res = _try_rank_49(n, m, p)
        if res is not None:
            return res
            
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
