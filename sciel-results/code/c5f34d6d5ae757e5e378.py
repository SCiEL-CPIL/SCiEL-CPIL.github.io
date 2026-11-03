import numpy as np
import time
from scipy.optimize import least_squares
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook decomposition (Rank 24)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # Try to find Rank 12
    target_rank = max(n * m, m * p, n * p)  # 12

    def residual(params):
        U_flat, V_flat, W_flat = params[:n*m*target_rank], params[n*m*target_rank:n*m*target_rank+m*p*target_rank], params[n*m*target_rank+m*p*target_rank:]
        U_mat = U_flat.reshape(n*m, target_rank)
        V_mat = V_flat.reshape(m*p, target_rank)
        W_mat = W_flat.reshape(p*n, target_rank)
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = 0
                        for r in range(target_rank):
                            val += W_mat[k*n+i, r] * U_mat[i*m+j, r] * V_mat[jp*p+k, r]
                        target = 1.0 if j == jp else 0.0
                        res.append(val - target)
        return np.array(res)

    def rational_reconstruct(x, max_den=8):
        """Reconstruct a float as a fraction with small denominator"""
        # Try to find a/b close to x where b <= max_den
        best_frac = Fraction(0, 1)
        best_err = float('inf')
        for b in range(1, max_den + 1):
            a = round(x * b)
            err = abs(x - a / b)
            if err < best_err:
                best_err = err
                best_frac = Fraction(a, b)
        # Only accept if very close
        if best_err < 1e-6:
            return float(best_frac)
        # Try with larger denominators
        for b in range(max_den + 1, 33):
            a = round(x * b)
            err = abs(x - a / b)
            if err < 1e-8:
                return a / b
        return x

    def clean_matrix(M):
        """Clean a matrix to rational values"""
        M_clean = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_clean[i, j] = rational_reconstruct(M[i, j])
        return M_clean

    def verify_decomposition(U, V, W):
        """Verify that U, V, W give exact matrix multiplication"""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    total = 0.0
                    for r in range(rank):
                        total += W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
                    target = 1.0 if j == j else 0.0  # This is always 1.0 for diagonal
                    # Actually we need: sum_r W[k*n+i,r] * U[i*m+j,r] * V[j'*p+k,r] = delta_{j,j'}
                    # For each (i,k), the result is a m x m matrix that should be identity
        # Better verification:
        for i in range(n):
            for k in range(p):
                # Compute the m x m result matrix
                for j in range(m):
                    for jp in range(m):
                        total = 0.0
                        for r in range(U.shape[1]):
                            total += W[k*n+i, r] * U[i*m+j, r] * V[jp*p+k, r]
                        target = 1.0 if j == jp else 0.0
                        if abs(total - target) > 1e-8:
                            return False
        return True

    def snap_and_verify(U, V, W, step):
        U_s = np.round(U / step) * step
        V_s = np.round(V / step) * step
        W_s = np.round(W / step) * step
        if verify_decomposition(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    found_rank_12 = False
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(attempt * 7919 + 13)
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(residual, params0, method='lm', max_nfev=8000)
            
            if result.cost < 1e-6:
                U_mat = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_mat = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                W_mat = result.x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                
                # Try rational reconstruction
                U_r = clean_matrix(U_mat)
                V_r = clean_matrix(V_mat)
                W_r = clean_matrix(W_mat)
                
                if verify_decomposition(U_r, V_r, W_r):
                    best_result = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    best_rank = target_rank
                    found_rank_12 = True
                    break
                
                # Try snapping to half-integers (step 0.5)
                snapped = snap_and_verify(U_mat, V_mat, W_mat, 0.5)
                if snapped:
                    U_s, V_s, W_s = snapped
                    best_result = {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
                    best_rank = target_rank
                    found_rank_12 = True
                    break
                else:
                    # Try snapping to integers (step 1.0)
                    snapped = snap_and_verify(U_mat, V_mat, W_mat, 1.0)
                    if snapped:
                        U_s, V_s, W_s = snapped
                        best_result = {
                            "U": U_s.tolist(),
                            "V": V_s.tolist(),
                            "W": W_s.tolist()
                        }
                        best_rank = target_rank
                        found_rank_12 = True
                        break
        except Exception:
            continue

    return best_result
