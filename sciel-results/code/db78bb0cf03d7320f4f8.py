"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        """Return the trivial (schoolbook) decomposition with rank = n*m*p = 54."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def compute_residual(U, V, W):
        """Compute the residual for all (i,j,k) triples."""
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i, j, k] = np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i])
        return residual
    
    def try_rank(rank, max_restarts=5):
        """Try to find a valid decomposition of given rank using scipy least_squares."""
        if time.time() > deadline - 15:
            return None
        
        target = np.ones((n, m, p))
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart * 1000)
            
            # Initialize with random values
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                U = x[:n*m*rank].reshape(n*m, rank)
                V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                return U, V, W
            
            def objective(x):
                U, V, W = unflatten(x)
                res = compute_residual(U, V, W)
                return (res - target).flatten()
            
            x0 = flatten(U, V, W)
            
            try:
                result = least_squares(objective, x0, method='lm', max_nfev=5000)
                U_opt, V_opt, W_opt = unflatten(result.x)
                err = np.sum(result.fun**2)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    
                if err < 1e-10:
                    break
            except Exception:
                continue
        
        if best_err < 1e-6 and best_U is not None:
            # Try to round to integers or half-integers
            for scale in [1, 2, 4]:
                U_round = np.round(best_U * scale) / scale
                V_round = np.round(best_V * scale) / scale
                W_round = np.round(best_W * scale) / scale
                
                res = compute_residual(U_round, V_round, W_round)
                if np.allclose(res, target, atol=1e-6):
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
            
            # Try integer rounding
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            res = compute_residual(U_int, V_int, W_int)
            if np.allclose(res, target, atol=1e-6):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None
    
    # Try to find a low-rank solution using structured optimization
    best_result = None
    
    # Try ranks in increasing order from theoretical lower bound
    for rank in [23, 24, 25, 26, 27, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]:
        if time.time() > deadline - 30:
            break
        result = try_rank(rank, max_restarts=3)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
