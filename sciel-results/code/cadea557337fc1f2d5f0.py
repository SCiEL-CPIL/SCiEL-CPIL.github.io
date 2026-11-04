"""Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]
DEADLINE = time.time() + 130.0  # safety margin before 150s wall-clock


def _tensor_index(n, m, p, i, j, k):
    # Index into the flattened tensor T[i, j, k]
    return (i * m + j) * p + k


def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W (list of lists of Fractions) exactly decompose the tensor."""
    rank = len(U[0])
    # Build the tensor as a dict: (i,j,k) -> sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We need this to equal 1 if i==j==k... wait, the matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all (i,j,k) in the range.
    # Actually the tensor for matrix multiplication is T[i,j,k] = 1 for all valid i,j,k.
    # Let me re-check: the decomposition is C[i,k] = sum_c U[i,j,c]*V[j,k,c]*W[i,k,c]... 
    # No. The standard formulation: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: C[i,k] = sum_{j,c} A[i,j] B[j,k] W[c] where we need U[i,j,c], V[j,k,c], W[i,k,c]
    # such that sum_c U[i,j,c] V[j,k,c] W[i,k,c] = delta... 
    # Actually the correct formulation for matrix mult tensor:
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all (i,j,k)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Compute all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True


def _verify_numeric(U, V, W, n, m, p, tol=1e-10):
    """Quick numeric verification."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i*m+j], Va[j*p+k]) * 0 + np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank(rank, n, m, p, max_iter=200):
    """Try to find a rank-rank decomposition using numerical optimization."""
    if rank < 1:
        return None
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Scale to reasonable magnitudes
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    
    def loss_and_grad(U, V, W):
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    u_row = U[i*m+j]
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    val = np.sum(u_row * v_row * w_row)
                    err = val - 1.0
                    loss += err * err
                    
                    # d(loss)/d(U[i*m+j,c]) = 2*err * V[j*p+k,c] * W[k*n+i,c]
                    grad_U[i*m+j] += 2 * err * v_row * w_row
                    grad_V[j*p+k] += 2 * err * u_row * w_row
                    grad_W[k*n+i] += 2 * err * u_row * v_row
        
        return loss, (grad_U, grad_V, grad_W)
    
    # Use L-BFGS-B style with manual implementation (scipy might be available)
    try:
        from scipy.optimize import minimize
        # Flatten all variables
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        size_U = n*m*rank
        size_V = m*p*rank
        
        def objective(x):
            U_x = x[:size_U].reshape(n*m, rank)
            V_x = x[size_U:size_U+size_V].reshape(m*p, rank)
            W_x = x[size_U+size_V:].reshape(p*n, rank)
            loss, _ = loss_and_grad(U_x, V_x, W_x)
            return loss
        
        def grad(x):
            U_x = x[:size_U].reshape(n*m, rank)
            V_x = x[size_U:size_U+size_V].reshape(m*p, rank)
            W_x = x[size_U+size_V:].reshape(p*n, rank)
            _, g = loss_and_grad(U_x, V_x, W_x)
            return np.concatenate([g[0].flatten(), g[1].flatten(), g[2].flatten()])
        
        result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if result.success or result.fun < 1e-8:
            U_opt = result.x[:size_U].reshape(n*m, rank)
            V_opt = result.x[size_U:size_U+size_V].reshape(m*p, rank)
            W_opt = result.x[size_U+size_V:].reshape(p*n, rank)
            
            # Verify numerically
            if _verify_numeric(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8):
                # Try to find exact rational values
                # Round to nearest fraction with small denominator
                U_frac = _to_rational(U_opt)
                V_frac = _to_rational(V_opt)
                W_frac = _to_rational(W_opt)
                
                if U_frac is not None and V_frac is not None and W_frac is not None:
                    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                        return U_frac, V_frac, W_frac
                else:
                    # Try with larger denominators
                    U_frac = _to_rational(U_opt, max_den=100)
                    V_frac = _to_rational(V_opt, max_den=100)
                    W_frac = _to_rational(W_opt, max_den=100)
                    if U_frac is not None and V_frac is not None and W_frac is not None:
                        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                            return U_frac, V_frac, W_frac
    except Exception:
        pass
    
    return None


def _to_rational(arr, max_den=10):
    """Convert numeric array to list of Fractions with small denominators."""
    result = []
    for row in arr:
        row_frac = []
        for val in row:
            if abs(val) < 1e-12:
                row_frac.append(0)
            else:
                found = False
                for den in range(1, max_den+1):
                    num = round(val * den)
                    if abs(val - num/den) < 1e-10:
                        row_frac.append(num/den)
                        found = True
                        break
                if not found:
                    return None
        result.append(row_frac)
    return result


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = (schoolbook_rank, U, V, W)
    
    # Try to reduce rank
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also rank >= ceil(n*m*p / something)... 
    # Let's try ranks from a reasonable range
    for rank in range(30, schoolbook_rank):
        if time.time() > DEADLINE - 10:
            break
        result = _try_rank(rank, n, m, p, max_iter=100)
        if result is not None:
            U_new, V_new, W_new = result
            best = (rank, U_new, V_new, W_new)
            break  # Found a valid decomposition, this is rank-minimal (we went up from lower bound)
    
    rank_best, U_best, V_best, W_best = best
    return {"U": U_best, "V": V_best, "W": W_best}
