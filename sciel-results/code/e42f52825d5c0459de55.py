import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def _target_tensor():
    n, m, p = 2, 2, 3
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _error(params, T, rank):
    n, m, p = 2, 2, 3
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.sum((T_calc - T)**2)
    return err

def _try_rank10_search(T, max_time=100):
    n, m, p = 2, 2, 3
    rank = 10
    target_size = n*m*rank + m*p*rank + p*n*rank
    start_time = time.time()
    
    # Try a few random restarts
    for restart in range(20):
        if time.time() - start_time > max_time:
            break
            
        # Initialize with small random values
        np.random.seed(42 + restart)
        params = np.random.randn(target_size) * 0.1
        
        # Also try initializing from rank 11 solution projected? 
        # That's complex. Random is simpler to start.
        
        res = minimize(_error, params, args=(T, rank), method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < 1e-8:
            # Found a potential solution
            U = res.x[:n*m*rank].reshape(n*m, rank)
            V = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Check if it's exact
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_calc, T, atol=1e-6):
                # Try to snap to integers or half-integers
                # Scale by 2 to check for integers
                U2 = U * 2
                V2 = V * 2
                W2 = W * 2
                
                # Check if close to integers
                if (np.allclose(U2, np.round(U2), atol=1e-4) and
                    np.allclose(V2, np.round(V2), atol=1e-4) and
                    np.allclose(W2, np.round(W2), atol=1e-4)):
                    U_int = np.round(U2).astype(int) / 2.0
                    V_int = np.round(V2).astype(int) / 2.0
                    W_int = np.round(W2).astype(int) / 2.0
                    
                    # Verify integer solution
                    T_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                    if np.allclose(T_int, T, atol=1e-9):
                        return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
                # If not integer, return floats (might be accepted if exact enough, but contract says integer/half-integer)
                # Let's try to return the float version if it's very accurate, but the contract is strict.
                # We will return None if we can't snap, to be safe.
                pass
        
        # If L-BFGS-B fails, try a different initialization or method?
        # For now, just random restarts.

    return None

def _rank11_solution():
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[2][2] = 1; W[2][4] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return U, V, W

def construct():
    _t0 = time.time()
    _deadline = 110.0  # Leave 10s margin for 120s limit
    
    T = _target_tensor()
    
    # Try to find rank 10
    if time.time() - _t0 < _deadline:
        result = _try_rank10_search(T, max_time=_deadline - (time.time() - _t0))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 11
    U, V, W = _rank11_solution()
    return {"U": U, "V": V, "W": W}
