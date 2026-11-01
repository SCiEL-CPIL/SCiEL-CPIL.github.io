"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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


def _verify_exact(U, V, W, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check compatible triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > tol:
                    return False
    
    # Check incompatible triples
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_c == i_a:
                    continue
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s) > tol:
                    return False
    return True


def _compute_residual(U, V, W):
    """Compute the residual norm of the decomposition."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    residual = 0.0
    # Compatible triples should have dot product = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                residual += (s - 1.0) ** 2
    
    # Incompatible triples should have dot product = 0
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_c == i_a:
                    continue
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                residual += s ** 2
    
    return np.sqrt(residual)


def _try_rank_r(r, max_iters=500, lr=0.01, n_restarts=5):
    """Try to find a rank-r decomposition using gradient descent."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    best_residual = float('inf')
    best_solution = None
    
    for restart in range(n_restarts):
        if time.time() > _START_TIME + _TIME_LIMIT - 10:
            break
        
        # Random initialization
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # Gradient descent
        for it in range(max_iters):
            if time.time() > _START_TIME + _TIME_LIMIT - 5:
                break
            
            U_arr = U
            V_arr = V
            W_arr = W
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # Compatible triples
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                        err = s - 1.0
                        # dU[a] += 2*err * V[b] * W[c]
                        dU[a] += 2 * err * V_arr[b] * W_arr[c]
                        # dV[b] += 2*err * U[a] * W[c]
                        dV[b] += 2 * err * U_arr[a] * W_arr[c]
                        # dW[c] += 2*err * U[a] * V[b]
                        dW[c] += 2 * err * U_arr[a] * V_arr[b]
            
            # Incompatible triples
            for a in range(nm):
                i_a = a // m
                j_a = a % m
                for b in range(mp):
                    j_b = b // p
                    k_b = b % p
                    for c in range(pn):
                        k_c = c // n
                        i_c = c % n
                        if j_a == j_b and k_b == k_c and i_c == i_a:
                            continue
                        s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                        err = s
                        dU[a] += 2 * err * V_arr[b] * W_arr[c]
                        dV[b] += 2 * err * U_arr[a] * W_arr[c]
                        dW[c] += 2 * err * U_arr[a] * V_arr[b]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check residual
            if it % 50 == 0:
                res = _compute_residual(U, V, W)
                if res < best_residual:
                    best_residual = res
                    best_solution = (U.copy(), V.copy(), W.copy())
                if res < 1e-8:
                    return U, V, W, res
        
        # Final check for this restart
        res = _compute_residual(U, V, W)
        if res < best_residual:
            best_residual = res
            best_solution = (U.copy(), V.copy(), W.copy())
    
    if best_solution is not None:
        return best_solution[0], best_solution[1], best_solution[2], best_residual
    return None, None, None, float('inf')


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Verify schoolbook is correct (it should be)
    if not _verify_exact(best["U"], best["V"], best["W"]):
        return best

    # Try to find lower rank decompositions
    # Theoretical lower bound is at least max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But known results suggest rank is around 30-35 for 2x4x5
    
    for r in range(20, best_rank):
        if time.time() > _START_TIME + _TIME_LIMIT - 30:
            break
        
        U, V, W, residual = _try_rank_r(r, max_iters=300, lr=0.005, n_restarts=3)
        
        if U is not None and residual < 1e-6:
            # Verify exact solution
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify_exact(U_list, V_list, W_list, tol=1e-8):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break
        elif U is not None and residual < 1e-3:
            # Close but not exact - try polishing
            # This would require more sophisticated optimization
            pass
    
    return best
# EVOLVE-BLOCK-END
