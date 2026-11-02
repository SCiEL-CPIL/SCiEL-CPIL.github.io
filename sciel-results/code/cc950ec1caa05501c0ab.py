"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i,j,k] = (i*m+j, j*p+k, k*n+i) as a dict."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = (i * m + j, j * p + k, k * n + i)
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Check if U, V, W (lists of lists) form a valid rank-r decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must be 1 for all (i,j,k).
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # T is a 3-tensor: T[i,j,k]
    # We can check by constructing the full tensor or checking all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, timeout=10.0):
    """
    Try to find a rank-r decomposition using random initialization and gradient descent.
    Returns (U, V, W) as lists of lists of Fractions if found, else None.
    """
    start_time = time.time()
    np.random.seed(42)  # Deterministic for reproducibility
    
    # We use a simple heuristic: random init + gradient descent on the loss
    # Loss = sum_{i,j,k} (1 - sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c])^2
    # This is non-convex. We'll try a few random inits.
    
    n_vars = (n * m + m * p + p * n) * r
    # This is too many variables for large r, but for r <= 18 it's manageable?
    # n=2, m=3, p=3 -> n*m=6, m*p=9, p*n=6. Total rows = 21.
    # r=17 -> 21*17 = 357 variables.
    
    # We'll use a simple gradient descent.
    # To speed up, we can fix one matrix? No, symmetry is broken.
    
    # Let's try to use the known structure: maybe some entries are 0 or 1?
    # For now, random init.
    
    best_loss = float('inf')
    best_params = None
    
    for attempt in range(5):
        if time.time() - start_time > timeout:
            break
            
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Normalize?
        # U /= np.linalg.norm(U, axis=0, keepdims=True)
        
        lr = 0.01
        for step in range(200):
            if time.time() - start_time > timeout:
                break
            
            # Compute gradients
            # L = sum_{i,j,k} (1 - sum_c U[i,j,c] V[j,k,c] W[k,i,c])^2
            # dL/dU[i,j,c] = -2 * sum_{k} (1 - T[i,j,k]) * V[j,k,c] * W[k,i,c]
            # where T[i,j,k] = sum_c' U[i,j,c'] V[j,k,c'] W[k,i,c']
            
            T = np.zeros((n, m, p))
            for c in range(r):
                Uc = U[:, c]
                Vc = V[:, c]
                Wc = W[:, c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
            
            loss = np.sum((1.0 - T) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_params = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-10:
                # Found a solution
                return best_params[0], best_params[1], best_params[2]
            
            # Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = 1.0 - T[i, j, k]
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for c in range(r):
                            dU[u_idx, c] += -2 * err * V[v_idx, c] * W[w_idx, c]
                            dV[v_idx, c] += -2 * err * U[u_idx, c] * W[w_idx, c]
                            dW[w_idx, c] += -2 * err * U[u_idx, c] * V[v_idx, c]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive LR
            if loss > best_loss * 1.1:
                lr *= 0.9
            else:
                lr = min(lr * 1.05, 0.05)
        
        if best_loss < 1e-8:
            # Close enough, try to polish
            U, V, W = best_params
            # Convert to Fractions and check
            U_f = [[Fraction(x).limit_denominator(1000) for x in row] for row in U]
            V_f = [[Fraction(x).limit_denominator(1000) for x in row] for row in V]
            W_f = [[Fraction(x).limit_denominator(1000) for x in row] for row in W]
            
            if _check_decomposition(U_f, V_f, W_f, n, m, p, tol=1e-9):
                return U_f, V_f, W_f
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find lower rank
    # Known: rank(2x3, 3x3) is likely 18, but let's check 17, 16...
    # We have 120s total. Let's allocate 100s to search.
    deadline = time.time() + 100
    
    # Start from rank_schoolbook - 1 down to max(rank_schoolbook - 5, 10)
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 5, 10) - 1, -1):
        if time.time() > deadline - 10:
            break
        
        print(f"Trying rank {r}...")
        res = _try_rank(r, n, m, p, timeout=15.0)
        if res is not None:
            U, V, W = res
            # Convert to lists of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Final check
            if _check_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
                print(f"Found rank {r} decomposition!")
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Schoolbook
    rank = rank_schoolbook
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
