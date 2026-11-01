"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p, tol=1e-10):
    """Verify exact decomposition using integer/rational arithmetic if possible."""
    rank = len(U[0])
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i][c]) for c in range(rank)] for i in range(n * m)]
    V_frac = [[Fraction(V[i][c]) for c in range(rank)] for i in range(m * p)]
    W_frac = [[Fraction(W[i][c]) for c in range(rank)] for i in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if val != 1:
                    return False
    return True

def _verify_numeric(U, V, W, n, m, p, tol=1e-9):
    """Fast numeric verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _build_target_tensor(n, m, p)
    return np.max(np.abs(T_approx - T_target)) < tol

def _round_to_halfint(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2.0

def _try_polish(U, V, W, rank, n, m, p, T_flat, max_iter=200):
    """Polish a near-exact decomposition to exactness using Newton's method on the tensor equations."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    for _ in range(max_iter):
        if _remaining() < 5:
            break
        
        # Compute current tensor
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T_target  # shape (n*m, m*p, p*n)
        max_err = np.max(np.abs(diff))
        
        if max_err < 1e-12:
            break
        
        # Gradient-based step with adaptive learning rate
        lr = 0.01
        for _ in range(10):
            dU = np.einsum('abc,br,cr->ar', diff, V, W)
            dV = np.einsum('abc,ar,cr->br', diff, U, W)
            dW = np.einsum('abc,ar,br->cr', diff, U, V)
            
            U_new = U - lr * dU
            V_new = V - lr * dV
            W_new = W - lr * dW
            
            T_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            new_err = np.max(np.abs(T_new - T_target))
            
            if new_err < max_err:
                U, V, W = U_new, V_new, W_new
                max_err = new_err
                break
            else:
                lr *= 0.5
        
        if max_err < 1e-12:
            break
    
    return U, V, W, max_err

def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor
    T_target = _build_target_tensor(n, m, p)
    T_flat = T_target.reshape(-1)
    
    # Schoolbook decomposition (rank 18)
    rank_sb = n * m * p
    U0 = np.zeros((n * m, rank_sb))
    V0 = np.zeros((m * p, rank_sb))
    W0 = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # Known results suggest rank is around 13-15 for 2x3x3
    rng = np.random.default_rng(42)
    
    for target_rank in range(17, 12, -1):  # Try 17 down to 13
        if _remaining() < 10:
            break
        
        found = False
        max_restarts = 30
        for restart in range(max_restarts):
            if _remaining() < 5:
                break
            
            # Random initialization
            U = rng.standard_normal((n * m, target_rank)) * 0.5
            V = rng.standard_normal((m * p, target_rank)) * 0.5
            W = rng.standard_normal((p * n, target_rank)) * 0.5
            
            # Optimize using gradient descent with adaptive step
            lr = 0.01
            for iteration in range(300):
                if _remaining() < 5:
                    break
                
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_approx - T_target
                max_err = np.max(np.abs(diff))
                
                if max_err < 1e-10:
                    break
                
                dU = np.einsum('abc,br,cr->ar', diff, V, W)
                dV = np.einsum('abc,ar,cr->br', diff, U, W)
                dW = np.einsum('abc,ar,br->cr', diff, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adapt learning rate
                if max_err > 1.0:
                    lr *= 0.5
                elif iteration % 50 == 0:
                    lr *= 1.1
            
            # Polish the solution
            U, V, W, final_err = _try_polish(U, V, W, target_rank, n, m, p, T_flat)
            
            if final_err < 1e-10:
                # Try to round to half-integers and verify
                U_list = [[_round_to_halfint(x) for x in row] for row in U]
                V_list = [[_round_to_halfint(x) for x in row] for row in V]
                W_list = [[_round_to_halfint(x) for x in row] for row in W]
                
                if _verify_numeric(U_list, V_list, W_list, n, m, p, tol=1e-9):
                    best_U, best_V, best_W = U_list, V_list, W_list
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Keep the unrounded version if it's numerically exact
                    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue  # Try lower rank
    
    # Return the best result
    if best_rank < rank_sb:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback: schoolbook
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
