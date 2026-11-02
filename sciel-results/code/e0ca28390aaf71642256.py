import numpy as np
import time
import scipy.optimize as opt
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def flatten(U, V, W):
    """Flatten U, V, W into a single vector."""
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def unflatten(x, n, m, p, rank):
    """Unflatten vector x into U, V, W."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = x[:size_u].reshape(n * m, rank)
    V = x[size_u:size_u + size_v].reshape(m * p, rank)
    W = x[size_u + size_v:size_u + size_v + size_w].reshape(p * n, rank)
    return U, V, W

def loss_function(x, n, m, p, rank):
    """Compute the loss: sum of squared errors."""
    U, V, W = unflatten(x, n, m, p, rank)
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                # val = sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                error += (val - 1.0) ** 2
    return error

def gradient(x, n, m, p, rank):
    """Compute the gradient of the loss function."""
    U, V, W = unflatten(x, n, m, p, rank)
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                err = val - 1.0
                # d/dU = 2 * err * V * W
                dU[idx_u, :] += 2 * err * (V[idx_v, :] * W[idx_w, :])
                dV[idx_v, :] += 2 * err * (U[idx_u, :] * W[idx_w, :])
                dW[idx_w, :] += 2 * err * (U[idx_u, :] * V[idx_v, :])
                
    return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def snap_to_rationals(U, V, W, denominators=[1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]):
    """Try to snap U, V, W to rational numbers with small denominators."""
    U_snap = U.copy()
    V_snap = V.copy()
    W_snap = W.copy()
    
    for den in denominators:
        inv_den = 1.0 / den
        U_round = np.round(U / inv_den) * inv_den
        V_round = np.round(V / inv_den) * inv_den
        W_round = np.round(W / inv_den) * inv_den
        
        # Check if the snapped version is close to the original
        if (np.linalg.norm(U - U_round) < 1e-5 and 
            np.linalg.norm(V - V_round) < 1e-5 and 
            np.linalg.norm(W - W_round) < 1e-5):
            # Verify exactness
            if validate_exact(U_round.tolist(), V_round.tolist(), W_round.tolist(), 2, 5, 5):
                return U_round, V_round, W_round, True
                
    return None, None, None, False

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with schoolbook
    U_best, V_best, W_best = schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Target ranks: 26 is the known minimum for 2x5x5
    # We try from 26 upwards to 45
    target_ranks = list(range(26, 46))
    
    for rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt + rank)
            # Initialize with small random values
            x0 = np.random.randn(n * m * rank + m * p * rank + p * n * rank) * 0.1
            
            # Use L-BFGS-B optimizer
            try:
                result = opt.minimize(
                    loss_function, 
                    x0, 
                    args=(n, m, p, rank),
                    method='L-BFGS-B',
                    jac=gradient,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.fun < 1e-6:
                    # Unflatten
                    U_opt, V_opt, W_opt = unflatten(result.x, n, m, p, rank)
                    
                    # Try to snap to rationals
                    U_snap, V_snap, W_snap, success = snap_to_rationals(U_opt, V_opt, W_opt)
                    
                    if success:
                        U_best, V_best, W_best = U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        best_rank = rank
                        found = True
                        break
            except Exception as e:
                pass
        
        if found:
            break
            
    return {"U": U_best, "V": V_best, "W": W_best}
