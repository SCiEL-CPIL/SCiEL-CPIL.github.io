"""Search for a lower-rank exact tensor decomposition of 3x4x8 matrix multiplication."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _matmul_decomposition(n, m, p, U, V, W):
    """
    Check whether U (n*m, r), V (m*p, r), W (p*n, r) exactly decompose the matmul tensor.
    The condition is: for each (i, j, k):
        sum_r  U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] == 1
    """
    r = U.shape[1]
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for rr in range(r):
                    val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _verify(n, m, p, U, V, W):
    """
    Verify exact decomposition. U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Uses the tensor condition:
      For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                if abs(s - 1.0) > 1e-7:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
    
    def to_list(M):
        result = []
        for row in M:
            r = []
            for x in row:
                if abs(x * 2 - round(x * 2)) < 1e-10:
                    val = x * 2
                    iv = int(round(val))
                    if iv == 0:
                        r.append(0)
                    elif abs(iv % 2) == 0:
                        r.append(iv // 2)
                    else:
                        r.append(iv / 2.0)
                else:
                    r.append(x)
            result.append(r)
        return result
    
    best = (rank_sb, U_sb.copy(), V_sb.copy(), W_sb.copy())
    best_verified = True
    
    random.seed(42)
    
    # Try to find lower rank decompositions
    # Strategy: iterative tensor decomposition using random initialization + local refinement
    # We search from rank_sb-1 down
    
    for target_rank in range(rank_sb - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations at this rank
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with small random values
            scale = 0.5
            U_r = np.random.randn(n * m, target_rank) * scale
            V_r = np.random.randn(m * p, target_rank) * scale
            W_r = np.random.randn(p * n, target_rank) * scale
            
            # Simple local search: gradient-free, random perturbations
            best_cost = None
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute cost: sum over all (i,j,k) of (sum_r U*V*W - 1)^2
                cost = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(target_rank):
                                s += U_r[i*m+j, rr] * V_r[j*p+k, rr] * W_r[k*n+i, rr]
                            cost += (s - 1.0) ** 2
                
                if best_cost is None:
                    best_cost = cost
                    best_U = U_r.copy()
                    best_V = V_r.copy()
                    best_W = W_r.copy()
                
                if cost < best_cost:
                    best_cost = cost
                    best_U = U_r.copy()
                    best_V = V_r.copy()
                    best_W = W_r.copy()
                
                if best_cost < 1e-8:
                    break
                
                # Random perturbation
                mask = np.random.choice(target_rank, size=max(1, target_rank // 3), replace=False)
                U_r[:, mask] += np.random.randn(n * m, len(mask)) * 0.1
                V_r[:, mask] += np.random.randn(m * p, len(mask)) * 0.1
                W_r[:, mask] += np.random.randn(p * n, len(mask)) * 0.1
            
            if best_cost < 1e-10:
                # Verify
                if _verify(n, m, p, best_U, best_V, best_W):
                    # Try to reduce to integer/half-integer
                    # Check if entries are close to half-integers
                    close = True
                    for M in [best_U, best_V, best_W]:
                        for val in M.flat:
                            if abs(val * 2 - round(val * 2)) > 1e-8:
                                close = False
                                break
                        if not close:
                            break
                    if close:
                        best = (target_rank, best_U, best_V, best_W)
                        best_verified = True
                        # Don't break; we might find even lower
                        break
            if time.time() > deadline:
                break
        
        # If we found a verified decomposition at this rank, continue searching lower
        # If we're getting slow, reduce attempts
        if time.time() > deadline - 30:
            break
    
    if best_verified:
        r, Ur, Vr, Wr = best
        return {
            "U": to_list(Ur),
            "V": to_list(Vr),
            "W": to_list(Wr)
        }
    
    # Fallback: schoolbook
    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }

# EVOLVE-BLOCK-END
