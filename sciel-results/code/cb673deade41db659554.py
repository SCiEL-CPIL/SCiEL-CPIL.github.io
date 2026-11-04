import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_target(n, m, p):
    """Build the target tensor T where T[i*m+j, j*p+k, k*n+i] = 1"""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check off-diagonal entries
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
    return True

def _search_333(rank, max_steps=500, lr=0.01):
    """Search for a rank decomposition of 3x3x3 matrix multiplication."""
    n, m, p = 3, 3, 3
    T = _build_target(n, m, p)
    nm, mp, pn = n*m, m*p, p*n
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try a few random restarts
    for trial in range(3):
        np.random.seed(42 + rank * 10 + trial)
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        for step in range(max_steps):
            # Compute error and gradients
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_rec - T
            err = np.sum(diff**2)
            
            if err < 1e-10:
                return U, V, W
            
            grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Simple SGD with momentum-like decay
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            lr *= 0.99
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if step % 50 == 0 and err < 1e-4:
                # Try snapping
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap

    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy: Find low-rank 3x3x3 and map to 3x3x6
    # If rank_333 is r, rank_336 is 2*r
    # Try ranks for 333: 19, 20, 21, 22, 23, 24, 25, 26, 27
    
    for r_333 in range(19, 28):
        if time.time() > deadline - 20:
            break
            
        res = _search_333(r_333)
        if res is not None:
            U_333, V_333, W_333 = res
            rank_336 = 2 * r_333
            
            # Map to 3x3x6
            U_336 = np.zeros((9, rank_336))
            V_336 = np.zeros((18, rank_336))
            W_336 = np.zeros((18, rank_336))
            
            # Copy first half (k=0,1,2)
            U_336[:, :r_333] = U_333
            for j in range(3):
                for k in range(3):
                    V_336[j*6+k, :r_333] = V_333[j*3+k, :]
                    W_336[k*3+i_idx, :r_333] = W_333[k*3+i_idx, :] # Wait, W index mapping
                    # W index in 333 is k*3+i. In 336 it is k*3+i for k<3.
            # Actually, let's do it properly
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        # First block
                        V_336[j*6+k, :r_333] = V_333[j*3+k, :]
                        W_336[k*3+i, :r_333] = W_333[k*3+i, :]
                        # Second block (shift k by 3)
                        V_336[j*6+(k+3), r_333:] = V_333[j*3+k, :]
                        W_336[(k+3)*3+i, r_333:] = W_333[k*3+i, :]
                        
            # Verify
            if _verify(U_336, V_336, W_336, n, m, p):
                if rank_336 < best_rank:
                    best_rank = rank_336
                    best_U, best_V, best_W = U_336.copy(), V_336.copy(), W_336.copy()
                break # Found a good one, stop searching lower ranks? No, we went up.
                    # We should search from low to high. So if we find r=19, we stop.
                    
    # If no 333 solution found, try direct 336 search?
    # For now, return best found
    
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
