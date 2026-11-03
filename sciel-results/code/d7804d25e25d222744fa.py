"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 120.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # 1
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # 2
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            # 3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # 4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            # 5
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            # 6
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # 7
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    return U, V, W

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _flatten_tensor(U, V, W, n, m, p):
    """
    Flatten the tensor defined by U, V, W into a vector of size n*m*p.
    The tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    """
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        # u_r is vector of size n*m
        # v_r is vector of size m*p
        # w_r is vector of size p*n
        # T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can vectorize this using einsum or loops
        for i in range(n):
            for j in range(m):
                u_val = U[i*m+j, r]
                if u_val == 0: continue
                for k in range(p):
                    T[i,j,k] += u_val * V[j*p+k, r] * W[k*n+i, r]
    return T.flatten()

def _try_reduce_rank(U, V, W, n, m, p):
    """
    Try to reduce rank by checking linear dependencies using Least Squares.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # Precompute the flattened tensor for the current decomposition
    # This is the target vector
    target_T = _flatten_tensor(current_U, current_V, current_W, n, m, p)
    
    removed = []
    
    # Try to remove terms one by one
    # We sort by norm to remove small ones first? Or random?
    # Let's try all, but limit iterations
    order = np.arange(current_rank)
    
    for r in order:
        if r in removed:
            continue
        if _time_left() < 20:
            break
            
        # Check if term r is in the span of the others
        # We want to solve sum_{s != r} c_s T_s = T_r
        # Where T_s is the flattened tensor for term s
        
        # Build the matrix of other terms
        # Size: (n*m*p) x (current_rank - 1)
        # This might be memory intensive if done naively, but 120x120=14400 floats is fine.
        
        other_indices = [i for i in range(current_rank) if i != r and i not in removed]
        if not other_indices:
            continue
            
        # Construct A matrix: columns are flattened tensors of other terms
        # Construct b vector: flattened tensor of term r
        
        # Efficiently compute flattened tensors for all terms
        # We already have target_T which is sum of all.
        # T_r = target_T - sum_{s != r} T_s.
        # This doesn't help directly for the LS problem unless we compute T_s individually.
        
        # Let's compute T_s for all s once, store them.
        # This is expensive if done in the loop.
        # Better: Precompute all T_s.
        
    # Precompute all individual tensors T_s
    T_terms = []
    for s in range(current_rank):
        T_s = np.zeros((n*m*p))
        for i in range(n):
            for j in range(m):
                u_val = current_U[i*m+j, s]
                if u_val == 0: continue
                for k in range(p):
                    T_s[i*m*p + j*p + k] += u_val * current_V[j*p+k, s] * current_W[k*n+i, s]
        T_terms.append(T_s)
        
    T_terms = np.array(T_terms).T # Shape: (n*m*p, current_rank)
    
    # Now try to remove terms
    active_indices = list(range(current_rank))
    
    for r in range(current_rank):
        if _time_left() < 10:
            break
        if r not in active_indices:
            continue
            
        # Try to express T_r as linear combo of others in active_indices \ {r}
        others = [i for i in active_indices if i != r]
        if not others:
            continue
            
        A = T_terms[:, others] # (N, R-1)
        b = T_terms[:, r]      # (N,)
        
        # Solve A c = b
        try:
            # Use least squares
            c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check residual
            res_norm = np.linalg.norm(A @ c - b)
            if res_norm < 1e-8:
                # Term r is redundant
                active_indices.remove(r)
                print(f"Removed term {r}, rank: {len(active_indices)}")
        except:
            pass
            
    if len(active_indices) < current_rank:
        new_U = current_U[:, active_indices]
        new_V = current_V[:, active_indices]
        new_W = current_W[:, active_indices]
        return new_U, new_V, new_W, len(active_indices)
        
    return current_U, current_V, current_W, current_rank

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_54_rank(n, m, p)
    
    if not _verify_full(U, V, W, n, m, p):
        # Fallback
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    if _time_left() > 30:
        U_r, V_r, W_r, new_rank = _try_reduce_rank(U, V, W, n, m, p)
        if new_rank < U.shape[1]:
            if _verify_full(U_r, V_r, W_r, n, m, p):
                U, V, W = U_r, V_r, W_r
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
