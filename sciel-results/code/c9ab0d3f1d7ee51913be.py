"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # We need: sum_r U[a,r] V[b,r] W[c,r] == Target[a,b,c]
    
    # Construct Target Tensor
    # Target is 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Else 0
    
    # Efficient check:
    # Iterate over all non-zero target entries and check sum is 1
    # Iterate over a sample of zero entries and check sum is 0
    
    U_t = U.T # (R, nm)
    V_t = V.T # (R, mp)
    W_t = W.T # (R, pn)
    
    # Check 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U_t[:, a] * V_t[:, b] * W_t[:, c])
                if abs(s - 1.0) > tol:
                    return False
    
    # Check 0s (Sample or Full)
    # Full check is 10*25*10 = 2500 checks. Very fast.
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                s = np.sum(U_t[:, a] * V_t[:, b] * W_t[:, c])
                if abs(s) > tol:
                    return False
    return True

def _make_schoolbook(n, m, p):
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

def _als_decompose(n, m, p, R, max_iters=50, time_limit=5.0):
    """
    Try to find a decomposition of rank R using Alternating Least Squares.
    Returns (U, V, W) if successful, else None.
    """
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor flattened for LS
    # We need T_flat of shape (nm, mp * pn)
    # T_flat[a, b * pn + c] = 1 if valid, else 0
    
    T_flat = np.zeros((nm, mp * pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_flat[a, b * pn + c] = 1.0
    
    # Random initialization
    U = np.random.randn(nm, R)
    V = np.random.randn(mp, R)
    W = np.random.randn(pn, R)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        # T_flat = U @ Y_U
        # Y_U has shape (mp * pn, R)
        # Row (b, c) of Y_U is V[b] * W[c] (element-wise)
        Y_U = np.zeros((mp * pn, R))
        for b in range(mp):
            for c in range(pn):
                Y_U[b * pn + c, :] = V[b, :] * W[c, :]
        
        # Solve U Y_U = T_flat
        # U = T_flat @ Y_U^T @ (Y_U Y_U^T)^-1
        # Use np.linalg.lstsq
        try:
            U, _, _, _ = np.linalg.lstsq(Y_U.T, T_flat.T, rcond=None)
            U = U.T # Back to (nm, R)
        except:
            return None
        
        # Update V
        # T_flat_reshaped_V: (mp, nm * pn)
        # T_flat[a, b*pn+c] -> T_V[b, a*pn+c]
        T_V = T_flat.T.reshape(mp, nm * pn).T # (nm * pn, mp) -> transpose to (mp, nm*pn)?
        # Let's be careful.
        # We want V @ Y_V = T_V_target
        # T_V_target[b, a*pn+c] = T_flat[a, b*pn+c]
        # Y_V row (a, c) is U[a] * W[c]
        
        T_V_target = np.zeros((mp, nm * pn))
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    T_V_target[b, a * pn + c] = T_flat[a, b * pn + c]
        
        Y_V = np.zeros((nm * pn, R))
        for a in range(nm):
            for c in range(pn):
                Y_V[a * pn + c, :] = U[a, :] * W[c, :]
        
        try:
            V, _, _, _ = np.linalg.lstsq(Y_V.T, T_V_target.T, rcond=None)
            V = V.T
        except:
            return None
            
        # Update W
        # T_W_target[c, a*mp+b] = T_flat[a, b*pn+c]
        # Y_W row (a, b) is U[a] * V[b]
        
        T_W_target = np.zeros((pn, nm * mp))
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    T_W_target[c, a * mp + b] = T_flat[a, b * pn + c]
        
        Y_W = np.zeros((nm * mp, R))
        for a in range(nm):
            for b in range(mp):
                Y_W[a * mp + b, :] = U[a, :] * V[b, :]
        
        try:
            W, _, _, _ = np.linalg.lstsq(Y_W.T, T_W_target.T, rcond=None)
            W = W.T
        except:
            return None
            
    # Round to nearest 0.5
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def still_going():
        return time.time() < deadline
    
    # Try to find lower rank decompositions
    # Theoretical rank might be lower, but exact integer solutions are rare.
    # Try ranks 49 down to 45
    for target_rank in range(best_rank - 1, max(40, best_rank - 10), -1):
        if not still_going():
            break
        
        remaining_time = deadline - time.time()
        # Allocate time per rank
        time_per_rank = min(10.0, remaining_time * 0.5)
        
        # Try multiple random restarts for this rank
        attempts = 0
        found = False
        while time.time() - start_time < deadline and not found:
            attempts += 1
            # Limit attempts to prevent infinite loops if time limit is generous
            if attempts > 5: 
                break
            
            result = _als_decompose(n, m, p, target_rank, max_iters=20, time_limit=time_per_rank / 5)
            if result is not None:
                best = result
                best_rank = target_rank
                found = True
                break
                
    return best
# EVOLVE-BLOCK-END
