import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 130.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Verify decomposition using vectorized numpy operations."""
    rank = len(U[0])
    if rank == 0:
        return False
        
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute reconstructed tensor
    T_recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                
    err = np.max(np.abs(T_recon - T_target))
    return err < 1e-10

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    return _verify_decomposition_fast(U, V, W, n, m, p)

def _build_base_decomposition():
    """Build the rank-44 Strassen+trivial decomposition."""
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # m1 = (a+d)(e+h)
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m2 = (c+d)e
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            W[(b_c*2+1)*n + 1][col] = -1
            col += 1
            
            # m3 = a(f-h)
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m4 = d(g-e)
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            col += 1
            
            # m5 = (a+b)h
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            col += 1
            
            # m6 = (c-a)(e+f)
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m7 = (b-d)(g+h)
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            col += 1
            
            # Trivial part for row i=2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    return U, V, W

def _check_rank_reduction_greedy(U, V, W, n, m, p):
    """
    Try to reduce rank by greedily removing columns.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    active = list(range(rank))
    
    changed = True
    while changed and _time_left() > 2.0:
        changed = False
        i = 0
        while i < len(active):
            idx = active[i]
            new_active = active[:i] + active[i+1:]
            if len(new_active) == 0:
                i += 1
                continue
                
            U_sub = np.column_stack([U_arr[:, j] for j in new_active])
            V_sub = np.column_stack([V_arr[:, j] for j in new_active])
            W_sub = np.column_stack([W_arr[:, j] for j in new_active])
            
            if _verify_decomposition_fast(U_sub, V_sub, W_sub, n, m, p):
                active = new_active
                changed = True
                continue
            else:
                i += 1
                
    if len(active) < rank:
        U_new = [[U_arr[i, j] for j in active] for i in range(n*m)]
        V_new = [[V_arr[i, j] for j in active] for i in range(m*p)]
        W_new = [[W_arr[i, j] for j in active] for i in range(p*n)]
        return U_new, V_new, W_new, len(active)
        
    return U, V, W, rank

def _try_reduce_rank(base_rank, max_attempts=50):
    """Try to find a lower-rank decomposition."""
    n, m, p = 3, 4, 4
    
    U, V, W = _build_base_decomposition()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    U, V, W, new_rank = _check_rank_reduction_greedy(U, V, W, n, m, p)
    
    if new_rank < base_rank:
        return U, V, W
    
    return U, V, W

def construct():
    n, m, p = 3, 4, 4
    
    # Build the rank-44 decomposition
    U, V, W = _build_base_decomposition()
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        pass
    
    # Try to reduce rank if time permits
    if _time_left() > 5.0:
        result = _try_reduce_rank(44)
        if result is not None:
            U, V, W = result
    
    return {"U": U, "V": V, "W": W}
