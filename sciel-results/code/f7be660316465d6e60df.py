import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 2, 4
    
    # Target tensor T of shape (n*m, m*p, p*n) = (4, 8, 8)
    # T[u_idx, v_idx, w_idx] = 1 if it's a matrix multiplication term
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    # Term exists if A[i,j] * B[j,k] contributes to C[k,i]
    # So T[i*m+j, j*p+k, k*n+i] = 1
    
    T = np.zeros((4, 8, 8), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Try rank 13
    rank = 13
    # Initialize with Strassen-like structure or random
    # Better: Initialize with the 14-term solution and drop one term, then optimize?
    # Or just random init with gauge fixing.
    
    # Gauge fixing:
    # We can fix the first column of U to be [1, 0, 0, 0]^T?
    # No, U is 4x13.
    # We can fix U[0, 0] = 1, U[0, 1:] = 0?
    # And V[0, 0] = 1, etc?
    # This is tricky.
    
    # Let's try a simpler approach:
    # Use the 14-term solution, and try to find a linear dependence among the 14 vectors in the tensor space?
    # No, that's not how rank works.
    
    # Let's run a short optimization for rank 13.
    # We need to be careful with the cost function.
    # Cost: sum (sum_k U[i,k] V[j,k] W[l,k] - T[i,j,l])^2
    
    # Reshape T to (4*8*8, 1)
    T_flat = T.flatten()
    
    # Parameters: flatten U, V, W
    # U: 4*13 = 52
    # V: 8*13 = 104
    # W: 8*13 = 104
    # Total params: 260
    
    # Initial guess: Use the 14-term solution, take the first 13 terms?
    # Or average them?
    # Let's construct the 14-term solution first.
    
    U14 = np.zeros((4, 14))
    V14 = np.zeros((8, 14))
    W14 = np.zeros((8, 14))
    
    # Strassen terms for 2x2x2
    # (u, v, w)
    # u: (i, j), v: (j, k), w: (k, i)
    # k in {0, 1} for block 1, k in {2, 3} for block 2
    
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (0,0,-1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    for block_idx, k_offset in enumerate([0, 2]):
        col_base = block_idx * 7
        for t_idx, (u_entries, v_entries, w_entries) in enumerate(strassen_terms):
            col = col_base + t_idx
            for i, j, val in u_entries:
                U14[i * m + j, col] += val
            for j, k, val in v_entries:
                k_local = k + k_offset
                V14[j * p + k_local, col] += val
            for k, i, val in w_entries:
                k_local = k + k_offset
                W14[k_local * n + i, col] += val
                
    # Check rank 14 error
    # def compute_error(params):
    #     ...
    
    # Try to optimize rank 13 starting from U14[:, :13], etc.
    # This might work if one term is redundant or can be approximated.
    
    def residual(params):
        U = params[:52].reshape(4, 13)
        V = params[52:156].reshape(8, 13)
        W = params[156:260].reshape(8, 13)
        
        # Compute tensor product
        # T_calc[i, j, l] = sum_k U[i, k] V[j, k] W[l, k]
        # Use einsum for speed
        T_calc = np.einsum('ik,jk,lk->ijl', U, V, W)
        return T_calc.flatten() - T_flat
        
    # Initial guess
    x0 = np.concatenate([U14[:, :13].flatten(), V14[:, :13].flatten(), W14[:, :13].flatten()])
    
    # Optimize
    try:
        result = least_squares(residual, x0, method='lm', max_nfev=1000)
        if result.cost < 1e-10:
            # Found rank 13!
            U = result.x[:52].reshape(4, 13)
            V = result.x[52:156].reshape(8, 13)
            W = result.x[156:260].reshape(8, 13)
            # Round to nearest integer/half-integer if possible?
            # The problem asks for integer/half-integer entries.
            # If the solution is close to such, we can snap it.
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify snapped solution
            T_snap = np.einsum('ik,jk,lk->ijl', U_int, V_int, W_int)
            if np.allclose(T_snap.flatten(), T_flat, atol=1e-9):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    except Exception as e:
        pass
        
    # Fallback to rank 14
    U14_list = U14.tolist()
    V14_list = V14.tolist()
    W14_list = W14.tolist()
    
    return {"U": U14_list, "V": V14_list, "W": W14_list}
