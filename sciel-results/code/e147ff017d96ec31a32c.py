import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    Target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                Target[alpha, beta, gamma] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        if not np.allclose(T_calc, Target, atol=tol):
            return False
        return True

    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0

    def is_half_integer(x, tol=1e-5):
        return np.allclose(x, snap_to_half_int(x), atol=tol)

    def compute_residual(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_calc - Target) ** 2)

    # Precompute reshaped Target matrices for fast solving
    # For solving W: T is (dim_U, dim_V, dim_W). We want B (dim_U*dim_V, dim_W)
    B_W = Target.reshape(dim_U * dim_V, dim_W)
    
    # For solving U: T is (dim_U, dim_V, dim_W). We want T_U (dim_U, dim_V*dim_W)
    T_U = Target.reshape(dim_U, dim_V * dim_W)
    
    # For solving V: T is (dim_U, dim_V, dim_W). We want T_V (dim_V, dim_U*dim_W)
    # Note: T_V[beta, alpha*dim_W + gamma] = T[alpha, beta, gamma]
    T_V = np.zeros((dim_V, dim_U * dim_W))
    for a in range(dim_U):
        for b in range(dim_V):
            for g in range(dim_W):
                T_V[b, a * dim_W + g] = Target[a, b, g]

    def solve_for_W(U, V):
        """
        Given U and V, solve for W using least squares.
        A (dim_U*dim_V, R) where A[alpha*dim_V+beta, r] = U[alpha, r]*V[beta, r]
        B (dim_U*dim_V, dim_W)
        W (dim_W, R)
        W = (B A^+)^T ? No.
        B = A W^T
        W^T = A^+ B
        W = (A^+ B)^T = B^T A^+^T
        """
        R = U.shape[1]
        # M (dim_U, dim_V, R)
        M = np.einsum('ar,br->abr', U, V)
        A = M.reshape(dim_U * dim_V, R)
        
        # A^+ (R, dim_U*dim_V)
        try:
            A_pinv = np.linalg.pinv(A)
            W_T = A_pinv @ B_W # (R, dim_W)
            W = W_T.T # (dim_W, R)
        except:
            W = np.zeros((dim_W, R))
            
        return W

    def solve_for_U(V, W):
        """
        Given V and W, solve for U.
        T_U (dim_U, dim_V*dim_W)
        C (dim_V*dim_W, R) where C[beta*dim_W+gamma, r] = V[beta, r]*W[gamma, r]
        T_U = U C^T
        U = T_U (C^T)^+ = T_U C^+^T
        """
        R = V.shape[1]
        # N (dim_V, dim_W, R)
        N = np.einsum('br,cr->bcr', V, W)
        C = N.reshape(dim_V * dim_W, R)
        
        try:
            C_pinv = np.linalg.pinv(C) # (R, dim_V*dim_W)
            U_T = (C_pinv @ T_U.T).T # (dim_U, R)
            U = U_T
        except:
            U = np.zeros((dim_U, R))
            
        return U

    def solve_for_V(U, W):
        """
        Given U and W, solve for V.
        T_V (dim_V, dim_U*dim_W)
        D (dim_U*dim_W, R) where D[alpha*dim_W+gamma, r] = U[alpha, r]*W[gamma, r]
        T_V = V D^T
        V = T_V D^+^T
        """
        R = U.shape[1]
        # O (dim_U, dim_W, R)
        O = np.einsum('ar,cr->acr', U, W)
        D = O.reshape(dim_U * dim_W, R)
        
        try:
            D_pinv = np.linalg.pinv(D) # (R, dim_U*dim_W)
            V_T = (D_pinv @ T_V.T).T # (dim_V, R)
            V = V_T
        except:
            V = np.zeros((dim_V, R))
            
        return V

    start_time = time.time()
    time_limit = 140.0 # Leave margin for 150s kill

    # Schoolbook solution (Rank 75)
    best_rank = n * m * p
    U_sb = np.zeros((dim_U, best_rank))
    V_sb = np.zeros((dim_V, best_rank))
    W_sb = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    if verify(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        best_result = None

    found = False
    
    # Values for discrete search
    # Include a wider range of half-integers
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    for R in range(25, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Number of restarts per rank
        # Allocate time budget dynamically? Or fixed?
        # Let's try a fixed number of restarts, but check time frequently.
        num_restarts = 500 # Increased because solvers are faster
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V with random half-integers
            U = np.array([random.choice(values) for _ in range(dim_U * R)]).reshape(dim_U, R)
            V = np.array([random.choice(values) for _ in range(dim_V * R)]).reshape(dim_V, R)
            W = np.array([random.choice(values) for _ in range(dim_W * R)]).reshape(dim_W, R)
            
            # ALS Phase
            # Run a few iterations to converge
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    break
                W = solve_for_W(U, V)
                U = solve_for_U(V, W)
                V = solve_for_V(U, W)
            
            # Snap and Verify
            U_snap = snap_to_half_int(U)
            V_snap = snap_to_half_int(V)
            W_snap = snap_to_half_int(W)
            
            if verify(U_snap, V_snap, W_snap):
                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                best_rank = R
                found = True
                break
            
            # Local Search Phase on snapped values
            # Try to improve the residual by changing one entry and re-solving
            current_res = compute_residual(U_snap, V_snap, W_snap)
            improved = True
            max_steps = 20 # Keep short
            
            while improved and max_steps > 0 and (time.time() - start_time < time_limit):
                improved = False
                max_steps -= 1
                
                # Try changing a random entry in U, V, or W
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat_name = 'U'
                    arr = U_snap
                elif mat_idx == 1:
                    mat_name = 'V'
                    arr = V_snap
                else:
                    mat_name = 'W'
                    arr = W_snap
                    
                r_idx, c_idx = random.randint(0, arr.shape[0]-1), random.randint(0, arr.shape[1]-1)
                
                old_val = arr[r_idx, c_idx]
                new_val = random.choice([v for v in values if v != old_val])
                
                # Apply change
                if mat_name == 'U':
                    U_snap[r_idx, c_idx] = new_val
                elif mat_name == 'V':
                    V_snap[r_idx, c_idx] = new_val
                else:
                    W_snap[r_idx, c_idx] = new_val
                    
                # Re-solve the other two matrices
                if mat_name == 'U':
                    # U is fixed (snapped). Solve V, W.
                    V_new = solve_for_V(U_snap, W_snap)
                    V_new_snap = snap_to_half_int(V_new)
                    W_new = solve_for_W(U_snap, V_new_snap)
                    W_new_snap = snap_to_half_int(W_new)
                    
                    res_new = compute_residual(U_snap, V_new_snap, W_new_snap)
                    
                    if res_new < current_res:
                        current_res = res_new
                        V_snap = V_new_snap
                        W_snap = W_new_snap
                        improved = True
                        if current_res < 1e-9:
                            if verify(U_snap, V_snap, W_snap):
                                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                                best_rank = R
                                found = True
                                break
                elif mat_name == 'V':
                    V_new = solve_for_U(U_snap, W_snap)
                    V_new_snap = snap_to_half_int(V_new)
                    W_new = solve_for_W(U_snap, V_new_snap)
                    W_new_snap = snap_to_half_int(W_new)
                    
                    res_new = compute_residual(U_snap, V_new_snap, W_snap) # Wait, W is also updated?
                    # We updated V and W.
                    res_new = compute_residual(U_snap, V_new_snap, W_new_snap)
                    
                    if res_new < current_res:
                        current_res = res_new
                        V_snap = V_new_snap
                        W_snap = W_new_snap
                        improved = True
                        if current_res < 1e-9:
                            if verify(U_snap, V_snap, W_snap):
                                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                                best_rank = R
                                found = True
                                break
                else:
                    U_new = solve_for_U(V_snap, W_snap)
                    U_new_snap = snap_to_half_int(U_new)
                    V_new = solve_for_V(U_new_snap, W_snap)
                    V_new_snap = snap_to_half_int(V_new)
                    
                    res_new = compute_residual(U_new_snap, V_new_snap, W_snap)
                    
                    if res_new < current_res:
                        current_res = res_new
                        U_snap = U_new_snap
                        V_snap = V_new_snap
                        improved = True
                        if current_res < 1e-9:
                            if verify(U_snap, V_snap, W_snap):
                                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                                best_rank = R
                                found = True
                                break
                if found:
                    break

        if found:
            break

    if best_result is None:
        # Fallback to schoolbook if no better found
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
# EVOLVE-BLOCK-END
