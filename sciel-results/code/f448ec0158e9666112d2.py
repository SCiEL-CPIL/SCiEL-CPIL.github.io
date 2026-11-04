"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find the minimal rank r such that this holds for all i,j,k.
    
    def schoolbook():
        rank = n * m * p
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

    def verify(U, V, W):
        """
        Verify if U, V, W form a valid decomposition.
        U, V, W are numpy arrays or lists.
        Returns True if valid, False otherwise.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Build tensor efficiently using einsum
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_small(U, V, W, n_s, m_s, p_s):
        """Verify decomposition for small dimensions."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        U_reshaped = U_arr.reshape(n_s, m_s, rank)
        V_reshaped = V_arr.reshape(m_s, p_s, rank)
        W_reshaped = W_arr.reshape(p_s, n_s, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return [[int(round(x * 2)) / 2.0 for x in row] for row in arr]

    def solve_direct(n_s, m_s, p_s, rank_s, max_attempts=10, max_iters=200):
        """
        Solve the polynomial system for rank rank_s decomposition of n_s x m_s x p_s
        directly using alternating minimization and least_squares.
        """
        num_constraints = n_s * m_s * p_s
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Different initialization strategies
            if attempt % 5 == 0:
                scale = 0.5
            elif attempt % 5 == 1:
                scale = 1.0
            elif attempt % 5 == 2:
                scale = 2.0
            elif attempt % 5 == 3:
                scale = 0.3
            else:
                scale = 1.5
            
            U_s = np.random.randn(n_s*m_s, rank_s) * scale
            V_s = np.random.randn(m_s*p_s, rank_s) * scale
            W_s = np.random.randn(p_s*n_s, rank_s) * scale
            
            # Try alternating minimization first (faster)
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                # Solve for U
                U_new = np.zeros((n_s*m_s, rank_s))
                for i in range(n_s):
                    for j in range(m_s):
                        A = np.zeros(rank_s)
                        for k in range(p_s):
                            A += V_s[j*p_s+k, :] * W_s[k*n_s+i, :]
                        norm_sq = np.sum(A*A)
                        if norm_sq > 1e-9:
                            U_new[i*m_s+j, :] = A / norm_sq
                U_s = U_new
                
                # Solve for V
                V_new = np.zeros((m_s*p_s, rank_s))
                for j in range(m_s):
                    for k in range(p_s):
                        A = np.zeros(rank_s)
                        for i in range(n_s):
                            A += U_s[i*m_s+j, :] * W_s[k*n_s+i, :]
                        norm_sq = np.sum(A*A)
                        if norm_sq > 1e-9:
                            V_new[j*p_s+k, :] = A / norm_sq
                V_s = V_new
                
                # Solve for W
                W_new = np.zeros((p_s*n_s, rank_s))
                for k in range(p_s):
                    for i in range(n_s):
                        A = np.zeros(rank_s)
                        for j in range(m_s):
                            A += U_s[i*m_s+j, :] * V_s[j*p_s+k, :]
                        norm_sq = np.sum(A*A)
                        if norm_sq > 1e-9:
                            W_new[k*n_s+i, :] = A / norm_sq
                W_s = W_new
                
                # Check using einsum
                U_reshaped = U_s.reshape(n_s, m_s, rank_s)
                V_reshaped = V_s.reshape(m_s, p_s, rank_s)
                W_reshaped = W_s.reshape(p_s, n_s, rank_s)
                T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                max_err = np.max(np.abs(T - 1.0))
                
                if max_err < 1e-10:
                    U_snap = snap_to_half_int(U_s.tolist())
                    V_snap = snap_to_half_int(V_s.tolist())
                    W_snap = snap_to_half_int(W_s.tolist())
                    if verify_small(U_snap, V_snap, W_snap, n_s, m_s, p_s):
                        return (U_snap, V_snap, W_snap)
            
            # If alternating minimization didn't converge, try least_squares
            def flatten(Us, Vs, Ws):
                return np.concatenate([Us.flatten(), Vs.flatten(), Ws.flatten()])
            
            def unflatten(x):
                size_U = n_s*m_s*rank_s
                size_V = m_s*p_s*rank_s
                size_W = p_s*n_s*rank_s
                Us = x[:size_U].reshape(n_s*m_s, rank_s)
                Vs = x[size_U:size_U+size_V].reshape(m_s*p_s, rank_s)
                Ws = x[size_U+size_V:size_U+size_V+size_W].reshape(p_s*n_s, rank_s)
                return Us, Vs, Ws
            
            def residuals(x):
                Us, Vs, Ws = unflatten(x)
                U_reshaped = Us.reshape(n_s, m_s, rank_s)
                V_reshaped = Vs.reshape(m_s, p_s, rank_s)
                W_reshaped = Ws.reshape(p_s, n_s, rank_s)
                T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                return (T - 1.0).flatten()
            
            x0 = flatten(U_s, V_s, W_s)
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=5000, xtol=1e-14, ftol=1e-14)
                
                if result.success or result.cost < 1e-12:
                    Us, Vs, Ws = unflatten(result.x)
                    
                    # Check if solution is valid
                    U_reshaped = Us.reshape(n_s, m_s, rank_s)
                    V_reshaped = Vs.reshape(m_s, p_s, rank_s)
                    W_reshaped = Ws.reshape(p_s, n_s, rank_s)
                    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                    max_err = np.max(np.abs(T - 1.0))
                    
                    if max_err < 1e-8:
                        # Snap to half-integers
                        U_snap = snap_to_half_int(Us.tolist())
                        V_snap = snap_to_half_int(Vs.tolist())
                        W_snap = snap_to_half_int(Ws.tolist())
                        
                        # Verify snapped
                        if verify_small(U_snap, V_snap, W_snap, n_s, m_s, p_s):
                            return (U_snap, V_snap, W_snap)
                    
                    # Try without snapping - check if values are close to half-integers
                    all_close = True
                    for arr in [Us, Vs, Ws]:
                        for row in arr:
                            for val in row:
                                if abs(val * 2 - round(val * 2)) > 1e-4:
                                    all_close = False
                                    break
                            if not all_close:
                                break
                        if not all_close:
                            break
                    
                    if all_close:
                        U_snap = snap_to_half_int(Us.tolist())
                        V_snap = snap_to_half_int(Vs.tolist())
                        W_snap = snap_to_half_int(Ws.tolist())
                        if verify_small(U_snap, V_snap, W_snap, n_s, m_s, p_s):
                            return (U_snap, V_snap, W_snap)
                            
            except Exception:
                pass
        
        return None

    def construct_from_small(U_s, V_s, W_s, n_s, m_s, p_s, rank_s, factor):
        """
        Construct a decomposition for n x m x (p_s * factor) from a decomposition 
        for n_s x m_s x p_s.
        Assumes n=n_s, m=m_s, p=p_s*factor.
        """
        if n != n_s or m != m_s:
            return None
        
        p_big = p_s * factor
        rank_big = rank_s * factor
        
        U_big = [[0.0] * rank_big for _ in range(n*m)]
        V_big = [[0.0] * rank_big for _ in range(m*p_big)]
        W_big = [[0.0] * rank_big for _ in range(p_big*n)]
        
        # U_big[i*m+j, factor*c + q] = U_s[i*m+j, c]
        for i in range(n):
            for j in range(m):
                for c in range(rank_s):
                    for q in range(factor):
                        U_big[i*m+j][factor*c + q] = U_s[i*m_s+j][c]
        
        # V_big[j*p_big+k, factor*c + q] = V_s[j*p_s+r, c] if q == k//p_s else 0
        # k = p_s * q + r
        for j in range(m):
            for q in range(factor):
                for r in range(p_s):
                    k = p_s*q + r
                    for c in range(rank_s):
                        V_big[j*p_big + k][factor*c + q] = V_s[j*p_s + r][c]
        
        # W_big[k*n+i, factor*c + q] = W_s[r*n+i, c] if q == k//p_s else 0
        # k = p_s * q + r
        for q in range(factor):
            for r in range(p_s):
                for i in range(n):
                    k = p_s*q + r
                    for c in range(rank_s):
                        W_big[k*n + i][factor*c + q] = W_s[r*n + i][c]
        
        # Verify
        if verify(U_big, V_big, W_big):
            return {"U": U_big, "V": V_big, "W": W_big}
        
        return None

    # Try to find a low rank decomposition for 2x3x3
    n_s, m_s, p_s = 2, 3, 3
    factor = 3
    
    # Try ranks from lower bound upward
    lower_bound = max(n_s*m_s, m_s*p_s, p_s*n_s)  # = 9
    
    for rank_s in range(lower_bound, lower_bound + 10):  # Try 9 through 18
        if time.time() > deadline - 15:
            break
        
        res_small = solve_direct(n_s, m_s, p_s, rank_s, max_attempts=8, max_iters=150)
        if res_small is not None:
            U_s, V_s, W_s = res_small
            res_big = construct_from_small(U_s, V_s, W_s, n_s, m_s, p_s, rank_s, factor)
            if res_big is not None:
                return res_big

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
