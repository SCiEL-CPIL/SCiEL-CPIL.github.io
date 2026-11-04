import numpy as np
import random
import time
import itertools
from fractions import Fraction
import scipy.optimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = 140  # seconds, leave margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((nm, rank_schoolbook))
    V_schoolbook = np.zeros((mp, rank_schoolbook))
    W_schoolbook = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
                
    # Convert to lists for return format if needed, but numpy arrays are usually fine 
    # unless the evaluator strictly requires lists. The prompt says "factor matrices of shapes...".
    # The seed program returns lists of lists. I should stick to that or ensure compatibility.
    # Let's return lists of lists to be safe, as the seed did.
    
    def to_lists(mat):
        return mat.tolist()
        
    best_rank = rank_schoolbook
    best_decomp = (U_schoolbook, V_schoolbook, W_schoolbook)
    
    # Target tensor T
    # T is a 3D array of shape (nm, mp, pn)
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def decompose_to_tensor(U, V, W):
        """
        U: (nm, r)
        V: (mp, r)
        W: (pn, r)
        Returns T: (nm, mp, pn)
        T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        """
        r = U.shape[1]
        T = np.zeros((nm, mp, pn))
        # Efficient computation: T = np.einsum('ar,br,cr->abc', U, V, W)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return T

    def error_function(params, U_shape, V_shape, W_shape):
        """
        params: flattened array of size nm*r + mp*r + pn*r
        """
        nm_r = U_shape[0] * U_shape[1]
        mp_r = V_shape[0] * V_shape[1]
        pn_r = W_shape[0] * W_shape[1]
        
        U_flat = params[:nm_r]
        V_flat = params[nm_r:nm_r+mp_r]
        W_flat = params[nm_r+mp_r:nm_r+mp_r+pn_r]
        
        U = U_flat.reshape(U_shape)
        V = V_flat.reshape(V_shape)
        W = W_flat.reshape(W_shape)
        
        T_approx = decompose_to_tensor(U, V, W)
        error = np.sum((T_approx - T_target) ** 2)
        return error

    def try_rank(r, max_attempts=20):
        if time.time() - start_time > deadline - 5:
            return None
            
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 5:
                break
                
            # Random initialization
            # Scale matters. Let's try scale 1.
            U_init = np.random.randn(nm, r)
            V_init = np.random.randn(mp, r)
            W_init = np.random.randn(pn, r)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimization
            try:
                result = scipy.optimize.minimize(
                    error_function, 
                    params_init, 
                    args=( (nm, r), (mp, r), (pn, r) ),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9}
                )
            except Exception:
                continue
                
            if result.success or result.fun < 1e-6:
                # Extract U, V, W
                nm_r = nm * r
                mp_r = mp * r
                pn_r = pn * r
                
                U_opt = result.x[:nm_r].reshape(nm, r)
                V_opt = result.x[nm_r:nm_r+mp_r].reshape(mp, r)
                W_opt = result.x[nm_r+mp_r:nm_r+mp_r+pn_r].reshape(pn, r)
                
                # Round to nearest half-integer
                # Multiply by 2, round, divide by 2
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify exactness
                T_check = decompose_to_tensor(U_rounded, V_rounded, W_rounded)
                if np.allclose(T_check, T_target, atol=1e-9):
                    return (U_rounded, V_rounded, W_rounded)
                
                # If not exact, maybe try rounding to integers?
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                T_check_int = decompose_to_tensor(U_int, V_int, W_int)
                if np.allclose(T_check_int, T_target, atol=1e-9):
                    return (U_int, V_int, W_int)
                
        return None

    # Try ranks from 15 up to 17 (known to be 15 or 16, schoolbook 18)
    for r in range(15, rank_schoolbook):
        if time.time() - start_time > deadline - 10:
            break
            
        decomp = try_rank(r, max_attempts=10)
        if decomp is not None:
            best_rank = r
            best_decomp = decomp
            break # Found the lowest rank in this range, stop
            
    # If no lower rank found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": to_lists(U_schoolbook), "V": to_lists(V_schoolbook), "W": to_lists(W_schoolbook)}
    else:
        U, V, W = best_decomp
        return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
# EVOLVE-BLOCK-END
