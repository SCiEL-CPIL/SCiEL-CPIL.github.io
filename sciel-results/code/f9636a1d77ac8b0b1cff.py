"""Tensor decomposition for 2x2 by 2x5 matrix multiplication."""

import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    n, m, p = DIMENSIONS
    
    # Target Tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a*m + b
                j = b*p + c
                k = c*n + a
                T_target[i,j,k] = 1.0

    # Function to check if U, V, W decompose T_target
    def check_decomposition(U, V, W, tol=1e-9):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        T_calc = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        return np.allclose(T_calc, T_target, atol=tol)

    # Function to snap to integers or half-integers
    def snap_to_half_int(arr):
        # Multiply by 2, round to nearest integer, divide by 2
        return np.round(arr * 2) / 2.0

    # Try to find a decomposition of rank R
    def find_decomposition(rank, max_iters=50, time_limit=10.0):
        # Initialize U, V, W
        # Using random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to avoid scaling issues
        # This is a simple normalization, ALS usually handles it
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        best_error = np.inf
        best_U, best_V, best_W = U, V, W
        
        for it in range(max_iters):
            if time.time() > start_time + time_limit:
                break
                
            # Update U
            # T_target[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # For fixed V, W, we solve for U[i, r]
            # T_target[i, :, :] = sum_r U[i, r] (V[:, r] W[:, r]^T)
            # This is a linear system for each i, r?
            # Actually, ALS for tensor decomposition:
            # U[:, r] = argmin_U || T - sum_r U[:, r] V[:, r] W[:, r]^T ||
            
            # We can vectorize this.
            # Residual R = T_target - sum_{r' != r} U[:, r'] V[:, r'] W[:, r']^T
            # U[:, r] = argmin || R - U[:, r] (V[:, r] W[:, r]^T) ||
            # This is a least squares problem.
            
            # Efficient ALS step:
            # Precompute contractions
            # For each r:
            #   V_r_W_r = np.outer(V[:, r], W[:, r]) # Shape (mp, pn)
            #   Target_slice = T_target - sum_{r' != r} U[:, r'] V_r_W_r'
            #   U[:, r] = argmin || Target_slice - U[:, r] V_r_W_r ||
            #   This is: U[:, r] = (V_r_W_r^T V_r_W_r)^{-1} V_r_W_r^T Target_slice?
            #   No, U is (nm, 1) for fixed r.
            #   Let M_r = V[:, r] W[:, r]^T. Shape (mp, pn).
            #   We want U[:, r] such that U[:, r] M_r approx T_target.
            #   This is not standard least squares because U is on the left.
            #   Actually, T_target is 3D.
            #   T_target[i, j, k] = U[i, r] V[j, r] W[k, r].
            #   For fixed r, we want to find U[:, r] to best fit the residual.
            #   Residual R[i, j, k] = T_target[i, j, k] - sum_{r' != r} U[i, r'] V[j, r'] W[k, r'].
            #   We want U[i, r] to minimize sum_{j,k} (R[i, j, k] - U[i, r] V[j, r] W[k, r])^2.
            #   This is a linear system for U[i, r].
            #   Let w_r = V[:, r] W[:, r]^T. No, V and W are separate.
            #   The term is U[i, r] * (V[:, r] W[:, r]^T)[j, k].
            #   So for each i, we have a vector U[i, r] multiplying the matrix M_r = np.outer(V[:, r], W[:, r]).
            #   We want to fit R[i, :, :] with U[i, r] M_r.
            #   This is a least squares problem: U[i, r] = argmin || R[i] - U[i, r] M_r ||_F.
            #   Solution: U[i, r] = <R[i], M_r> / ||M_r||^2.
            
            # This is very fast!
            
            # Calculate M_r for all r
            # M_r = np.outer(V[:, r], W[:, r])
            # We can compute the numerator and denominator for all i, r at once.
            
            # Residual calculation is O(R * N*M*P).
            # We can update one factor at a time.
            
            # Update U
            for r in range(rank):
                M_r = np.outer(V[:, r], W[:, r])
                # Calculate residual for this r
                # T_calc_part = np.einsum('ir,jr,kr->ijk', U, V, W)
                # This is expensive.
                # Instead, we can compute the contraction of V and W for all r, then subtract.
                pass # Too complex for inline comment, let's just use the direct formula if we can vectorize.
            
            # Alternative: Use a library or a simpler optimization.
            # Given the small size, we can use scipy.optimize.minimize on the flattened parameters.
            # But ALS is faster.
            
            # Let's implement the ALS step properly.
            # T_target shape (4, 10, 10).
            # U (4, R), V (10, R), W (10, R).
            
            # Update U:
            # For each r, M_r = np.outer(V[:, r], W[:, r]) (10x10)
            # We need to compute R = T_target - sum_{r' != r} U[:, r'] V[:, r'] W[:, r']^T
            # This is O(R^2 * 4 * 10 * 10) which is small.
            
            # We can precompute the full tensor T_calc once per iteration?
            # T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            # Residual = T_target - T_calc
            # Then for each r, we need to add back the r-th term to get the residual for r.
            # Residual_r = Residual + U[:, r] V[:, r] W[:, r]^T
            
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            Residual = T_target - T_calc
            
            for r in range(rank):
                Term_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res_r = Residual + Term_r
                M_r = np.outer(V[:, r], W[:, r])
                # U[:, r] = <Res_r, M_r> / ||M_r||^2
                # <Res_r, M_r> is a vector of size n*m
                # Res_r[i, j, k] * M_r[j, k] summed over j, k
                Num = np.einsum('ijk,jk->i', Res_r, M_r)
                Den = np.sum(M_r**2)
                if Den > 1e-12:
                    U[:, r] = Num / Den
            
            # Update V
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            Residual = T_target - T_calc
            for r in range(rank):
                Term_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res_r = Residual + Term_r
                M_r = np.outer(U[:, r], W[:, r]) # Shape (nm, pn) -> (4, 10)
                # V[:, r] = <Res_r, M_r> / ||M_r||^2
                # Res_r[i, j, k] * M_r[i, k] summed over i, k
                Num = np.einsum('ijk,ik->j', Res_r, M_r)
                Den = np.sum(M_r**2)
                if Den > 1e-12:
                    V[:, r] = Num / Den

            # Update W
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            Residual = T_target - T_calc
            for r in range(rank):
                Term_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res_r = Residual + Term_r
                M_r = np.outer(U[:, r], V[:, r]) # Shape (nm, mp) -> (4, 10)
                # W[:, r] = <Res_r, M_r> / ||M_r||^2
                # Res_r[i, j, k] * M_r[i, j] summed over i, j
                Num = np.einsum('ijk,ij->k', Res_r, M_r)
                Den = np.sum(M_r**2)
                if Den > 1e-12:
                    W[:, r] = Num / Den
            
            # Check error
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T_calc - T_target)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < 1e-8:
                break

        # Snap to half-integers
        U_snap = snap_to_half_int(best_U)
        V_snap = snap_to_half_int(best_V)
        W_snap = snap_to_half_int(best_W)
        
        # Verify exactness
        if check_decomposition(U_snap, V_snap, W_snap, tol=1e-9):
            return U_snap, V_snap, W_snap, True
        
        return best_U, best_V, best_W, False

    # Search for lower ranks
    for rank in range(13, 18):
        if time.time() > deadline - margin:
            break
        U, V, W, success = find_decomposition(rank, max_iters=100, time_limit=10.0)
        if success:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to hardcoded rank 18
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(t_offset, k_start):
        U[0][t_offset] = 1
        U[3][t_offset] = 1
        U[2][t_offset+1] = 1
        U[3][t_offset+1] = 1
        U[0][t_offset+2] = 1
        U[3][t_offset+3] = 1
        U[0][t_offset+4] = 1
        U[1][t_offset+4] = 1
        U[2][t_offset+5] = 1
        U[0][t_offset+5] = -1
        U[1][t_offset+6] = 1
        U[3][t_offset+6] = -1
        
        k0 = k_start
        k1 = k_start + 1
        V[k0][t_offset] = 1
        V[p+k1][t_offset] = 1
        V[k0][t_offset+1] = 1
        V[k1][t_offset+2] = 1
        V[p+k1][t_offset+2] = -1
        V[p+k0][t_offset+3] = 1
        V[k0][t_offset+3] = -1
        V[p+k1][t_offset+4] = 1
        V[k0][t_offset+5] = 1
        V[k1][t_offset+5] = 1
        V[p+k0][t_offset+6] = 1
        V[p+k1][t_offset+6] = 1
        
        W[k0*n][t_offset] = 1
        W[k1*n+1][t_offset] = 1
        W[k0*n+1][t_offset+1] = 1
        W[k1*n+1][t_offset+1] = -1
        W[k1*n][t_offset+2] = 1
        W[k1*n+1][t_offset+2] = 1
        W[k0*n][t_offset+3] = 1
        W[k0*n+1][t_offset+3] = 1
        W[k0*n][t_offset+4] = -1
        W[k1*n][t_offset+4] = 1
        W[k1*n+1][t_offset+5] = 1
        W[k0*n][t_offset+6] = 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    
    t = 14
    U[0][t] = 1
    V[0*p + 4][t] = 1
    W[4*n + 0][t] = 1
    t += 1
    U[1][t] = 1
    V[1*p + 4][t] = 1
    W[4*n + 0][t] = 1
    t += 1
    U[2][t] = 1
    V[0*p + 4][t] = 1
    W[4*n + 1][t] = 1
    t += 1
    U[3][t] = 1
    V[1*p + 4][t] = 1
    W[4*n + 1][t] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
