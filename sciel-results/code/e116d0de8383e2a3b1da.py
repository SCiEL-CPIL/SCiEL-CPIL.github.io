"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    def _verify_fast(U, V, W):
        """Faster verification using einsum"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # We need: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _snap_to_half_int(arr):
        """Snap values to nearest half-integer"""
        return np.round(2 * arr) / 2.0

    def _try_gradient_search(rank):
        """
        Try to find U, V, W of given rank using gradient descent on the squared error.
        """
        if rank >= n * m * p:
            return None

        # Initialize U, V, W with small random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1

        # Precompute indices for faster access
        # We need to evaluate E = sum_{i,j,k} (sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} - 1)^2
        # This is a high-dimensional non-convex optimization.
        # Simple gradient descent might be slow. Let's try a few steps.
        
        # To speed up, we can use the fact that the tensor is small.
        # n=2, m=3, p=4.
        # Total equations: 24.
        
        # Let's use a simple iterative solver:
        # Fix U, V, solve for W? No, it's trilinear.
        # We can use alternating minimization:
        # 1. Fix U, V. The equation is linear in W?
        #    sum_r W_{k,i,r} (U_{i,j,r} V_{j,k,r}) = 1
        #    For a fixed (k,i), the term is sum_r W_{k,i,r} A_{k,i,r} = 1, where A_{k,i,r} = sum_j U_{i,j,r} V_{j,k,r}?
        #    No, the sum is over r.
        #    For each (i,j,k), we have sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1.
        #    This is not linear in W alone because U and V are coupled in the product.
        #    However, if we fix U and V, then for each (k,i), we have a system of linear equations in W_{k,i,r}?
        #    No, W_{k,i,r} appears in multiple equations (for different j).
        #    Equation for (i,j,k): sum_r (U_{i,j,r} V_{j,k,r}) W_{k,i,r} = 1.
        #    Let C_{j, k, i, r} = U_{i,j,r} V_{j,k,r}.
        #    Then sum_r C_{j, k, i, r} W_{k,i,r} = 1.
        #    This is a linear system in W for fixed U, V.
        #    The unknowns are W_{k,i,r} for all k,i,r.
        #    The equations are indexed by (i,j,k).
        #    There are n*m*p = 24 equations.
        #    There are p*n*rank unknowns.
        #    If rank is small, we might have more unknowns than equations, so we can find a solution that minimizes error.
        
        for iteration in range(100):
            if time.time() - start_time > time_limit:
                return None
            
            # 1. Fix U, V, solve for W
            # Build linear system A_W x_W = b_W
            # x_W is flattened W of size p*n*rank
            # A_W has shape (n*m*p, p*n*rank)
            # b_W is ones of size n*m*p
            
            A_W = np.zeros((n * m * p, p * n * rank))
            b_W = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            # Coefficient of W[k*n+i, r] in equation (i,j,k)
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            unk_idx = (k * n + i) * rank + r
                            A_W[eq_idx, unk_idx] = coeff
                        eq_idx += 1
            
            # Solve least squares for W
            W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_flat.reshape(p * n, rank)
            
            # 2. Fix V, W, solve for U
            A_U = np.zeros((n * m * p, n * m * rank))
            b_U = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            # Coefficient of U[i*m+j, r] in equation (i,j,k)
                            coeff = V[j * p + k, r] * W[k * n + i, r]
                            unk_idx = (i * m + j) * rank + r
                            A_U[eq_idx, unk_idx] = coeff
                        eq_idx += 1
            
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_flat.reshape(n * m, rank)
            
            # 3. Fix U, W, solve for V
            A_V = np.zeros((n * m * p, m * p * rank))
            b_V = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            # Coefficient of V[j*p+k, r] in equation (i,j,k)
                            coeff = U[i * m + j, r] * W[k * n + i, r]
                            unk_idx = (j * p + k) * rank + r
                            A_V[eq_idx, unk_idx] = coeff
                        eq_idx += 1
            
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_flat.reshape(m * p, rank)
            
            # Check convergence
            if iteration % 10 == 0:
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                            error += (val - 1.0) ** 2
                if error < 1e-12:
                    break
        
        # Verify the solution
        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
            # Snap to half-integers
            U_snap = _snap_to_half_int(U)
            V_snap = _snap_to_half_int(V)
            W_snap = _snap_to_half_int(W)
            
            # Re-verify after snapping
            if _verify_fast(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        return None

    best_rank = rank_schoolbook
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Start from a reasonable lower bound. 
    # For 2x3x4, schoolbook is 24.
    # Try 20, 21, 22, 23.
    for r in range(20, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        result = _try_gradient_search(r)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If we found a better solution, return it
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        U = [[float(x) for x in row] for row in U]
        V = [[float(x) for x in row] for row in V]
        W = [[float(x) for x in row] for row in W]
        return {"U": U, "V": V, "W": W}
    
    # Otherwise return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
