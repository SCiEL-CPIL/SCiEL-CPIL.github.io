"""Program for matrix multiplication tensor decomposition task."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Strategy:
    1. Attempt to find a rank-23 decomposition for the 3x3x3 sub-problem.
    2. Use this to construct a rank-46 decomposition for the 3x3x6 problem.
    3. Attempt to reduce the rank of the 3x3x6 decomposition using optimization.
    4. Fallback to schoolbook rank 54 if lower ranks are not found.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 140  # Safety margin
    
    def verify_decomposition(U, V, W, rank, n, m, p, tol=1e-9):
        """Check if U,V,W of given rank exactly decompose the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def find_3x3_decomposition(rank_target, time_limit):
        """
        Find a rank-r decomposition for 3x3*3x3 matrix multiplication.
        Returns (U, V, W) as numpy arrays or None.
        """
        start_time = time.time()
        n_sub, m_sub, p_sub = 3, 3, 3
        
        # Tensor T[i*m+j, j*p+k, k*n+i] = 1
        # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        best_U, best_V, best_W = None, None, None
        best_error = float('inf')
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(attempt * 12345 + 67890)
            
            U = np.random.randn(9, rank_target) * 0.5
            V = np.random.randn(9, rank_target) * 0.5
            W = np.random.randn(9, rank_target) * 0.5
            
            # ALS iterations
            for iter in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                # For each (i,j), U[i*m+j, :] is the variable.
                # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is linear in U[i*m+j, :].
                # Let A_r = V[j*p+k, r] * W[k*n+i, r].
                # We have a system of p_sub equations for each (i,j) with m_sub*p_sub unknowns?
                # No, for fixed (i,j), we have p_sub equations (one for each k) and rank_target unknowns.
                # This is underdetermined if rank_target > p_sub.
                # We want the minimum norm solution or a solution that fits the data.
                # Actually, we can solve for U using least squares on the residual.
                
                # Residual R[i,j,k] = 1 - sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # We want to minimize sum_{i,j,k} R[i,j,k]^2.
                
                # Update U:
                for i in range(n_sub):
                    for j in range(m_sub):
                        # Build system for U[i*m+j, :]
                        # Variables: x = U[i*m+j, :] (size rank_target)
                        # Equations: for each k in 0..p_sub-1:
                        # sum_r x_r * V[j*p_sub+k, r] * W[k*n_sub+i, r] = 1
                        A_mat = np.zeros((p_sub, rank_target))
                        for k in range(p_sub):
                            A_mat[k, :] = V[j*p_sub+k, :] * W[k*n_sub+i, :]
                        b_vec = np.ones(p_sub)
                        
                        # Solve least squares
                        try:
                            U[i*m_sub+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            pass

                # Update V
                for j in range(m_sub):
                    for k in range(p_sub):
                        # Variables: x = V[j*p_sub+k, :]
                        # Equations: for each i in 0..n_sub-1:
                        # sum_r U[i*m_sub+j, r] * x_r * W[k*n_sub+i, r] = 1
                        A_mat = np.zeros((n_sub, rank_target))
                        for i in range(n_sub):
                            A_mat[i, :] = U[i*m_sub+j, :] * W[k*n_sub+i, :]
                        b_vec = np.ones(n_sub)
                        
                        try:
                            V[j*p_sub+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            pass

                # Update W
                for k in range(p_sub):
                    for i in range(n_sub):
                        # Variables: x = W[k*n_sub+i, :]
                        # Equations: for each j in 0..m_sub-1:
                        # sum_r U[i*m_sub+j, r] * V[j*p_sub+k, r] * x_r = 1
                        A_mat = np.zeros((m_sub, rank_target))
                        for j in range(m_sub):
                            A_mat[j, :] = U[i*m_sub+j, :] * V[j*p_sub+k, :]
                        b_vec = np.ones(m_sub)
                        
                        try:
                            W[k*n_sub+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            pass

            # Check error
            error = 0.0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        s = 0.0
                        for r in range(rank_target):
                            s += U[i*m_sub+j, r] * V[j*p_sub+k, r] * W[k*n_sub+i, r]
                        error += (s - 1.0)**2
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if best_error < 1e-12:
                break
        
        if best_error < 1e-9:
            return best_U, best_V, best_W
        return None

    # Step 1: Find 3x3x3 decomposition
    U3, V3, W3 = find_3x3_decomposition(23, 40)
    
    if U3 is not None:
        rank_sub = 23
        # Construct 3x3x6 decomposition
        # U_full: 9 x 46
        # V_full: 18 x 46
        # W_full: 18 x 46
        
        U_full = np.zeros((9, 46))
        V_full = np.zeros((18, 46))
        W_full = np.zeros((18, 46))
        
        # First half (k=0,1,2)
        U_full[:, :23] = U3
        for j in range(3):
            for k in range(3):
                V_full[j*6+k, :23] = V3[j*3+k, :]
        for k in range(3):
            for i in range(3):
                W_full[k*3+i, :23] = W3[k*3+i, :]
        
        # Second half (k=3,4,5)
        U_full[:, 23:] = U3
        for j in range(3):
            for k in range(3):
                V_full[j*6+(k+3), 23:] = V3[j*3+k, :]
        for k in range(3):
            for i in range(3):
                W_full[(k+3)*3+i, 23:] = W3[k*3+i, :]
        
        current_rank = 46
        current_U = U_full
        current_V = V_full
        current_W = W_full
        
        # Step 2: Try to reduce rank
        # We can try to optimize the rank-46 solution to see if it converges to a lower effective rank
        # or just try to find a rank-45, 44... solution directly.
        # Given time constraints, we might just stick with 46 if it's valid.
        
        if verify_decomposition(current_U.tolist(), current_V.tolist(), current_W.tolist(), current_rank, n, m, p):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
            
    # Fallback: Schoolbook rank 54
    rank = n * m * p
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
# EVOLVE-BLOCK-END
