# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 110.0

def construct():
    n, m, p = DIMENSIONS
    
    # ... (Rank 84 construction code same as above) ...
    
    # ... (Verification of Rank 84) ...
    
    # Attempt Rank 83 via ALS
    def try_als_reduce(rank_target):
        if time.time() > _DEADLINE - 20:
            return None
            
        size_U, size_V, size_W = n*m, m*p, p*n
        
        # Try dropping different columns
        for drop_idx in range(rank_base):
            if time.time() > _DEADLINE - 5:
                return None
                
            # Init U, V, W with rank_target
            # Take all columns except drop_idx
            mask = [i for i in range(rank_base) if i != drop_idx]
            U_init = U_np[:, mask].copy()
            V_init = V_np[:, mask].copy()
            W_init = W_np[:, mask].copy()
            
            # Add small noise to avoid symmetry issues
            U_init += np.random.normal(0, 1e-4, U_init.shape)
            V_init += np.random.normal(0, 1e-4, V_init.shape)
            W_init += np.random.normal(0, 1e-4, W_init.shape)
            
            # ALS Loop
            U, V, W = U_init, V_init, W_init
            prev_err = float('inf')
            
            for _ in range(20): # Max 20 iterations
                if time.time() > _DEADLINE - 2:
                    return None
                    
                # Solve for U
                # For each i, j: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
                # This is linear in U if V, W fixed?
                # No, it's a system of linear equations for U.
                # Let A_c = V[:, c] * W[:, c] (element-wise product of vectors of size m*p and p*n? No.)
                # V is (m*p, r), W is (p*n, r).
                # For a fixed c, the term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
                # For fixed i, j, k, this is linear in U[i*m+j, c].
                # We can set up a linear system for U.
                
                # Construct matrix M_U of size (n*m*p) x (n*m*r) ? No.
                # Variables: U has n*m*r variables.
                # Equations: n*m*p equations.
                # For each (i,j,k), equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
                
                # This is a sparse linear system.
                # Let's use numpy.linalg.lstsq on a flattened system.
                
                # Build A_U matrix
                # Rows: n*m*p. Cols: n*m*r.
                # This is large: 270 x (15*83) = 270 x 1245.
                # This is small enough for lstsq.
                
                A_U = np.zeros((n*m*p, size_U * rank_target))
                b_U = np.ones(n*m*p)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank_target):
                                A_U[idx, (i*m+j)*rank_target + c] = V[j*p+k, c] * W[k*n+i, c]
                            idx += 1
                
                # Solve A_U * u_flat = b_U
                u_flat, residuals, rank, sv = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = u_flat.reshape((size_U, rank_target))
                
                # Solve for V
                A_V = np.zeros((n*m*p, size_V * rank_target))
                b_V = np.ones(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank_target):
                                A_V[idx, (j*p+k)*rank_target + c] = U[i*m+j, c] * W[k*n+i, c]
                            idx += 1
                v_flat, residuals, rank, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
                V = v_flat.reshape((size_V, rank_target))
                
                # Solve for W
                A_W = np.zeros((n*m*p, size_W * rank_target))
                b_W = np.ones(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank_target):
                                A_W[idx, (k*n+i)*rank_target + c] = U[i*m+j, c] * V[j*p+k, c]
                            idx += 1
                w_flat, residuals, rank, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = w_flat.reshape((size_W, rank_target))
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                            
                if err < 1e-10:
                    # Found a solution!
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
                if err > prev_err * 1.01:
                    break
                prev_err = err
                
        return None

    # Try to reduce to 83
    result_83 = try_als_reduce(83)
    if result_83 is not None:
        return result_83
        
    return {"U": U_base, "V": V_base, "W": W_base}
# EVOLVE-BLOCK-END
