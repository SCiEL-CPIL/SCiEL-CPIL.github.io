"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication.
    Tries to find rank < 36 via optimization, falls back to rank 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    _t0 = time.time()
    _DEADLINE = 115  # seconds (safety margin)
    
    # Build the multiplication tensor T[i,j,k] = 1 if this term contributes to C[i,k]
    # T has shape (n, m, p) = (2, 4, 5)
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] for all i,j,k
    # where W is indexed as W[k*n+i, r] per the convention
    
    def build_tensor():
        """Return the multiplication tensor as a 3D array T[i,j,k]."""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T
    
    T = build_tensor()
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W correctly decompose the tensor."""
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
                    result[i, j, k] = s
        return np.allclose(result, T, atol=tol)
    
    def make_strassen_decomposition():
        """Create the known rank-36 Strassen-based decomposition."""
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
            for (i, j), val in u_coeffs.items():
                U[i*m + j, idx] += val
            for (j, k), val in v_coeffs.items():
                V[j*p + k, idx] += val
            for (k, i), val in w_coeffs.items():
                W[k*n + i, idx] += val

        def strassen_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            add_term(idx_start, 
                     {(0, a_row_offset): 1, (1, a_row_offset+1): 1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                     {(c_col_offset, 0): 1, (c_col_offset+1, 1): 1})
            add_term(idx_start+1,
                     {(1, a_row_offset): 1, (1, a_row_offset+1): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1, (c_col_offset+1, 1): -1})
            add_term(idx_start+2,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset+1): 1, (b_row_offset+1, c_col_offset+1): -1},
                     {(c_col_offset+1, 0): 1, (c_col_offset+1, 1): 1})
            add_term(idx_start+3,
                     {(1, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},
                     {(c_col_offset, 0): 1, (c_col_offset, 1): 1})
            add_term(idx_start+4,
                     {(0, a_row_offset): 1, (0, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset+1): 1},
                     {(c_col_offset, 0): -1, (c_col_offset+1, 0): 1})
            add_term(idx_start+5,
                     {(1, a_row_offset): 1, (0, a_row_offset): -1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset+1): 1},
                     {(c_col_offset+1, 1): 1})
            add_term(idx_start+6,
                     {(0, a_row_offset+1): 1, (1, a_row_offset+1): -1},
                     {(b_row_offset+1, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                     {(c_col_offset, 0): 1})

        def schoolbook_2x2x1_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            add_term(idx_start,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start+1,
                     {(0, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start+2,
                     {(1, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1})
            add_term(idx_start+3,
                     {(1, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset): 1},
                     {(c_col_offset, 1): 1})

        idx = 0
        strassen_block(0, 0, 0, idx); idx += 7
        strassen_block(0, 0, 2, idx); idx += 7
        schoolbook_2x2x1_block(0, 0, 4, idx); idx += 4
        strassen_block(2, 2, 0, idx); idx += 7
        strassen_block(2, 2, 2, idx); idx += 7
        schoolbook_2x2x1_block(2, 2, 4, idx); idx += 4
        
        return U, V, W, rank
    
    U0, V0, W0, rank0 = make_strassen_decomposition()
    
    # Try to reduce rank via optimization
    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0
    
    def try_reduce_rank(start_rank, max_attempts=10):
        """Try to find a decomposition of lower rank via optimization."""
        if time.time() - _t0 > _DEADLINE - 10:
            return None
        
        for target_rank in range(start_rank - 1, 20, -1):
            if time.time() - _t0 > _DEADLINE - 10:
                break
            
            success = False
            for attempt in range(max_attempts):
                if time.time() - _t0 > _DEADLINE - 10:
                    break
                
                # Initialize with random perturbation of the known solution
                # Project the rank-36 solution to rank-target by SVD truncation
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                
                # Use the first target_rank terms as initialization
                U_init[:, :target_rank] = U0[:, :target_rank]
                V_init[:, :target_rank] = V0[:, :target_rank]
                W_init[:, :target_rank] = W0[:, :target_rank]
                
                # Add small random perturbation
                noise_scale = 0.1
                U_init += noise_scale * np.random.randn(n*m, target_rank)
                V_init += noise_scale * np.random.randn(m*p, target_rank)
                W_init += noise_scale * np.random.randn(p*n, target_rank)
                
                # Optimize using scipy
                def objective(params):
                    r = target_rank
                    U_p = params[:n*m*r].reshape(n*m, r)
                    V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    U3 = U_p.reshape(n, m, r)
                    V3 = V_p.reshape(m, p, r)
                    W3 = W_p.reshape(p, n, r)
                    
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for rr in range(r):
                                    s += U3[i, j, rr] * V3[j, k, rr] * W3[k, i, rr]
                                err += (s - 1.0) ** 2
                    return err
                
                params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                try:
                    result = scipy.optimize.minimize(
                        objective, params0, method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if result.fun < 1e-6:
                        # Success! Verify
                        r = target_rank
                        U_sol = result.x[:n*m*r].reshape(n*m, r)
                        V_sol = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                        W_sol = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                        if verify_decomposition(U_sol, V_sol, W_sol, tol=1e-6):
                            return (U_sol, V_sol, W_sol, target_rank)
                except Exception:
                    pass
            
            if not success and target_rank < start_rank - 2:
                # If we can't reduce by more than 2, stop trying lower
                break
    
    # Try to reduce from rank 36
    result = try_reduce_rank(rank0, max_attempts=5)
    if result is not None:
        U, V, W, rank = result
        best_U, best_V, best_W, best_rank = U, V, W, rank
    else:
        U, V, W, rank = U0, V0, W0, rank0
    
    # Deadline guard
    if time.time() - _t0 > _DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
