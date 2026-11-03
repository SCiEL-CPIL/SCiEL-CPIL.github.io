"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 100
    
    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, get the schoolbook as a fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    def _compute_error(U, V, W):
        """Compute Frobenius norm squared of T - reconstruction."""
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        diff = T - T_recon
        return np.sum(diff * diff)
    
    def _als_optimize(U, V, W, max_iters=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Fix V and W, solve for U
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For each i: T[i,:,:] = sum_c U[i,c] * (V[:,c] * W[:,c])
            # This is: T_i = U[i,:] @ M_i where M_i[j,k] = sum_c V[j,c]*W[k,c]
            # Actually: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # So for fixed V,W: T[:,j,k] as function of i is linear in U[i,:]
            # T[i,j,k] = sum_c U[i,c] * (V[j,c] * W[k,c])
            # Let M_c[j,k] = V[j,c]*W[k,c], then T[i,j,k] = sum_c U[i,c]*M_c[j,k]
            # So T[:,:,j,k] = U @ (V[:,c]*W[:,c]) for each c... 
            # Better: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # = sum_c U[i,c] * (V[j,c] * W[k,c])
            # For each i: T[i,:,:] = sum_c U[i,c] * (V[:,c] * W[:,c])
            # Let A_c = V[:,c] * W[:,c] (element-wise product), shape (mp, pn)
            # Then T[i,:,:] = sum_c U[i,c] * A_c
            # This is a linear system: T_i = U[i,:] @ A_stack
            # where A_stack has columns A_c flattened
            
            # More efficient: 
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed V,W: define B[j,k,c] = V[j,c]*W[k,c]
            # Then T[i,j,k] = sum_c U[i,c]*B[j,k,c]
            # So T[i] (vectorized) = U[i,:] @ B_vec where B_vec is (mp*pn, rank)
            # B_vec[j*pn+k, c] = V[j,c]*W[k,c]
            
            B_mat = np.einsum('jc,kc->jkc', V, W).reshape(mp * pn, rank)
            T_flat = T.reshape(nm, mp * pn)
            # Solve: T_flat[i,:] = U[i,:] @ B_mat
            # U[i,:] = T_flat[i,:] @ pinv(B_mat)
            try:
                U_new, residuals, rank_b, sv = np.linalg.lstsq(B_mat, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Fix U and W, solve for V
            C_mat = np.einsum('ic,kc->ikc', U, W).reshape(nm * pn, rank)
            T_flat2 = T.reshape(nm, mp, pn).transpose(0, 2, 1).reshape(nm * pn, mp)
            try:
                V_new, residuals, rank_c, sv = np.linalg.lstsq(C_mat, T_flat2.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Fix U and V, solve for W
            D_mat = np.einsum('ic,jc->ijc', U, V).reshape(nm * mp, rank)
            T_flat3 = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W_new, residuals, rank_d, sv = np.linalg.lstsq(D_mat, T_flat3.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Check convergence
            err = _compute_error(U, V, W)
            if err < tol:
                break
        
        return U, V, W, _compute_error(U, V, W)
    
    # Try structured initializations based on Kronecker products
    # For matrix multiplication, we can use block structures
    
    def _kronecker_init(rank, seed=42):
        """Initialize using Kronecker product structure."""
        np.random.seed(seed)
        
        # Use random but structured initialization
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        return U, V, W
    
    # Try ranks from 70 down to 40
    for rank in range(70, 39, -2):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            U, V, W = _kronecker_init(rank, seed)
            
            # Run ALS
            U, V, W, err = _als_optimize(U, V, W, max_iters=100)
            
            if err < 1e-8:
                # Polish with tighter tolerance
                U, V, W, err = _als_optimize(U, V, W, max_iters=50, tol=1e-12)
                
                if err < 1e-10:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
    
    # If we found a better result, use it; otherwise use schoolbook
    if best_result is not None:
        U_final, V_final, W_final = best_result
    else:
        U_final, V_final, W_final = U_school, V_school, W_school
    
    # Convert to lists for the required output format
    U_out = [[float(U_final[i][j]) for j in range(U_final.shape[1])] for i in range(U_final.shape[0])]
    V_out = [[float(V_final[i][j]) for j in range(V_final.shape[1])] for i in range(V_final.shape[0])]
    W_out = [[float(W_final[i][j]) for j in range(W_final.shape[1])] for i in range(W_final.shape[0])]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
