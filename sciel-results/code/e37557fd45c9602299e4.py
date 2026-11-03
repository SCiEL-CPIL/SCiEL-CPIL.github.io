"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p  # 45
    
    # Known lower bound for 3x3 * 3x5 is at least 27 (from 3x3x3=23, scaling)
    # But we don't know the exact minimal rank. Let's try to find a decomposition
    # with rank less than 45 using numerical optimization.
    
    # The matrix multiplication tensor T has entries T[i,j,k] where:
    # T maps (A, B) -> C = A @ B
    # T_{i,k}^{j,l} = delta_{i,l} * delta_{j,k} ... wait, let me be careful.
    
    # Actually, the tensor for matrix multiplication of (n x m) * (m x p) -> (n x p)
    # is a 4-tensor T_{i,k,j,l} = delta_{i,l} delta_{j,k}
    # where i=0..n-1, k=0..p-1, j=0..m-1, l=0..m-1
    # Wait, let me re-derive.
    
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has indices (i, k) for output, (j, l) for... no.
    # The multiplication tensor is T_{i,k}^{j,l} such that T(A,B)_{i,k} = sum_{j,l} T_{i,k}^{j,l} A_{j,l} B_{l,k}? No.
    
    # Standard: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T_{i,k,j} = delta_{j,j}... no.
    # T_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a trilinear form: sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k} where T_{i,j,k} = delta_{i,i}delta_{k,k}... 
    
    # Let me think of it as a 3-tensor (i, j, k) -> but that's not right either.
    # The multiplication tensor M_{n,m,p} has entries M_{(i,k),(j,l)} where... 
    
    # OK let me use the standard formulation. The matrix multiplication tensor 
    # mu_{n,m,p} is an element of R^{n*p} ⊗ R^{n*m} ⊗ R^{m*p}
    # such that mu_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,k} ⊗ e_{i,j} ⊗ e_{j,k}
    
    # So the tensor has a "slice" for each (i,j,k) which is e_{i,k} in the first factor,
    # e_{i,j} in the second, e_{j,k} in the third.
    
    # A rank-r decomposition is: mu = sum_{t=1}^r a^{(t)} ⊗ b^{(t)} ⊗ c^{(t)}
    # where a^{(t)} in R^{n*p}, b^{(t)} in R^{n*m}, c^{(t)} in R^{m*p}
    
    # The factor matrices are:
    # U: shape (n*m, r) with U[i*m+j, t] = b^{(t)}_{i,j}
    # V: shape (m*p, r) with V[j*p+k, t] = c^{(t)}_{j,k}
    # W: shape (p*n, r) with W[k*n+i, t] = a^{(t)}_{i,k}
    
    # Note: W uses the trace convention W[k*n+i, r] which means the index is k*n+i,
    # so the first index is k (0..p-1) and second is i (0..n-1).
    # So W[k*n+i, t] = a^{(t)}_{i,k}
    
    # The schoolbook decomposition has rank n*m*p with:
    # For each (i,j,k): b^{(t)} = e_{i,j}, c^{(t)} = e_{j,k}, a^{(t)} = e_{i,k}
    
    # Let's set up the tensor T with shape (n*p, n*m, m*p)
    T = np.zeros((n*p, n*m, m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*p+k, i*m+j, j*p+k] = 1
    
    # We want to find rank-r decomposition. Let's try r = 27, 28, ...
    # For each r, we have 3*r*n*p + 3*r*n*m + 3*r*m*p variables... no, 
    # we have r vectors of length n*p, r of length n*m, r of length m*p.
    # Total variables: r*(n*p + n*m + m*p) = r*(15 + 9 + 15) = r*39
    
    # The constraint is that the tensor equals the sum of outer products.
    # This is a system of nonlinear equations.
    
    # Let's try a numerical approach: minimize the Frobenius norm of the residual.
    # residual = T - sum_t outer(a_t, outer(b_t, c_t)) reshaped
    
    # Actually, let's parameterize as:
    # A: shape (n*p, r)
    # B: shape (n*m, r)  
    # C: shape (m*p, r)
    # and the reconstructed tensor is sum_t A[:,t] ⊗ B[:,t] ⊗ C[:,t]
    
    # Let's try r from 27 upward, using random initialization and gradient descent.
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    best_result = None
    best_rank = rank_school
    
    # First, let's verify the schoolbook solution
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly reproduces the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Reconstruct tensor
        T_recon = np.zeros((n*p, n*m, m*p))
        for t in range(rank):
            T_recon += np.einsum('i,j,k->ijk', W[:,t], U[:,t], V[:,t])
        # Compare with original tensor
        residual = np.linalg.norm(T - T_recon)
        return residual < 1e-10
    
    # Schoolbook
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Known: 3x3x3 has rank 23 (Landsberg 1990)
    # For 3x3x5, the rank is unknown but likely between 27 and 45
    # Let's try a few ranks with random restarts
    
    np.random.seed(42)
    
    for r in range(27, rank_school):
        if time.time() > deadline - 10:
            break
        
        success = False
        n_restarts = 5 if r < 35 else 3
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            A = np.random.randn(n*p, r) * 0.1
            B = np.random.randn(n*m, r) * 0.1
            C = np.random.randn(m*p, r) * 0.1
            
            # Normalize columns to help with convergence
            for t in range(r):
                A[:,t] /= (np.linalg.norm(A[:,t]) + 1e-10)
                B[:,t] /= (np.linalg.norm(B[:,t]) + 1e-10)
                C[:,t] /= (np.linalg.norm(C[:,t]) + 1e-10)
            
            # Objective: ||T - sum_t A_t ⊗ B_t ⊗ C_t||_F^2
            def objective(params):
                A_flat = params[0:n*p*r].reshape(n*p, r)
                B_flat = params[n*p*r:n*p*r+n*m*r].reshape(n*m, r)
                C_flat = params[n*p*r+n*m*r:].reshape(m*p, r)
                
                T_recon = np.zeros((n*p, n*m, m*p))
                for t in range(r):
                    T_recon += np.einsum('i,j,k->ijk', C_flat[:,t], A_flat[:,t], B_flat[:,t])
                
                return np.sum((T - T_recon)**2)
            
            # Gradient
            def gradient(params):
                A_flat = params[0:n*p*r].reshape(n*p, r)
                B_flat = params[n*p*r:n*p*r+n*m*r].reshape(n*m, r)
                C_flat = params[n*p*r+n*m*r:].reshape(m*p, r)
                
                grad_A = np.zeros_like(A_flat)
                grad_B = np.zeros_like(B_flat)
                grad_C = np.zeros_like(C_flat)
                
                for t in range(r):
                    # d/dA_{i,t} = 2 * sum_{j,k} (T_{i,j,k} - recon_{i,j,k}) * B_{j,t} * C_{k,t}
                    # Wait, let me be more careful.
                    # T_recon_{i,j,k} = sum_t A_{i,t} B_{j,t} C_{k,t}
                    # d(T_recon_{i,j,k})/dA_{i',t'} = delta_{i,i'} delta_{t,t'} B_{j,t'} C_{k,t'}
                    # So grad_A_{i',t'} = 2 * sum_{j,k} (T_{i',j,k} - T_recon_{i',j,k}) B_{j,t'} C_{k,t'}
                    
                    for i in range(n*p):
                        for j in range(n*m):
                            for k in range(m*p):
                                residual_ijk = T[i,j,k] - sum(A_flat[i,tt]*B_flat[j,tt]*C_flat[k,tt] for tt in range(r))
                                grad_A[i,t] += 2 * residual_ijk * B_flat[j,t] * C_flat[k,t]
                                grad_B[j,t] += 2 * residual_ijk * A_flat[i,t] * C_flat[k,t]
                                grad_C[k,t] += 2 * residual_ijk * A_flat[i,t] * B_flat[j,t]
                
                return np.concatenate([grad_A.flatten(), grad_B.flatten(), grad_C.flatten()])
            
            # This gradient is too slow. Let's use a vectorized version.
            def objective_vec(params):
                A_flat = params[0:n*p*r].reshape(n*p, r)
                B_flat = params[n*p*r:n*p*r+n*m*r].reshape(n*m, r)
                C_flat = params[n*p*r+n*m*r:].reshape(m*p, r)
                
                # T_recon_{i,j,k} = sum_t A_{i,t} B_{j,t} C_{k,t}
                # = einsum('it,jt,kt->ijk', A, B, C)
                T_recon = np.einsum('it,jt,kt->ijk', A_flat, B_flat, C_flat)
                return np.sum((T - T_recon)**2)
            
            def gradient_vec(params):
                A_flat = params[0:n*p*r].reshape(n*p, r)
                B_flat = params[n*p*r:n*p*r+n*m*r].reshape(n*m, r)
                C_flat = params[n*p*r+n*m*r:].reshape(m*p, r)
                
                T_recon = np.einsum('it,jt,kt->ijk', A_flat, B_flat, C_flat)
                R = T - T_recon  # residual
                
                # grad_A_{i,t} = 2 * sum_{j,k} R_{i,j,k} B_{j,t} C_{k,t}
                # = 2 * einsum('ijk,jt,kt->it', R, B_flat, C_flat)
                grad_A = 2 * np.einsum('ijk,jt,kt->it', R, B_flat, C_flat)
                grad_B = 2 * np.einsum('ijk,it,kt->jt', R, A_flat, C_flat)
                grad_C = 2 * np.einsum('ijk,it,jt->kt', R, A_flat, B_flat)
                
                return np.concatenate([grad_A.flatten(), grad_B.flatten(), grad_C.flatten()])
            
            x0 = np.concatenate([A.flatten(), B.flatten(), C.flatten()])
            
            try:
                result = minimize(objective_vec, x0, jac=gradient_vec, method='L-BFGS-B', 
                                  options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
                
                if result.fun < 1e-6:
                    A_opt = result.x[0:n*p*r].reshape(n*p, r)
                    B_opt = result.x[n*p*r:n*p*r+n*m*r].reshape(n*m, r)
                    C_opt = result.x[n*p*r+n*m*r:].reshape(m*p, r)
                    
                    # Check exactness
                    T_recon = np.einsum('it,jt,kt->ijk', A_opt, B_opt, C_opt)
                    if np.linalg.norm(T - T_recon) < 1e-8:
                        # Found a valid decomposition!
                        # Convert to half-integer if possible
                        U = B_opt  # U[i*m+j, t] = B_{i,j,t}... wait
                        # B is shape (n*m, r), so B[i*m+j, t] is the (i,j) entry of b^{(t)}
                        # V is shape (m*p, r), so V[j*p+k, t] is the (j,k) entry of c^{(t)}
                        # W is shape (p*n, r), so W[k*n+i, t] is the (i,k) entry of a^{(t)}
                        # But C is shape (m*p, r), so C[j*p+k, t]
                        # And A is shape (n*p, r), so A[i*p+k, t]
                        # We need W[k*n+i, t] = A[i*p+k, t]
                        
                        W_opt = np.zeros((p*n, r))
                        for i in range(n):
                            for k in range(p):
                                W_opt[k*n+i, :] = A[i*p+k, :]
                        
                        U_list = B_opt.tolist()
                        V_list = C_opt.tolist()
                        W_list = W_opt.tolist()
                        
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r
                        success = True
                        break
                        
            except Exception as e:
                pass
            
            if success:
                break
    
    # If we found a better rank, return it; otherwise return schoolbook
    return best_result
# EVOLVE-BLOCK-END
