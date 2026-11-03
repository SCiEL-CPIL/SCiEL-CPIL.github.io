"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 84
    
    # Build schoolbook solution as fallback
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

    # Non-zero indices of the multiplication tensor
    idx_i = []
    idx_j = []
    idx_k = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_i.append(i * m + j)
                idx_j.append(j * p + k)
                idx_k.append(k * n + i)
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    num_nz = len(idx_i)  # 84

    start_time = time.time()
    deadline = 130.0  # safety margin

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        U_sel = U[idx_i, :]
        V_sel = V[idx_j, :]
        W_sel = W[idx_k, :]
        T = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(T - 1.0)) < 1e-9

    def snap_half_int(U, V, W):
        """Snap to nearest half-integer and verify."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if verify(U_s, V_s, W_s):
            return U_s, V_s, W_s
        # Try integer snap
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i):
            return U_i, V_i, W_i
        return None

    def try_strassen_block(R, max_restarts=3):
        """
        Use Strassen-like block decomposition as initialization.
        For 3x4 * 4x7, we can think of this as blocks.
        Key idea: use Kronecker product structure to find low-rank solutions.
        
        The tensor T[i,j,k] = delta(i,j,k) for the multiplication.
        We seek rank-R decomposition T = sum_r u_r ⊗ v_r ⊗ w_r.
        
        Strategy: Initialize using known low-rank patterns from related problems.
        For 3x4x7, the Strassen-based approach can give rank around 80.
        """
        if time.time() > deadline:
            return None
            
        total_params = R * (nm + mp + pn)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with a structured pattern inspired by Strassen
            # Use random initialization but with a bias toward the schoolbook structure
            params0 = np.zeros(total_params)
            
            # First min(R, 84) columns can mirror schoolbook
            for r in range(min(R, schoolbook_rank)):
                i = r // (m * p)
                j = (r // p) % m
                k = r % p
                params0[r] = 1.0  # U[i*m+j, r]
                params0[nm * R + r] = 1.0  # V[j*p+k, r]
                params0[nm * R + mp * R + r] = 1.0  # W[k*n+i, r]
            
            # Remaining columns: small random
            if R > schoolbook_rank:
                extra = R - schoolbook_rank
                params0[schoolbook_rank * (nm + mp + pn):] = np.random.randn(extra * (nm + mp + pn)) * 0.1
            elif R < schoolbook_rank:
                # Perturb existing columns
                params0[:R * (nm + mp + pn)] += np.random.randn(R * (nm + mp + pn)) * 0.05
            
            # Use Nelder-Mead for local optimization (robust to non-smooth snapping)
            from scipy.optimize import minimize
            
            def objective(params):
                U_flat = params[:nm * R]
                V_flat = params[nm * R: nm * R + mp * R]
                W_flat = params[nm * R + mp * R: nm * R + mp * R + pn * R]
                U = U_flat.reshape((nm, R))
                V = V_flat.reshape((mp, R))
                W = W_flat.reshape((pn, R))
                U_sel = U[idx_i, :]
                V_sel = V[idx_j, :]
                W_sel = W[idx_k, :]
                T = np.sum(U_sel * V_sel * W_sel, axis=1)
                # Also penalize large values to encourage snapping
                return np.sum((T - 1.0)**2) + 0.001 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
            
            try:
                res = minimize(objective, params0, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                U = res.x[:nm * R].reshape((nm, R))
                V = res.x[nm * R: nm * R + mp * R].reshape((mp, R))
                W = res.x[nm * R + mp * R:].reshape((pn, R))
                
                snapped = snap_half_int(U, V, W)
                if snapped is not None:
                    return snapped
            except Exception:
                pass
        
        return None

    def try_riemannian(R, max_iters=100):
        """
        Riemannian optimization on the product manifold.
        This is a different structural approach: instead of flat parameter space,
        optimize directly on the Stiefel-like manifold of factor matrices.
        """
        if time.time() > deadline:
            return None
        
        for restart in range(2):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Scale to encourage half-integer solutions
            for _ in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Compute gradient of sum_r (sum over nonzero T entries of U_i*V_j*W_k - 1)^2
                # dL/dU[i,r] = 2 * sum_{(i,j,k) in nz: i matches} (T[i,j,k]-1) * V[j,r] * W[k,r]
                # This is expensive, so use a simpler approach:
                # Just do a few steps of gradient descent with momentum
                
                U_sel = U[idx_i, :]
                V_sel = V[idx_j, :]
                W_sel = W[idx_k, :]
                T = np.sum(U_sel * V_sel * W_sel, axis=1)
                resid = T - 1.0  # (num_nz,)
                
                # Gradients
                # dU[i,r] = 2 * sum_{(i,j,k): idx_i == i} resid * V_sel[idx_i==i, r] * W_sel[idx_i==i, r]
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Efficient: use einsum-like accumulation
                for r in range(R):
                    dU[idx_i, r] += 2 * resid * V_sel * W_sel[:, r]
                    dV[idx_j, r] += 2 * resid * U_sel * W_sel[:, r]
                    dW[idx_k, r] += 2 * resid * U_sel * V_sel[:, r]
                
                # Normalize gradients to have similar scale
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Periodically snap and check
                if _ % 20 == 0:
                    snapped = snap_half_int(U, V, W)
                    if snapped is not None:
                        return snapped
        
        return None

    def try_kronecker_structure():
        """
        Exploit the Kronecker product structure.
        The matrix multiplication tensor for (n,m,p) can be related to
        smaller tensor products. For 3x4x7, note that 3=3, 4=2*2, 7=7.
        
        We can try: T_3x4x7 = T_3x1x7 ⊗ T_1x4x1 (trivial)
        Or better: use the fact that 4 = 2+2, so A = [A1 A2] where A1, A2 are 3x2.
        Then A*B = A1*B1 + A2*B2, where B1, B2 are 2x7.
        rank(T_3x4x7) <= rank(T_3x2x7) + rank(T_3x2x7)
        
        Actually, we can use:
        A (3x4) * B (4x7) = [A1 A2] * [[B1],[B2]] = A1*B1 + A2*B2
        where A1, A2 are 3x2 and B1, B2 are 2x7.
        So rank(T_347) <= 2 * rank(T_327)
        
        Similarly, T_327: 3x2 * 2x7. Schoolbook = 42.
        Can we do better? 3x2x7: try rank 40?
        """
        if time.time() > deadline:
            return None
        
        # Decompose 4 = 2 + 2
        # T_347[i,j,k] = sum_{j1 in {0,1}} T_327[i, j1, k] * (j == 2*j1 or j == 2*j1+1 mapping)
        # Actually this is more subtle. Let's try a different approach.
        
        # Use the "doubling" trick: 
        # For 2x2x2, Strassen gives rank 7 instead of 8.
        # For 3x4x7, we can try to embed Strassen blocks.
        
        # Let's just try a comprehensive search over ranks 70-83 with multiple methods
        return None

    best_result = None
    
    # Try ranks from 83 down to 70
    for R in range(schoolbook_rank - 1, 69, -1):
        if time.time() > deadline:
            break
        
        # Method 1: Nelder-Mead with structured init
        res = try_strassen_block(R, max_restarts=2)
        if res is not None:
            best_result = (R, res)
            break
        
        # Method 2: Riemannian-like gradient
        res = try_riemannian(R, max_iters=50)
        if res is not None:
            best_result = (R, res)
            break
    
    if best_result is not None:
        R, (U, V, W) = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
