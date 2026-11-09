import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Try to find a rank-26 decomposition using optimization
    # Start with rank 26 and try to refine
    best_rank = 28
    best_U = None
    best_V = None
    best_W = None
    
    # First, create the rank-28 solution as a baseline
    def make_rank28():
        U28 = np.zeros((n*m, 28))
        V28 = np.zeros((m*p, 28))
        W28 = np.zeros((p*n, 28))
        
        # Strassen terms for 2x2 matrix multiplication
        strassen_terms = [
            {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
            {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
            {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
            {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
            {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
            {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
            {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
        ]
        
        blocks = [
            {'j_offset': 0, 'k_offset': 0},
            {'j_offset': 0, 'k_offset': 2},
            {'j_offset': 2, 'k_offset': 0},
            {'j_offset': 2, 'k_offset': 2}
        ]
        
        col = 0
        for block in blocks:
            j_off = block['j_offset']
            k_off = block['k_offset']
            for term in strassen_terms:
                for (idx, coeff) in term['U']:
                    i = idx // 2
                    j = idx % 2
                    U28[i*m + (j + j_off), col] += coeff
                
                for (idx, coeff) in term['V']:
                    j = idx // 2
                    k = idx % 2
                    V28[(j + j_off)*p + (k + k_off), col] += coeff
                
                for (idx, coeff) in term['W']:
                    k = idx // 2
                    i = idx % 2
                    W28[(k + k_off)*n + i, col] += coeff
                
                col += 1
        
        return U28, V28, W28
    
    U28, V28, W28 = make_rank28()
    best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()
    
    # Build the full multiplication tensor
    # T[i, j, k] = 1 if the (i,k) entry of C = A*B uses A[i,j]*B[j,k]
    # For n x m times m x p: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Try rank 26 using optimization
    def try_rank(r, U0, V0, W0, max_iter=200):
        """Try to find a rank-r decomposition starting from initial guess."""
        # Flatten all parameters
        # U: n*m x r, V: m*p x r, W: p*n x r
        nU = n*m*r
        nV = m*p*r
        nW = p*n*r
        total = nU + nV + nW
        
        def loss(params):
            U = params[:nU].reshape(n*m, r)
            V = params[nU:nU+nV].reshape(m*p, r)
            W = params[nU+nV:].reshape(p*n, r)
            
            # Compute tensor from decomposition
            # T_approx[i,j,k] = sum_l U[i,j,l] * V[j,k,l] * W[k,i,l]
            # Note: W is indexed by (k, i) for the trace convention
            T_approx = np.zeros((n, m, p))
            for l in range(r):
                # U[:, :, l] is n x m
                # V[:, :, l] is m x p (but indexed as (j*p+k, l) so reshape to m x p)
                u_l = U[:, l].reshape(n, m)
                v_l = V[:, l].reshape(m, p)
                w_l = W[:, l].reshape(p, n)
                
                # T_approx[i,j,k] += u_l[i,j] * v_l[j,k] * w_l[k,i]
                # This is an outer product of three 2D arrays with specific index pattern
                for i in range(n):
                    for k in range(p):
                        T_approx[i, :, k] += u_l[i, :] * v_l[:, k] * w_l[k, i]
            
            diff = T - T_approx
            return np.sum(diff * diff)
        
        # Initial guess: truncate the rank-28 solution or use random
        if U0 is not None:
            U_flat = U0.reshape(-1)
            V_flat = V0.reshape(-1)
            W_flat = W0.reshape(-1)
        else:
            U_flat = np.random.randn(nU) * 0.1
            V_flat = np.random.randn(nV) * 0.1
            W_flat = np.random.randn(nW) * 0.1
        
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        # Use L-BFGS-B
        try:
            result = opt.minimize(loss, x0, method='L-BFGS-B', 
                                  options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                U_opt = result.x[:nU].reshape(n*m, r)
                V_opt = result.x[nU:nU+nV].reshape(m*p, r)
                W_opt = result.x[nU+nV:].reshape(p*n, r)
                return U_opt, V_opt, W_opt
        except Exception as e:
            pass
        
        return None
    
    # Try rank 26 with different initializations
    np.random.seed(42)
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        # Try with truncated rank-28 as initial guess (pad with zeros or use first 26 columns)
        if attempt == 0:
            U_init = U28[:, :26]
            V_init = V28[:, :26]
            W_init = W28[:, :26]
        elif attempt == 1:
            # Random initialization
            U_init = None
            V_init = None
            W_init = None
        else:
            # Perturbed rank-28
            U_init = U28[:, :26] + np.random.randn(n*m, 26) * 0.1
            V_init = V28[:, :26] + np.random.randn(m*p, 26) * 0.1
            W_init = W28[:, :26] + np.random.randn(p*n, 26) * 0.1
        
        result = try_rank(26, U_init, V_init, W_init)
        if result is not None:
            U26, V26, W26 = result
            # Verify
            T_check = np.zeros((n, m, p))
            for l in range(26):
                u_l = U26[:, l].reshape(n, m)
                v_l = V26[:, l].reshape(m, p)
                w_l = W26[:, l].reshape(p, n)
                for i in range(n):
                    for k in range(p):
                        T_check[i, :, k] += u_l[i, :] * v_l[:, k] * w_l[k, i]
            if np.max(np.abs(T_check - T)) < 1e-8:
                best_rank = 26
                best_U, best_V, best_W = U26, V26, W26
                break
    
    # Convert to list format with integer/half-integer rounding if needed
    def to_list(mat):
        arr = np.array(mat)
        # Round to nearest 0.5
        arr = np.round(arr * 2) / 2
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
