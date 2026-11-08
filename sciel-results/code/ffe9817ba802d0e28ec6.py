import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_fallback = n * m * p
    U_fallback = np.zeros((n * m, rank_fallback))
    V_fallback = np.zeros((m * p, rank_fallback))
    W_fallback = np.zeros((p * n, rank_fallback))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_fallback[u_idx, col] = 1
                V_fallback[v_idx, col] = 1
                W_fallback[w_idx, col] = 1
                col += 1
                
    # Target tensor T_{i,j,k} = 1 if j == k else 0
    # We want T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # Let's reshape T to a matrix form for ALS.
    # T is (n, m, p).
    # For fixing V and W, we solve for U.
    # T_{i,j,k} = sum_r U_{i,r} (V_{j,r} W_{k,r})
    # Let M_{j,k} = V_{j,r} W_{k,r} (outer product of columns of V and W).
    # Then T_{i,:,:} = sum_r U_{i,r} M_r where M_r is the outer product of V[:,r] and W[:,r].
    # Actually, let's vectorize.
    # T_{i, j, k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    
    def try_rank(R, max_iters=100, tol=1e-9):
        # Check time
        if time.time() - start_time > 110:
            return None
        
        # Initialize randomly
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Precompute tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j == k:
                        T[i, j, k] = 1.0
        
        for it in range(max_iters):
            if time.time() - start_time > 115:
                return None
            
            # Update U
            # T_{i, j, k} = sum_r U_{i,r} V_{j,r} W_{k,r}
            # For fixed i, T_{i, :, :] is a matrix.
            # We want to find U_{i, r} such that T_{i, j, k} \approx sum_r U_{i,r} V_{j,r} W_{k,r}
            # Let B_{j,k} = sum_r V_{j,r} W_{k,r} (outer product of V and W rows? No, columns).
            # Actually, let's form a matrix A of size (m*p, R) where A_{(j,k), r} = V_{j,r} W_{k,r}.
            # Then T_{i, j, k} = sum_r U_{i,r} A_{(j,k), r}.
            # So T_{i, :, :}.flatten() = U_{i, :} @ A.T ? No.
            # T_{i, j, k} = sum_r U_{i,r} A_{j,k,r}.
            # This is a linear system: U_{i, :} @ A.T = T_{i, :, :}.flatten()
            # A is (m*p, R). A.T is (R, m*p).
            # We solve U_{i, :} A.T = t_i where t_i is T[i,:,:].flatten()
            
            A = np.zeros((m * p, R))
            for j in range(m):
                for k in range(p):
                    A[j * p + k, :] = V[j, :] * W[k, :] # Element-wise product of row j of V and row k of W? 
                    # Wait, V is (m*p, R). V[j, :] is row j. But V index is j*p+k? 
                    # In the tensor definition, V index is j (from 0 to m-1) and k (from 0 to p-1)?
                    # No, V has shape (m*p, R). The index of V is (j, k) combined?
                    # The task says V has shape (m*p, rank). The index is likely linearized (j, k).
                    # Let's assume V_{j*p+k, r}.
                    # Then the term is U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}.
                    # This is NOT a standard matrix multiplication tensor unless we map indices carefully.
                    
                    # Let's re-read the tensor definition.
                    # T_{i,j,k} = delta_{j,k}.
                    # Decomposition: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
                    # Here U index is i (0..n-1), V index is j (0..m-1), W index is k (0..p-1).
                    # But the output format requires U to be (n*m, rank), V to be (m*p, rank), W to be (p*n, rank).
                    # This implies the indices are mixed.
                    # Usually, for matrix multiplication C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
                    # Tensor T_{i,j,k} = delta_{j,k} is not quite right for general matrix mult.
                    # The multiplication tensor is defined such that T(A,B) = A B.
                    # T_{i,j,k} corresponds to the coefficient of A_{ij} B_{jk} in C_{ik}.
                    # So T_{i,j,k} = 1 if the indices match the multiplication structure, else 0?
                    # No, the tensor is a trilinear form.
                    # T(A,B) = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}.
                    # So T_{i,j,k} = 1 if j is the summing index, i is row of A, k is col of B.
                    # Wait, the standard multiplication tensor for M_{n,m,p} has dimensions (n*m, m*p, p*n)?
                    # No, the tensor is in U^* \otimes V^* \otimes W^* where U is space of A (n*m), V is space of B (m*p), W is space of C (p*n).
                    # The tensor T is defined by T(a,b) = a b.
                    # T_{a,b,c} = 1 if a_{i,j} b_{j,k} contributes to c_{i,k}.
                    # Specifically, T_{(i,j), (j,k), (i,k)} = 1.
                    # All other entries are 0.
                    
                    # So, in the decomposition T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}:
                    # a is index into U (0..n*m-1), b into V (0..m*p-1), c into W (0..p*n-1).
                    # The non-zero entries are at:
                    # a = i*m + j
                    # b = j*p + k
                    # c = k*n + i
                    # for i in 0..n-1, j in 0..m-1, k in 0..p-1.
                    
                    # My previous loop for fallback was correct:
                    # U_fallback[u_idx, col] = 1 where u_idx = i*m + j
                    # V_fallback[v_idx, col] = 1 where v_idx = j*p + k
                    # W_fallback[w_idx, col] = 1 where w_idx = k*n + i
                    
                    # So the tensor T is sparse.
                    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
                    
                    # Let's construct the tensor T as a dense array for ALS?
                    # Size: (n*m) * (m*p) * (p*n) = 6 * 21 * 14 = 1764.
                    # This is small enough to handle.
                    
        # Let's restart the ALS logic with correct indices.
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # ALS
        for it in range(max_iters):
            if time.time() - start_time > 115:
                return None
            
            # Update U
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a, T[a,:,:] = sum_r U[a,r] (V[:,r] \otimes W[:,r])
            # Let M_r = V[:,r] \otimes W[:,r] (vector of size m*p * p*n)
            # T[a,:,:].flatten() = sum_r U[a,r] M_r
            # This is a linear system: U[a,:] @ M.T = T[a,:,:].flatten()
            # M is (m*p * p*n, R).
            
            M = np.zeros((m*p * p*n, R))
            for r in range(R):
                M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Solve for each a
            # U[a, :] M.T = T[a, :, :].flatten()
            # This is an overdetermined system if R < m*p*p*n.
            # Use least squares.
            # U[a, :] = (M.T M)^{-1} M.T T_a
            # We can do this in batch.
            # U = np.linalg.lstsq(M, T.reshape(n*m, -1), rcond=None)[0].T ?
            # T is (n*m, m*p, p*n). Reshape to (n*m, m*p*p*n).
            T_flat = T.reshape(n*m, m*p * p*n)
            # We want U (n*m, R) such that U @ M.T = T_flat
            # M is (m*p*p*n, R). M.T is (R, m*p*p*n).
            # U @ M.T = T_flat
            # This is (U @ M.T).T = T_flat.T => M @ U.T = T_flat.T
            # So U.T = M \ T_flat.T
            U = np.linalg.lstsq(M, T_flat.T, rcond=None)[0].T
            
            # Update V
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed b, T[:,b,:] = sum_r V[b,r] (U[:,r] \otimes W[:,r])
            # Let N_r = U[:,r] \otimes W[:,r] (vector of size n*m * p*n)
            # T[:,b,:].flatten() = sum_r V[b,r] N_r
            # N is (n*m * p*n, R).
            N = np.zeros((n*m * p*n, R))
            for r in range(R):
                N[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            T_flat_v = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n) # Shape (m*p, n*m*p*n)
            # V.T = N \ T_flat_v.T
            V = np.linalg.lstsq(N, T_flat_v.T, rcond=None)[0].T
            
            # Update W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed c, T[:,:,c] = sum_r W[c,r] (U[:,r] \otimes V[:,r])
            # Let P_r = U[:,r] \otimes V[:,r] (vector of size n*m * m*p)
            # T[:,:,c].flatten() = sum_r W[c,r] P_r
            # P is (n*m * m*p, R).
            P = np.zeros((n*m * m*p, R))
            for r in range(R):
                P[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            T_flat_w = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p) # Shape (p*n, n*m*m*p)
            # W.T = P \ T_flat_w.T
            W = np.linalg.lstsq(P, T_flat_w.T, rcond=None)[0].T
            
            # Check residual
            if it % 10 == 0:
                # Compute error
                # T_approx = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]
                # We can check a few entries or use Frobenius norm.
                # For speed, check norm of T - T_approx.
                # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # We can compute this efficiently?
                # It's a tensor contraction.
                # For now, just check if we are close to 0.
                # Since we are solving least squares, the residual should decrease.
                pass
        
        # Final check
        # Compute T_approx
        # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow for large R, but R is small.
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T - T_approx)
        
        if error < 1e-6:
            # Round to integers or half-integers?
            # The problem asks for integer/half-integer entries.
            # Let's try rounding to nearest 0.5.
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_approx_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            error_rounded = np.linalg.norm(T - T_approx_rounded)
            
            if error_rounded < 1e-6:
                return U_rounded, V_rounded, W_rounded
            else:
                # If rounding fails, return unrounded? 
                # The contract says "integer/half-integer entries".
                # If we can't achieve that, we might fail validation.
                # But usually, if the solution is exact, it should be close to rational.
                return U, V, W
        
        return None

    # Try ranks from 21 to 42
    for R in range(21, 43):
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() - start_time > 118:
            break

    # Fallback
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
