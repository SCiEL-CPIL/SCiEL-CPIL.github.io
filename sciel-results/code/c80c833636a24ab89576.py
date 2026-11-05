import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # T has shape (n*m, m*p, p*n) but we work with flattened indices
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # Contribution to output index (i,k) from input indices (i,j) and (j,k)
    
    # The tensor entry T[(i*m+j), (j*p+k), (k*n+i)] = 1 for valid (i,j,k)
    # Otherwise 0
    
    # First, try to find rank-16 decomposition using HOSVD-guided approach
    # Then fall back to rank-17 if rank-16 doesn't yield exact integer/half-integer solution
    
    def try_rank_r(r):
        """Try to find an exact decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        # Build the tensor in flat form: T[a,b,c] where
        # a = i*m + j (input A index), b = j*p + k (input B index), c = k*n + i (output C index)
        # T[a,b,c] = 1 if there exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
        # This means: i = a//m, j = a%m, must equal b//p, k = b%p, must satisfy c = k*n + i
        
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # For rank r, we need U (nm x r), V (mp x r), W (pn x r) such that
        # T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
        
        # HOSVD approach: First get a good initial estimate
        # Mode-1 unfolding: T1[a, b*p*n + c] = sum_c T[a,b,c]
        # Actually, let's use a simpler approach: start from the known rank-17 
        # and try to merge two terms or find a lower-rank approximation
        
        # Alternative: Use the structure. For 2x3x3, we know rank 17 works.
        # Try rank 16 by looking for a linear dependence.
        
        # Let's try a direct approach: fix U and V from a partial decomposition,
        # then solve for W, or use alternating minimization.
        
        # Start with the rank-17 solution as initialization
        U17, V17, W17 = _get_rank17_factors(n, m, p)
        
        if r >= 17:
            return U17, V17, W17
        
        # For rank 16, try to find a linear combination that eliminates one term
        # or use SVD-based compression
        
        # Approach: Take the rank-17 factors, form the 3-mode tensor, 
        # then try to find a rank-16 approximation
        
        # Reshape to 3D tensor for HOSVD
        T_est = np.einsum('at,bt,ct->abc', U17, V17, W17)
        
        # Perform HOSVD on T_est to get a rank-r approximation
        # Simple approach: truncate the singular values in each mode
        
        # Mode-1 SVD: T1[a, (b,c)]
        T1 = T_est.reshape(n*m, m*p*p*n)
        U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        
        # Mode-2 SVD: T2[b, (a,c)]
        T2 = np.transpose(T_est, (1, 0, 2)).reshape(m*p, n*m*p*n)
        U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        
        # Mode-3 SVD: T3[c, (a,b)]
        T3 = np.transpose(T_est, (2, 0, 1)).reshape(p*n, n*m*m*p)
        U3, S3, Vt3 = np.linalg.svd(T3, full_matrices=False)
        
        # Truncate to rank r
        U1_r = U1[:, :r]
        U2_r = U2[:, :r]
        U3_r = U3[:, :r]
        
        # Now we have approximate factors. Try to refine them to get exact integer/half-integer values.
        # Use alternating least squares
        
        U_curr = np.einsum('ai,ci->ac', U1_r, np.ones((r, r)))  # rough init
        V_curr = np.einsum('bj,cj->bc', U2_r, np.ones((r, r)))
        W_curr = np.einsum('ck,ak->ac', U3_r, np.ones((r, r)))
        
        # Better initialization: use the truncated SVD vectors directly
        U_curr = U1_r.copy()
        V_curr = U2_r.copy()
        W_curr = U3_r.copy()
        
        # Alternating minimization
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Update U: For each column t of U, solve least squares
            # T[a,b,c] ≈ U[a,t] * V[b,t] * W[c,t]
            # Fix V and W, solve for U:
            # For each t, U[:,t] should satisfy: T[:, :, :] ≈ outer(U[:,t], V[:,t] * W[:,t])
            # This is: T.reshape(n*m, -1)[:, b*p*n+c] = U[a,t] * V[b,t] * W[c,t]
            # For fixed t: T_flat[a, idx] = U[a,t] * (V[:,t] * W[:,t]).flatten()[idx]
            # So U[:,t] = pinv(VW[:,t]) @ T_flat[:, :] ... this is complex
            
            # Simpler: use gradient descent or direct linear system
            
            # For each t, the contribution is a rank-1 tensor u_t ⊗ v_t ⊗ w_t
            # We want sum_t u_t ⊗ v_t ⊗ w_t = T
            # Fix v_t, w_t for all t, solve for u_t:
            # T[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # For fixed (b,c): T[:, b, c] = sum_t U[:,t] * V[b,t] * W[c,t]
            # This is a linear system in U[:,t]
            
            # Build the system: For each (b,c), we have n*m equations
            # T[a, b, c] = sum_t U[a,t] * (V[b,t] * W[c,t])
            # Let M_bc[a, t] = V[b,t] * W[c,t]
            # Then T[:, b, c] = M_bc @ U[:, :]^T ... no
            
            # Actually: T[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # Fix V, W. For each a, T[a, b, c] = sum_t U[a,t] * (V[b,t] * W[c,t])
            # Let Y[a, (b,c)] = T[a, b, c].reshape(n*m, m*p*p*n)
            # Let Z[(b,c), t] = V[b,t] * W[c,t].reshape(m*p*p*n, r)
            # Then Y = U @ Z^T, so U = Y @ pinv(Z^T)
            
            Y = T.reshape(n*m, m*p * p*n)
            Z = (V_curr * W_curr[None, :, :]).reshape(m*p*p*n, r)  # Z[(b,c), t] = V[b,t]*W[c,t]
            # Wait, V_curr is (m*p, r), W_curr is (p*n, r)
            # We need V[b,t] * W[c,t] for each (b,c,t)
            # V_curr[b, t] * W_curr[c, t]
            # Create Z of shape (m*p, p*n, r) then reshape
            Z3 = V_curr[:, None, :] * W_curr[None, :, :]  # (m*p, p*n, r)
            Z2 = Z3.reshape(m*p*p*n, r)
            
            # Y[a, (b,c)] = sum_t U[a,t] * Z2[(b,c), t]
            # So Y = U @ Z2^T
            # U = Y @ (Z2^T)^+ = Y @ pinv(Z2.T)
            try:
                U_new = Y @ np.linalg.pinv(Z2.T)
            except:
                U_new = U_curr.copy()
            
            # Check if U_new is close to integer/half-integer
            U_curr = U_new
            
            # Similarly update V and W
            # T[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # Fix U, W. For each b: T[a, b, c] = sum_t V[b,t] * (U[a,t] * W[c,t])
            # Y2[b, (a,c)] = T[a, b, c].transpose(1, 0, 2).reshape(m*p, n*m*p*n)
            # Z3[(a,c), t] = U[a,t] * W[c,t]
            Y2 = np.transpose(T, (1, 0, 2)).reshape(m*p, n*m * p*n)
            Z3b = (U_curr[:, None, :] * W_curr[None, :, :]).reshape(n*m*p*n, r)
            try:
                V_new = Y2 @ np.linalg.pinv(Z3b.T)
            except:
                V_new = V_curr.copy()
            V_curr = V_new
            
            # Update W
            Y3 = np.transpose(T, (2, 0, 1)).reshape(p*n, n*m * m*p)
            Z3c = (U_curr[:, None, :] * V_curr[None, :, :]).reshape(n*m*m*p, r)
            try:
                W_new = Y3 @ np.linalg.pinv(Z3c.T)
            except:
                W_new = W_curr.copy()
            W_curr = W_new
        
        # Check if the solution is exact (within tolerance) and has integer/half-integer entries
        T_recon = np.einsum('at,bt,ct->abc', U_curr, V_curr, W_curr)
        err = np.max(np.abs(T_recon - T))
        
        if err > 1e-6:
            return None
        
        # Check for integer/half-integer entries
        def is_half_integer(arr):
            scaled = 2 * arr
            rounded = np.round(scaled)
            return np.max(np.abs(scaled - rounded)) < 1e-6
        
        if not (is_half_integer(U_curr) and is_half_integer(V_curr) and is_half_integer(W_curr)):
            # Try rounding
            U_round = np.round(2 * U_curr) / 2
            V_round = np.round(2 * V_curr) / 2
            W_round = np.round(2 * W_curr) / 2
            
            T_round = np.einsum('at,bt,ct->abc', U_round, V_round, W_round)
            err_round = np.max(np.abs(T_round - T))
            
            if err_round > 1e-6:
                return None
            
            U_curr, V_curr, W_curr = U_round, V_round, W_round
        
        return U_curr, V_curr, W_curr
    
    def _get_rank17_factors(n, m, p):
        """Return the known rank-17 factors."""
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val

        # M0: (A00 + A11)(B00 + B11)
        set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
        set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
        set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
        
        # M1: (A10 + A11)B00
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
        
        # M2: A00(B01 - B11)
        set_U(0, 0, 2, 1)
        set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
        set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
        
        # M3: A11(B10 - B00)
        set_U(1, 1, 3, 1)
        set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
        
        # M4: (A00 + A01)B11
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        # M5: (A10 - A00)(B00 + B01)
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        # M6: (A11 - A01)(B10 + B11)
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 0, 6, 1)

        # Term 7: A[0,0] * B[0,2]
        set_U(0, 0, 7, 1)
        set_V(0, 2, 7, 1)
        set_W(2, 0, 7, 1)
        
        # Term 8: A[0,1] * B[1,2]
        set_U(0, 1, 8, 1)
        set_V(1, 2, 8, 1)
        set_W(2, 0, 8, 1)
        
        # Term 9: A[1,0] * B[0,2]
        set_U(1, 0, 9, 1)
        set_V(0, 2, 9, 1)
        set_W(2, 1, 9, 1)
        
        # Term 10: A[1,1] * B[1,2]
        set_U(1, 1, 10, 1)
        set_V(1, 2, 10, 1)
        set_W(2, 1, 10, 1)
        
        # Term 11: A[0,2] * B[2,0]
        set_U(0, 2, 11, 1)
        set_V(2, 0, 11, 1)
        set_W(0, 0, 11, 1)
        
        # Term 12: A[0,2] * B[2,1]
        set_U(0, 2, 12, 1)
        set_V(2, 1, 12, 1)
        set_W(1, 0, 12, 1)
        
        # Term 13: A[0,2] * B[2,2]
        set_U(0, 2, 13, 1)
        set_V(2, 2, 13, 1)
        set_W(2, 0, 13, 1)
        
        # Term 14: A[1,2] * B[2,0]
        set_U(1, 2, 14, 1)
        set_V(2, 0, 14, 1)
        set_W(0, 1, 14, 1)
        
        # Term 15: A[1,2] * B[2,1]
        set_U(1, 2, 15, 1)
        set_V(2, 1, 15, 1)
        set_W(1, 1, 15, 1)
        
        # Term 16: A[1,2] * B[2,2]
        set_U(1, 2, 16, 1)
        set_V(2, 2, 16, 1)
        set_W(2, 1, 16, 1)
        
        return U, V, W
    
    # Try rank 16 first
    result_16 = try_rank_r(16)
    if result_16 is not None:
        U, V, W = result_16
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fall back to rank 17
    U17, V17, W17 = _get_rank17_factors(n, m, p)
    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }
