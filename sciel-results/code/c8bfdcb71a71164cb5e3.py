import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS

    def check_time():
        return time.time() > deadline

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
                    t += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact: for all i,j,k, sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] == delta_{ik} (for matrix mult)"""
        # Actually, the matrix multiplication tensor is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # The standard form: (C = A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # Tensor form: M[i,j,k] where the "multiplication" gives sum over j
        # The factorization: M = sum_t u_t ⊗ v_t ⊗ w_t
        # where (u_t)_{i,j} = U[i*m+j, t], (v_t)_{j,k} = V[j*p+k, t], (w_t)_{k,i} = W[k*n+i, t]
        # And the condition is: for each (i,k), sum_j M[i,j,k] * (A[i,j]*B[j,k]) = C[i,k]
        # Equivalently: the tensor M[i,j,k] must satisfy that for any A,B:
        #   C[i,k] = sum_{j,t} A[i,j] * U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        # This must equal sum_j A[i,j]*B[j,k] for all A,B
        # So we need: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = delta_{j,k} ... no wait
        # Let me re-derive. C[i,k] = sum_j A[i,j] B[j,k]
        # We want: C[i,k] = sum_t (sum_j A[i,j] * U[i*m+j,t]) * (sum_k' ... )
        # Actually the standard formulation:
        # C[i,k] = sum_t alpha_t * (sum_j A[i,j] * U_t[i,j]) * (sum_k B[j,k] * V_t[j,k]) ... no
        # 
        # Standard: C[i,k] = sum_t (sum_j A[i,j] U_t[i,j]) * (sum_j B[j,k] V_t[j,k])
        # Wait, that's not right either. Let me think more carefully.
        #
        # The matrix multiplication tensor M has entries M[i,j,k] such that
        # (A * B)[i,k] = sum_{j} A[i,j] * B[j,k]
        # In tensor form, this is a trilinear form: sum_{i,j,k} M[i,j,k] * A[i,j] * B[j,k] * C[i,k]
        # where M[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] contributes to C[i,k])
        # Wait no. The bilinear map is: (A,B) -> A*B
        # The tensor associated with this bilinear map has entries T[i,k; i',j',j'',k''] ... 
        # 
        # Let me use the standard tensor decomposition framework.
        # We want to compute C = A * B where A is n×m, B is m×p, C is n×p.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # A rank-1 decomposition of the multiplication tensor is:
        # C[i,k] = sum_t (sum_j A[i,j] * U_t[i,j]) * (sum_j B[j,k] * V_t[j,k])
        # where U_t and V_t are the "left" and "right" factors for term t, and we need
        # sum_t U_t[i,j] * V_t[j,k] = delta... no.
        #
        # Actually: C[i,k] = sum_t U_t[i,j] ... this doesn't work directly.
        #
        # The correct formulation: We want
        # C[i,k] = sum_t (sum_j A[i,j] * U_t[i,j]) * (sum_k' ... )
        #
        # No. The standard way: The multiplication tensor M ∈ R^{(n*m) × (m*p) × (n*p)} is defined by
        # M[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k' ... no.
        #
        # Let me use the formulation from the problem statement. The output factor W follows
        # W[k*n+i, r]. So W is indexed by (k,i) with row index k*n+i.
        #
        # The decomposition is: for each "output" (i,k), we have:
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  ... summed over j and r
        #
        # So: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        #
        # For this to equal sum_j A[i,j] * B[j,k] for all A, B, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,k} ... no, that's not right.
        #
        # We need: sum_{j,r} A[i,j] B[j,k] U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = sum_j A[i,j] B[j,k]
        #
        # This must hold for ALL A and B. So for each fixed (i,k), the coefficient of A[i,j1]*B[j2,k] must be:
        # = 1 if j1=j2, 0 otherwise
        #
        # The coefficient of A[i,j1]*B[j2,k] in the LHS is:
        # sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r]
        #
        # So we need: sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta_{j1,j2} for all i,k,j1,j2
        #
        # Wait, that can't be right because the left side depends on i and k but the right side doesn't.
        # Let me re-examine.
        #
        # Actually I think the standard formulation is different. Let me look at it from the tensor perspective.
        #
        # The matrix multiplication tensor M is a 3-way tensor with dimensions (n*m) x (m*p) x (n*p).
        # M[a, b, c] where a = i*m+j (input A index), b = j'*p+k (input B index), c = i'*n+k' (output C index)
        # M[a, b, c] = 1 if i=i' and j=j' and k=k', else 0.
        #
        # A rank-R decomposition: M = sum_{r=1}^R U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # i.e., M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        #
        # So we need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = 1 if (i,j,k)=(i',j',k') else 0
        #
        # Hmm, but that means the tensor has 27 ones (for the 3*3*3 = 27 positions where i=i', j=j', k=k')
        # and 0 elsewhere. But the tensor has 9*9*9 = 729 entries.
        #
        # Wait, I think I'm overcomplicating this. Let me re-read the problem.
        #
        # "exactly decompose the matrix multiplication tensor"
        # "output factor follows the official trace convention W[k*n+i,r]"
        #
        # I think the standard convention is:
        # The multiplication tensor T has entries T[i,j,k] for i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
        # where T[i,j,k] corresponds to the term A[i,j]*B[j,k] contributing to C[i,k].
        #
        # In the factorization: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
        #
        # No wait, that has an extra factor. Let me think again.
        #
        # I believe the correct formulation is:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
        #
        # No, that's not standard either. Let me just use the direct tensor formulation.
        #
        # The tensor M ∈ R^{(nm) × (mp) × (np)} where:
        # M[i*m+j, j'*p+k, i'*n+k'] = 1 if i=i', j=j', k=k' else 0
        #
        # Rank-R factorization: M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        #
        # So: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = [i=i'] * [j=j'] * [k=k']
        #
        # This is what we need to satisfy. Let me verify with schoolbook:
        # For schoolbook, rank=27, and for each (i,j,k) triple, we have one term r=(i,j,k) with
        # U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1, all others 0.
        # Then sum_r ... = 1 if (i,j,k)=(i',j',k') else 0. Correct!
        #
        # So the condition is: for all i,j,j',k,k',i':
        # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = [i=i']*[j=j']*[k=k']
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = len(U_arr[0])
        
        # Build the 3D tensor from the factorization
        # M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        M_reconstructed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Build the target tensor
        M_target = np.zeros((n*m, m*p, n*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        err = np.max(np.abs(M_reconstructed - M_target))
        return err < tol, err

    def try_rank(R, max_iters=500, seed=42):
        """Try to find a rank-R decomposition using ALS."""
        rng = np.random.RandomState(seed)
        
        U_arr = rng.randn(n*m, R) * 0.1
        V_arr = rng.randn(m*p, R) * 0.1
        W_arr = rng.randn(n*p, R) * 0.1
        
        # Target tensor
        M_target = np.zeros((n*m, m*p, n*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Reshape M_target to a matrix for ALS
        # M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed c: M[:,:,c] = U * diag(V[:,c]... ) no
        # M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Fix c: M[:,:,c] = U * (V * diag(W[:,c]))^T ... no
        # M[:,:,c]_{a,b} = sum_r U[a,r] * V[b,r] * W[c,r] = (U * diag(W[:,c])) @ V.T
        # So M[:,:,c] = U @ diag(W[:,c]) @ V.T
        
        # ALS: 
        # Fix V, W: minimize ||M[:,:,c] - U @ diag(W[:,c]) @ V.T||_F^2 for all c
        # This is: for each c, M[:,:,c] = U @ (diag(W[:,c]) @ V.T)
        # Let B_c = diag(W[:,c]) @ V.T, then M[:,:,c] = U @ B_c
        # Stack: M_stacked = [M[:,:,0], M[:,:,1], ..., M[:,:,np-1]] (each is nm x mp)
        # We want U such that for all c: U @ B_c ≈ M[:,:,c]
        # This is a linear least squares problem if we stack properly.
        
        for it in range(max_iters):
            if check_time():
                return None
            
            # Update U: fix V, W
            # For each c: M[:,:,c] = U @ (diag(W[:,c]) @ V.T)
            # Let C_c = diag(W[:,c]) @ V.T (R x mp)
            # Then M[:,:,c] = U @ C_c
            # Stack all c: M_stacked (nm x mp*np) = U @ C_stacked (R x mp*np)
            # where C_stacked = [C_0, C_1, ..., C_{np-1}]
            
            C_stacked = np.zeros((R, m*p*n*p))
            for c in range(n*p):
                C_c = np.diag(W_arr[:, c]) @ V_arr.T  # (R x mp)
                C_stacked[:, c*m*p:(c+1)*m*p] = C_c
            
            M_stacked = M_target.reshape(n*m, m*p*n*p)
            
            # Solve U @ C_stacked ≈ M_stacked
            U_arr, _, _, _ = np.linalg.lstsq(C_stacked.T, M_stacked.T, rcond=None)
            U_arr = U_arr.T
            
            # Update V: fix U, W
            # M[:,:,c] = U @ diag(W[:,c]) @ V.T
            # Let D_c = U @ diag(W[:,c]) (nm x R)
            # Then M[:,:,c] = D_c @ V.T
            # Stack: M_stacked = D_stacked @ V.T
            # where D_stacked = [D_0, D_1, ..., D_{np-1}] (nm x R*np)
            
            D_stacked = np.zeros((n*m, R*n*p))
            for c in range(n*p):
                D_c = U_arr @ np.diag(W_arr[:, c])  # (nm x R)
                D_stacked[:, c*R:(c+1)*R] = D_c
            
            # M_stacked = D_stacked @ V.T
            # So V.T = D_stacked^+ @ M_stacked, i.e., V = (D_stacked^+ @ M_stacked)^T
            V_arr, _, _, _ = np.linalg.lstsq(D_stacked, M_stacked, rcond=None)
            V_arr = V_arr.T  # Wait, let me check dimensions.
            # D_stacked is (nm x R*np), M_stacked is (nm x mp*np)
            # We want V.T (mp x R*np) such that D_stacked @ V.T ≈ M_stacked
            # So V.T = lstsq(D_stacked, M_stacked) gives (R*np x mp*np)... 
            # Hmm, let me redo.
            # M_stacked (nm x mp*np) = D_stacked (nm x R*np) @ X (R*np x mp*np)
            # X = V.T reshaped? No.
            # Actually M[:,:,c] = D_c @ V.T where D_c is (nm x R) and V.T is (R x mp)
            # So for each c: M[:,:,c] (nm x mp) = D_c (nm x R) @ V.T (R x mp)
            # Stacking over c: M_stacked (nm x mp*np) = [D_0, D_1, ..., D_{np-1}] (nm x R*np) @ 
            #   [V.T, V.T, ..., V.T] (R*np x mp*np)? No, that's not right.
            #
            # Let me think differently. For each c, we have:
            # M[:,:,c] = D_c @ V.T
            # This is a separate least squares problem for each c, but V is shared.
            # So we solve: minimize sum_c ||M[:,:,c] - D_c @ V.T||_F^2
            # = minimize ||M_stacked - D_stacked @ V_stacked||_F^2
            # where V_stacked is (R*np x mp*np) with V.T repeated np times? No.
            #
            # Actually, let's flatten differently.
            # M[:,:,c] (nm x mp) = D_c (nm x R) @ V.T (R x mp)
            # Vectorize: vec(M[:,:,c]) = (V ⊗ I_nm) vec(D_c) ... no, vec(ABC) = (C^T ⊗ A) vec(B)
            # vec(D_c @ V.T) = (V ⊗ I_nm) vec(D_c) ... let me just use the matrix form.
            #
            # For each c: M[:,:,c] = D_c @ V.T
            # => M[:,:,c].T = V @ D_c.T
            # => vec(M[:,:,c].T) = (D_c ⊗ I_mp) vec(V)
            # vec(M[:,:,c].T) is (mp*nm), D_c is (nm x R), so D_c ⊗ I_mp is (mp*nm x R*mp)
            # This is getting complicated. Let me just do it per-c and average.
            
            # Simpler approach: solve for V by stacking
            # For each c: M[:,:,c] = D_c @ V.T
            # Reshape: M_c_flat = M[:,:,c].T.reshape(-1) (mp*nm,)
            # D_c_flat = D_c.T.reshape(-1) ... no
            # M[:,:,c] (nm x mp) = D_c (nm x R) @ V.T (R x mp)
            # Let me just solve each one and average, or use a different stacking.
            
            # Alternative: M[:,:,c] = D_c @ V.T for all c
            # => [M[:,:,0], M[:,:,1], ..., M[:,:,np-1]] = [D_0, D_1, ..., D_{np-1}] @ V.T
            # But the RHS is (nm x R*np) @ (R*np x mp*np) which doesn't give (nm x mp*np) unless
            # V.T is repeated. Let me define V_rep = [V.T, V.T, ..., V.T] (R*np x mp*np)
            # Then M_stacked = D_stacked @ V_rep
            # But V_rep has repeated blocks, so we can't directly solve.
            
            # Let me just do a simple gradient step or use scipy.
            # Actually, let me reformulate.
            # M[:,:,c] = D_c @ V.T
            # => vec(M[:,:,c]) = (I_mp ⊗ D_c) vec(V.T) ... no
            # vec(D_c @ V.T) = (I_mp ⊗ D_c) vec(V.T) where V.T is (R x mp)
            # vec(V.T) is (mp*R,). I_mp ⊗ D_c is (mp*nm x mp*R).
            # This is a big system. Let me just use scipy.optimize or a simple approach.
            
            # For now, let me use a simpler ALS where I solve for V by:
            # For each c, M[:,:,c] = D_c @ V.T
            # => V.T = pinv(D_c) @ M[:,:,c]
            # Average over c: V.T = (1/np) * sum_c pinv(D_c) @ M[:,:,c]
            # This is not optimal but might work.
            
            V_new = np.zeros((m*p, R))
            for c in range(n*p):
                D_c = U_arr @ np.diag(W_arr[:, c])  # (nm x R)
                # M[:,:,c] = D_c @ V.T => V.T = pinv(D_c) @ M[:,:,c]
                V_c, _, _, _ = np.linalg.lstsq(D_c, M_target[:, :, c], rcond=None)
                V_new += V_c.T  # (R x mp) transposed to (mp x R)
            V_new /= n*p
            V_arr = V_new
            
            # Update W: fix U, V
            # M[:,:,c] = U @ diag(W[:,c]) @ V.T
            # Let E_c = U @ V.T (nm x mp) ... no, that's not right.
            # M[:,:,c]_{a,b} = sum_r U[a,r] * W[c,r] * V[b,r]
            # = (U * W[c,:]) @ V.T  where U*W[c,:] means element-wise multiply rows
            # Let F_c = U * W[c,:] (nm x R), then M[:,:,c] = F_c @ V.T
            # => F_c = M[:,:,c] @ pinv(V.T)
            # => U[a,r] * W[c,r] = F_c[a,r]
            # => W[c,r] = F_c[a,r] / U[a,r] for any a where U[a,r] != 0
            # This is tricky. Let me use a different approach.
            #
            # M[:,:,c] = (U * W[c,:]) @ V.T
            # vec(M[:,:,c]) = (V ⊗ I_nm) vec(U * W[c,:]) ... this is getting complicated.
            #
            # Let me just use a simple approach: for each c, solve for W[:,c]
            # M[:,:,c]_{a,b} = sum_r U[a,r] * W[c,r] * V[b,r]
            # For fixed a,b: M[:,:,c][a,b] = sum_r (U[a,r] * V[b,r]) * W[c,r]
            # So M[:,:,c][a,b] = (U[a,:] * V[b,:]) @ W[c,:]
            # This is a linear system in W[c,:]:
            # [U[a0,:]*V[b0,:]; U[a1,:]*V[b1,:]; ...] @ W[c,:] = [M[:,:,c][a0,b0]; M[:,:,c][a1,b1]; ...]
            
            W_new = np.zeros((n*p, R))
            for c in range(n*p):
                # Build the system
                A_mat = np.zeros((n*m*m*p, R))
                b_vec = np.zeros(n*m*m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A_mat[idx, :] = U_arr[a, :] * V_arr[b, :]
                        b_vec[idx] = M_target[a, b, c]
                        idx += 1
                W_c, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_new[c, :] = W_c
            W_arr = W_new
            
            # Check convergence
            M_reconstructed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
            err = np.max(np.abs(M_reconstructed - M_target))
            if err < 1e-10:
                # Convert to lists
                U_list = U_arr.tolist()
                V_list = V_arr.tolist()
                W_list = W_arr.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Final check
        M_reconstructed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        err = np.max(np.abs(M_reconstructed - M_target))
        if err < 1e-8:
            U_list = U_arr.tolist()
            V_list = V_arr.tolist()
            W_list = W_arr.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try rank 23
    for seed in range(10):
        if check_time():
            break
        result = try_rank(23, max_iters=200, seed=seed)
        if result is not None:
            valid, err = verify(result["U"], result["V"], result["W"])
            if valid:
                return result
    
    # Try rank 24
    if not check_time():
        for seed in range(5):
            if check_time():
                break
            result = try_rank(24, max_iters=200, seed=seed + 100)
            if result is not None:
                valid, err = verify(result["U"], result["V"], result["W"])
                if valid:
                    return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
