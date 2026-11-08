"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Build the multiplication tensor T[i,j,k] where:
    # i = row of A (0..n-1), j = col of A / row of B (0..m-1), k = col of B (0..p-1)
    # T[i,j,k] = 1 if this entry contributes to C[i,k]
    # Actually the tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[a_idx, b_idx, c_idx] = 1 if A[a_idx]*B[b_idx] contributes to C[c_idx]
    # a_idx = i*m + j (0..8), b_idx = j*p + k (0..11), c_idx = k*n + i (0..11)
    
    # T is a 9x12x12 tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Try rank 33
    rank = 33
    # Initialize with the rank-34 solution, then try to find a null vector
    
    # First, build the rank-34 solution
    U34 = np.zeros((n*m, 34))
    V34 = np.zeros((m*p, 34))
    W34 = np.zeros((p*n, 34))
    
    col = 0
    def add34(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U34[u_idx, col] += u_val
        V34[v_idx, col] += v_val
        W34[w_idx, col] += w_val
        col += 1
    
    # Strassen Left
    U34[0,col]=1;U34[4,col]=1;V34[0,col]=1;V34[5,col]=1;W34[0,col]=1;W34[4,col]=1;col+=1
    U34[3,col]=1;U34[4,col]=1;V34[0,col]=1;W34[1,col]=1;W34[4,col]=-1;col+=1
    U34[0,col]=1;V34[1,col]=1;V34[5,col]=-1;W34[3,col]=1;W34[4,col]=1;col+=1
    U34[4,col]=1;V34[4,col]=1;V34[0,col]=-1;W34[0,col]=1;W34[1,col]=1;col+=1
    U34[0,col]=1;U34[1,col]=1;V34[5,col]=1;W34[0,col]=-1;W34[3,col]=1;col+=1
    U34[3,col]=1;U34[0,col]=-1;V34[0,col]=1;V34[1,col]=1;W34[4,col]=1;col+=1
    U34[1,col]=1;U34[4,col]=-1;V34[4,col]=1;V34[5,col]=1;W34[0,col]=1;col+=1
    # A12*B2L
    U34[2,col]=1;V34[8,col]=1;W34[0,col]=1;col+=1
    U34[2,col]=1;V34[9,col]=1;W34[3,col]=1;col+=1
    U34[5,col]=1;V34[8,col]=1;W34[1,col]=1;col+=1
    U34[5,col]=1;V34[9,col]=1;W34[4,col]=1;col+=1
    # A21*B1L
    U34[6,col]=1;V34[0,col]=1;W34[2,col]=1;col+=1
    U34[6,col]=1;V34[1,col]=1;W34[5,col]=1;col+=1
    U34[7,col]=1;V34[4,col]=1;W34[2,col]=1;col+=1
    U34[7,col]=1;V34[5,col]=1;W34[5,col]=1;col+=1
    # A22*B2L
    U34[8,col]=1;V34[8,col]=1;W34[2,col]=1;col+=1
    U34[8,col]=1;V34[9,col]=1;W34[5,col]=1;col+=1
    # Strassen Right
    U34[0,col]=1;U34[4,col]=1;V34[2,col]=1;V34[7,col]=1;W34[6,col]=1;W34[10,col]=1;col+=1
    U34[3,col]=1;U34[4,col]=1;V34[2,col]=1;W34[7,col]=1;W34[10,col]=-1;col+=1
    U34[0,col]=1;V34[3,col]=1;V34[7,col]=-1;W34[9,col]=1;W34[10,col]=1;col+=1
    U34[4,col]=1;V34[6,col]=1;V34[2,col]=-1;W34[6,col]=1;W34[7,col]=1;col+=1
    U34[0,col]=1;U34[1,col]=1;V34[7,col]=1;W34[6,col]=-1;W34[9,col]=1;col+=1
    U34[3,col]=1;U34[0,col]=-1;V34[2,col]=1;V34[3,col]=1;W34[10,col]=1;col+=1
    U34[1,col]=1;U34[4,col]=-1;V34[6,col]=1;V34[7,col]=1;W34[6,col]=1;col+=1
    # A12*B2R
    U34[2,col]=1;V34[10,col]=1;W34[6,col]=1;col+=1
    U34[2,col]=1;V34[11,col]=1;W34[9,col]=1;col+=1
    U34[5,col]=1;V34[10,col]=1;W34[7,col]=1;col+=1
    U34[5,col]=1;V34[11,col]=1;W34[10,col]=1;col+=1
    # A21*B1R
    U34[6,col]=1;V34[2,col]=1;W34[8,col]=1;col+=1
    U34[6,col]=1;V34[3,col]=1;W34[11,col]=1;col+=1
    U34[7,col]=1;V34[6,col]=1;W34[8,col]=1;col+=1
    U34[7,col]=1;V34[7,col]=1;W34[11,col]=1;col+=1
    # A22*B2R
    U34[8,col]=1;V34[10,col]=1;W34[8,col]=1;col+=1
    U34[8,col]=1;V34[11,col]=1;W34[11,col]=1;col+=1
    
    # Verify rank-34 solution
    # Tensor from U,V,W: sum_t U[:,t] outer V[:,t] outer W[:,t]
    reconstructed = np.tensordot(U34, V34, axes=1)  # wrong approach
    # Actually: T_recon[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    T_recon = np.einsum('at,bt,ct->abc', U34, V34, W34)
    assert np.allclose(T_recon, T), "Rank-34 solution verification failed"
    
    # Now try to find rank 33
    # Strategy: look for a linear combination of the 34 terms that gives zero
    # i.e., find lambda[0..33] not all zero such that sum_t lambda_t * (U_t ⊗ V_t ⊗ W_t) = 0
    # This is a system of 9*12*12 = 1296 equations in 34 unknowns
    # If the 34 rank-1 tensors are linearly dependent, we can eliminate one
    
    # Build the 1296x34 matrix
    M = np.zeros((n*m * m*p * p*n, 34))
    for t in range(34):
        M[:, t] = np.einsum('a,b,c->abc', U34[:,t], V34[:,t], W34[:,t]).flatten()
    
    # Find null space
    try:
        null_vec = np.linalg.svd(M, full_matrices=True)[2][-1]
        # Check if it's truly in the null space
        residual = np.linalg.norm(M @ null_vec)
        if residual < 1e-10:
            # We have a dependency! Find the largest |lambda_t| and eliminate that term
            max_idx = np.argmax(np.abs(null_vec))
            # New rank-33 solution: combine all terms except max_idx
            # U_new[:, t'] = U34[:, t'] for t' != max_idx, but we need to absorb
            # Actually: sum_t lambda_t * T_t = 0 means sum_{t != max} lambda_t * T_t = -lambda_max * T_max
            # So T_max = -(1/lambda_max) * sum_{t != max} lambda_t * T_t
            # New decomposition: for t != max_idx, use (1, ...) and for the eliminated term, 
            # we replace it by the linear combination. But that changes the structure.
            # Better: the 33 terms {T_t : t != max_idx} still span T if lambda_max != 0
            # because T = sum_{t!=max} T_t + T_max = sum_{t!=max} T_t - (1/lambda_max)*sum_{t!=max} lambda_t*T_t
            # = sum_{t!=max} (1 - lambda_t/lambda_max) * T_t
            # So we can just rescale the remaining 33 terms!
            scale = np.ones(33)
            for new_t, old_t in enumerate(range(34)):
                if old_t == max_idx:
                    continue
                scale[new_t] = 1.0 - null_vec[old_t] / null_vec[max_idx]
            
            U33 = U34[:, [t for t in range(34) if t != max_idx]] * scale
            V33 = V34[:, [t for t in range(34) if t != max_idx]] * scale
            W33 = W34[:, [t for t in range(34) if t != max_idx]] * scale
            
            # Verify
            T33 = np.einsum('at,bt,ct->abc', U33, V33, W33)
            if np.allclose(T33, T, atol=1e-10):
                # Round to nearest integer or half-integer
                U33 = np.round(U33 * 2) / 2
                V33 = np.round(V33 * 2) / 2
                W33 = np.round(W33 * 2) / 2
                T33r = np.einsum('at,bt,ct->abc', U33, V33, W33)
                if np.allclose(T33r, T, atol=1e-10):
                    return {"U": U33.tolist(), "V": V33.tolist(), "W": W33.tolist()}
                else:
                    # Use unrounded
                    return {"U": U33.tolist(), "V": V33.tolist(), "W": W33.tolist()}
    except:
        pass
    
    # If null space approach didn't work, try a different strategy
    # Try to find rank-33 by direct optimization
    # Use the structure: fix most terms, optimize a few
    
    # Alternative: try all pairs of terms to see if one can be expressed as combo of others
    # This is expensive, so limit iterations
    if time.time() > deadline - 20:
        # Fall back to rank 34
        return {"U": U34.tolist(), "V": V34.tolist(), "W": W34.tolist()}
    
    # Try a more systematic approach: perturb the solution and use ALS
    # Initialize rank-33 from rank-34 by dropping the term with smallest "importance"
    # Importance: norm of the term's contribution
    importances = np.array([np.linalg.norm(U34[:,t] * V34[:,t] * W34[:,t]) for t in range(34)])
    
    best_result = None
    for drop_idx in np.argsort(importances)[:5]:  # Try dropping the 5 least important terms
        if time.time() > deadline - 10:
            break
        
        U_init = U34[:, [t for t in range(34) if t != drop_idx]].copy()
        V_init = V34[:, [t for t in range(34) if t != drop_idx]].copy()
        W_init = W34[:, [t for t in range(34) if t != drop_idx]].copy()
        
        # ALS optimization
        U_opt = U_init.copy()
        V_opt = V_init.copy()
        W_opt = W_init.copy()
        
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each t, U[:,t] = (T contracted with V[:,t] and W[:,t]) / (V[:,t]·W[:,t] elementwise sum)
            for t in range(33):
                # T[a,b,c] = U[a,t]*V[b,t]*W[c,t]
                # U[a,t] = sum_{b,c} T[a,b,c]*V[b,t]*W[c,t] / sum_{b,c} V[b,t]^2*W[c,t]^2
                numerator = np.einsum('abc,bt,ct->a', T, V_opt[:,t], W_opt[:,t])
                denominator = np.sum(V_opt[:,t]**2 * W_opt[:,t]**2)
                if denominator > 1e-15:
                    U_opt[:,t] = numerator / denominator
            
            for t in range(33):
                numerator = np.einsum('abc,at,ct->b', T, U_opt[:,t], W_opt[:,t])
                denominator = np.sum(U_opt[:,t]**2 * W_opt[:,t]**2)
                if denominator > 1e-15:
                    V_opt[:,t] = numerator / denominator
            
            for t in range(33):
                numerator = np.einsum('abc,at,bt->c', T, U_opt[:,t], V_opt[:,t])
                denominator = np.sum(U_opt[:,t]**2 * V_opt[:,t]**2)
                if denominator > 1e-15:
                    W_opt[:,t] = numerator / denominator
            
            # Check convergence
            T_current = np.einsum('at,bt,ct->abc', U_opt, V_opt, W_opt)
            error = np.linalg.norm(T_current - T)
            if error < 1e-10:
                # Success! Round to half-integers
                U_r = np.round(U_opt * 2) / 2
                V_r = np.round(V_opt * 2) / 2
                W_r = np.round(W_opt * 2) / 2
                T_r = np.einsum('at,bt,ct->abc', U_r, V_r, W_r)
                if np.allclose(T_r, T, atol=1e-10):
                    best_result = (U_r, V_r, W_r)
                    break
                else:
                    best_result = (U_opt, V_opt, W_opt)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_res, V_res, W_res = best_result
        return {"U": U_res.tolist(), "V": V_res.tolist(), "W": W_res.tolist()}
    
    # Fall back to rank 34
    return {"U": U34.tolist(), "V": V34.tolist(), "W": W34.tolist()}
