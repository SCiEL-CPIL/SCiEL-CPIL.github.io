"""Alternative strategy: Kronecker composition of two 2x2 factorizations + ALS refinement."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, tol=1e-9):
    """Verify exact 4x4 MM decomposition with W[k*n+i, r] convention."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
    target = np.ones((n, m, p))
    return np.max(np.abs(recon - target)) < tol

def _build_2x2_strassen():
    """Return (U2, V2, W2) for 2x2 MM, rank 7, with W[k*2+i, r] convention.
    Verified: sum_r U2[i,j,r]*V2[j,k,r]*W2[k,i,r] = 1 for all i,j,k in 0..1."""
    # Strassen 2x2:
    # m0=(a00+a11)(b00+b11), m1=(a10+a11)b00, m2=a00(b10-b11), m3=a11(b01-b00)
    # m4=(a00+a01)b11, m5=(a10-a00)(b00+b01), m6=(a01-a11)(b10+b11)
    # C00=m0+m3-m4+m6, C01=m2+m4, C10=m1+m3, C11=m0-m1+m2+m5
    # A flat: a00=0, a01=1, a10=2, a11=3
    # B flat: b00=0, b01=1, b10=2, b11=3
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))  # W2[k*2+i, r]
    
    # U2[i*2+j, q]
    # q=0: a00+a11 -> U2[0,0]=1, U2[3,0]=1
    # q=1: a10+a11 -> U2[2,1]=1, U2[3,1]=1
    # q=2: a00 -> U2[0,2]=1
    # q=3: a11 -> U2[3,3]=1
    # q=4: a00+a01 -> U2[0,4]=1, U2[1,4]=1
    # q=5: a10-a00 -> U2[2,5]=1, U2[0,5]=-1
    # q=6: a01-a11 -> U2[1,6]=1, U2[3,6]=-1
    U2[0,0]=1; U2[3,0]=1
    U2[2,1]=1; U2[3,1]=1
    U2[0,2]=1
    U2[3,3]=1
    U2[0,4]=1; U2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1
    U2[1,6]=1; U2[3,6]=-1
    
    # V2[j*2+k, q]
    # q=0: b00+b11 -> V2[0,0]=1, V2[3,0]=1
    # q=1: b00 -> V2[0,1]=1
    # q=2: b10-b11 -> V2[2,2]=1, V2[3,2]=-1
    # q=3: b01-b00 -> V2[1,3]=1, V2[0,3]=-1
    # q=4: b11 -> V2[3,4]=1
    # q=5: b00+b01 -> V2[0,5]=1, V2[1,5]=1
    # q=6: b10+b11 -> V2[2,6]=1, V2[3,6]=1
    V2[0,0]=1; V2[3,0]=1
    V2[0,1]=1
    V2[2,2]=1; V2[3,2]=-1
    V2[1,3]=1; V2[0,3]=-1
    V2[3,4]=1
    V2[0,5]=1; V2[1,5]=1
    V2[2,6]=1; V2[3,6]=1
    
    # W2[k*2+i, q]: C[i,k] = sum_q W2[k*2+i, q] * m_q
    # C00 = m0+m3-m4+m6: W2[0*2+0, q] for k=0,i=0 -> W2[0, q]
    #   q: 0->1, 3->1, 4->-1, 6->1
    W2[0,0]=1; W2[0,3]=1; W2[0,4]=-1; W2[0,6]=1
    # C01 = m2+m4: k=1,i=0 -> W2[1, q]
    W2[1,2]=1; W2[1,4]=1
    # C10 = m1+m3: k=0,i=1 -> W2[2, q]
    W2[2,1]=1; W2[2,3]=1
    # C11 = m0-m1+m2+m5: k=1,i=1 -> W2[3, q]
    W2[3,0]=1; W2[3,1]=-1; W2[3,2]=1; W2[3,5]=1
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                assert abs(val - 1.0) < 1e-12, f"2x2 verify failed at {i},{j},{k}: {val}"
    
    return U2, V2, W2

def _kronecker_compose(U2, V2, W2):
    """Compose two 2x2 factorizations via Kronecker product to get 4x4 factorization.
    Indexing: i=2*br+i2, j=2*bc+j2, k=2*kc+k2
    U4[i*4+j, q*7+s] = U2[br*2+i2, q] * U2[bc*2+j2, s]
    V4[j*4+k, q*7+s] = V2[kc*2+j2, q] * V2[kc*2+k2, s]  -- wait, need to be careful
    Actually: V4[j*4+k, q*7+s] = V2[bc*2+j2, q] * V2[kc*2+k2, s]
    Hmm, let me think about this more carefully.
    
    The 4x4 MM: C[i,k] = sum_j A[i,j] B[j,k]
    With i=2*br+i2, j=2*bc+j2, k=2*kc+k2:
    C[br,i2; kc,k2] = sum_{bc,j2} A[br,i2; bc,j2] B[bc,j2; kc,k2]
    
    The Kronecker/tensor product of two 2x2 MM tensors:
    T4[br,i2; bc,j2; kc,k2] = T2[br; bc; kc] * T2[i2; j2; k2]
    
    Factorization of T2: T2[a; b; c] = sum_r U2[a*2+b, r] V2[b*2+c, r] W2[c*2+a, r]
    
    So T4[br,i2; bc,j2; kc,k2] = sum_{q,s} U2[br*2+bc, q] V2[bc*2+kc, q] W2[kc*2+br, q]
                                  * U2[i2*2+j2, s] V2[j2*2+k2, s] W2[k2*2+i2, s]
    
    Therefore:
    U4[i*4+j, q*7+s] = U2[br*2+bc, q] * U2[i2*2+j2, s]
    V4[j*4+k, q*7+s] = V2[bc*2+kc, q] * V2[j2*2+k2, s]
    W4[k*4+i, q*7+s] = W2[kc*2+br, q] * W2[k2*2+i2, s]
    
    where i=2*br+i2, j=2*bc+j2, k=2*kc+k2
    """
    n = 4
    rank = 49
    U4 = np.zeros((n*n, rank))
    V4 = np.zeros((n*n, rank))
    W4 = np.zeros((n*n, rank))
    
    for br in range(2):
        for i2 in range(2):
            for bc in range(2):
                for j2 in range(2):
                    i = 2*br + i2
                    j = 2*bc + j2
                    for q in range(7):
                        for s in range(7):
                            col = q*7 + s
                            U4[i*4+j, col] = U2[br*2+bc, q] * U2[i2*2+j2, s]
    
    for bc in range(2):
        for j2 in range(2):
            for kc in range(2):
                for k2 in range(2):
                    j = 2*bc + j2
                    k = 2*kc + k2
                    for q in range(7):
                        for s in range(7):
                            col = q*7 + s
                            V4[j*4+k, col] = V2[bc*2+kc, q] * V2[j2*2+k2, s]
    
    for kc in range(2):
        for k2 in range(2):
            for br in range(2):
                for i2 in range(2):
                    k = 2*kc + k2
                    i = 2*br + i2
                    for q in range(7):
                        for s in range(7):
                            col = q*7 + s
                            W4[k*4+i, col] = W2[kc*2+br, q] * W2[k2*2+i2, s]
    
    return U4, V4, W4

def _als_refine(U, V, W, n_iter=30, tol=1e-10):
    """Alternating least squares refinement for the 4x4 MM tensor decomposition.
    Minimizes sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
    by cycling through U, V, W updates (each is a linear LS problem)."""
    n, m, p = 4, 4, 4
    rank = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    target = np.ones((n, m, p))
    
    for it in range(n_iter):
        # Update U: for each (i,j), minimize sum_k (sum_r U4[i,j,r]*V4[j,k,r]*W4[k,i,r] - 1)^2
        # This is: U4[i,j,:] @ (V4[j,:,:] * W4[:,i,:]).T @ U4[i,j,:] - 2*target[i,j,k]*... 
        # Actually: let B[i,j,k] = V4[j,k,:] * W4[k,i,:]  (element-wise product over r)
        # Then recon[i,j,k] = sum_r U4[i,j,r] * B[i,j,k,r]
        # Minimize sum_k (recon[i,j,k] - 1)^2 over U4[i,j,:]
        # This is a linear system: (sum_k outer(B[i,j,k], B[i,j,k])) @ U4[i,j,:] = sum_k B[i,j,k]
        
        for i in range(n):
            for j in range(m):
                # Build the normal equations
                A_mat = np.zeros((rank, rank))
                b_vec = np.zeros(rank)
                for k in range(p):
                    b_vec_k = V4[j, k, :] * W4[k, i, :]
                    A_mat += np.outer(b_vec_k, b_vec_k)
                    b_vec += b_vec_k
                # Regularize slightly
                A_mat += 1e-12 * np.eye(rank)
                try:
                    U4[i, j, :] = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    U4[i, j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        
        U = U4.reshape(n*m, rank)
        
        # Update V: for each (j,k), minimize sum_i (sum_r U4[i,j,r]*V4[j,k,r]*W4[k,i,r] - 1)^2
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((rank, rank))
                b_vec = np.zeros(rank)
                for i in range(n):
                    b_vec_k = U4[i, j, :] * W4[k, i, :]
                    A_mat += np.outer(b_vec_k, b_vec_k)
                    b_vec += b_vec_k
                A_mat += 1e-12 * np.eye(rank)
                try:
                    V4[j, k, :] = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    V4[j, k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        
        V = V4.reshape(m*p, rank)
        
        # Update W: for each (k,i), minimize sum_j (sum_r U4[i,j,r]*V4[j,k,r]*W4[k,i,r] - 1)^2
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((rank, rank))
                b_vec = np.zeros(rank)
                for j in range(m):
                    b_vec_k = U4[i, j, :] * V4[j, k, :]
                    A_mat += np.outer(b_vec_k, b_vec_k)
                    b_vec += b_vec_k
                A_mat += 1e-12 * np.eye(rank)
                try:
                    W4[k, i, :] = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    W4[k, i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        
        W = W4.reshape(p*n, rank)
        
        # Check convergence
        recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
        err = np.max(np.abs(recon - target))
        if err < tol:
            break
    
    return U, V, W

def _try_lower_rank_gradient(rank, time_budget, n_restarts=5):
    """Try to find a valid decomposition at the given rank using gradient descent."""
    n, m, p = 4, 4, 4
    deadline = time.time() + time_budget
    best_err = float('inf')
    best_solution = None
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        U4 = U.reshape(n, m, rank)
        V4 = V.reshape(m, p, rank)
        W4 = W.reshape(p, n, rank)
        target = np.ones((n, m, p))
        
        lr = 0.05
        best_local_err = float('inf')
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
            
            err = np.sum((recon - target)**2)
            if err < best_local_err:
                best_local_err = err
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
            
            if best_local_err < 1e-8:
                break
            
            # Gradients
            dU4 = np.zeros((n, m, rank))
            dV4 = np.zeros((m, p, rank))
            dW4 = np.zeros((p, n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = recon[i, j, k] - 1.0
                        for r in range(rank):
                            dU4[i, j, r] += 2*diff*V4[j, k, r]*W4[k, i, r]
                            dV4[j, k, r] += 2*diff*U4[i, j, r]*W4[k, i, r]
                            dW4[k, i, r] += 2*diff*U4[i, j, r]*V4[j, k, r]
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU4**2) + np.sum(dV4**2) + np.sum(dW4**2))
            if grad_norm > 1e-12:
                lr_eff = min(lr, 0.1 / grad_norm)
            else:
                lr_eff = lr
            
            U4 -= lr_eff * dU4
            V4 -= lr_eff * dV4
            W4 -= lr_eff * dW4
        
        if best_err < 1e-9:
            break
    
    return best_err, best_solution

def _schoolbook():
    """Rank-64 schoolbook decomposition (always valid)."""
    n, m, p = 4, 4, 4
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    total_budget = 140.0  # Leave 10s safety margin for 150s limit
    
    # Step 1: Build 2x2 Strassen and compose via Kronecker product
    U2, V2, W2 = _build_2x2_strassen()
    U4, V4, W4 = _kronecker_compose(U2, V2, W2)
    
    # Verify the Kronecker composition
    if not _verify(U4, V4, W4):
        # Fallback to schoolbook
        U_fb, V_fb, W_fb = _schoolbook()
        return {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}
    
    best_U, best_V, best_W = U4, V4, W4
    best_rank = 49
    
    # Step 2: Try ALS refinement (should maintain or improve validity)
    remaining = total_budget - (time.time() - start_time)
    if remaining > 10:
        U_ref, V_ref, W_ref = _als_refine(U4.copy(), V4.copy(), W4.copy(), n_iter=20)
        if _verify(U_ref, V_ref, W_ref):
            best_U, best_V, best_W = U_ref, V_ref, W_ref
    
    # Step 3: Try to find lower-rank decompositions
    remaining = total_budget - (time.time() - start_time)
    if remaining > 20:
        # Try ranks below 49
        for rank in [32, 36, 40, 44, 48]:
            if time.time() - start_time > total_budget - 15:
                break
            rank_budget = min(remaining - 10, 25.0)
            if rank_budget < 5:
                break
            err, sol = _try_lower_rank_gradient(rank, rank_budget, n_restarts=3)
            if err < 1e-9 and sol is not None:
                best_U, best_V, best_W = sol
                best_rank = rank
                break
    
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}

# EVOLVE-BLOCK-END
