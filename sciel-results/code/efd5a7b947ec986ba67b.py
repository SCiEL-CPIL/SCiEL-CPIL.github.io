"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    target = np.ones((n, m, p))
    
    # Time guard
    t0 = time.time()
    deadline = 130.0  # 130s out of 150s limit
    
    # Helper functions
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
    
    # Build rank-26 decomposition (Strassen + standard)
    R26 = 26
    U26 = np.zeros((n * m, R26))
    V26 = np.zeros((m * p, R26))
    W26 = np.zeros((p * n, R26))
    
    # Strassen 7 terms for 2x2 core
    # m1 = (a00+a11)(b00+b11) -> C00, C11
    r = 0
    U26[idx_u(0,0), r] = 1; U26[idx_u(1,1), r] = 1
    V26[idx_v(0,0), r] = 1; V26[idx_v(1,1), r] = 1
    W26[idx_w(0,0), r] = 1; W26[idx_w(1,1), r] = 1
    
    # m2 = (a10+a11)b00 -> C10, C11
    r = 1
    U26[idx_u(1,0), r] = 1; U26[idx_u(1,1), r] = 1
    V26[idx_v(0,0), r] = 1
    W26[idx_w(0,1), r] = 1; W26[idx_w(1,1), r] = -1
    
    # m3 = a00(b01-b11) -> C01, C11
    r = 2
    U26[idx_u(0,0), r] = 1
    V26[idx_v(0,1), r] = 1; V26[idx_v(1,1), r] = -1
    W26[idx_w(1,0), r] = 1; W26[idx_w(1,1), r] = 1
    
    # m4 = a11(b10-b00) -> C00, C10
    r = 3
    U26[idx_u(1,1), r] = 1
    V26[idx_v(1,0), r] = 1; V26[idx_v(0,0), r] = -1
    W26[idx_w(0,0), r] = 1; W26[idx_w(0,1), r] = 1
    
    # m5 = (a00+a01)b11 -> C00, C01
    r = 4
    U26[idx_u(0,0), r] = 1; U26[idx_u(0,1), r] = 1
    V26[idx_v(1,1), r] = 1
    W26[idx_w(0,0), r] = -1; W26[idx_w(1,0), r] = 1
    
    # m6 = (a10-a00)(b00+b01) -> C11
    r = 5
    U26[idx_u(1,0), r] = 1; U26[idx_u(0,0), r] = -1
    V26[idx_v(0,0), r] = 1; V26[idx_v(0,1), r] = 1
    W26[idx_w(1,1), r] = 1
    
    # m7 = (a01-a11)(b10+b11) -> C00
    r = 6
    U26[idx_u(0,1), r] = 1; U26[idx_u(1,1), r] = -1
    V26[idx_v(1,0), r] = 1; V26[idx_v(1,1), r] = 1
    W26[idx_w(0,0), r] = 1
    
    # Standard 19 terms for border
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
    
    def reconstruct(U, V, W, R):
        """Reconstruct tensor from factor matrices."""
        Ur = U.reshape((n, m, R))
        Vr = V.reshape((m, p, R))
        Wr = W.reshape((p, n, R))
        return np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    
    def objective(params, R):
        """Objective: ||T - U⊗V⊗W||_F^2"""
        size_u = n * m
        size_v = m * p
        size_w = p * n
        U = params[:size_u*R].reshape((size_u, R))
        V = params[size_u*R : size_u*R + size_v*R].reshape((size_v, R))
        W = params[size_u*R + size_v*R :].reshape((size_w, R))
        T_recon = reconstruct(U, V, W, R)
        diff = T_recon - target
        return np.sum(diff * diff)
    
    def try_rank(R, U_init, V_init, W_init, max_iters=5000, tol=1e-10):
        """Try to optimize to rank R decomposition."""
        if time.time() > deadline:
            return None
        
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        # Flatten initial params
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use gradient descent with momentum
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(params)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute gradient
            U = params[:size_u*R].reshape((size_u, R))
            V = params[size_u*R : size_u*R + size_v*R].reshape((size_v, R))
            W = params[size_u*R + size_v*R :].reshape((size_w, R))
            
            Ur = U.reshape((n, m, R))
            Vr = V.reshape((m, p, R))
            Wr = W.reshape((p, n, R))
            
            T_recon = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
            diff = T_recon - target  # shape (n, m, p)
            
            # Gradients
            # dU[i,j,r] = sum_{k} diff[i,j,k] * V[j,k,r] * W[k,i,r]
            # dV[j,k,r] = sum_{i} diff[i,j,k] * U[i,j,r] * W[k,i,r]
            # dW[k,i,r] = sum_{j} diff[i,j,k] * U[i,j,r] * V[j,k,r]
            
            dU = np.einsum('ijk,jkr,kir->ijr', diff, Vr, Wr)
            dV = np.einsum('ijk,ijr,kir->jkr', diff, Ur, Wr)
            dW = np.einsum('ijk,ijr,jkr->kir', diff, Ur, Vr)
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            # Momentum update
            v = momentum * v - lr * grad
            params = params + v
            
            # Check convergence
            if it % 100 == 0:
                obj = objective(params, R)
                if obj < tol:
                    break
        
        # Extract final params
        U_opt = params[:size_u*R].reshape((size_u, R))
        V_opt = params[size_u*R : size_u*R + size_v*R].reshape((size_v, R))
        W_opt = params[size_u*R + size_v*R :].reshape((size_w, R))
        
        return U_opt, V_opt, W_opt
    
    def round_and_verify(U, V, W, R):
        """Round to half-integers and verify exactness."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_recon = reconstruct(U_r, V_r, W_r, R)
        if np.allclose(T_recon, target, atol=1e-9):
            return U_r, V_r, W_r, True
        return None, None, None, False
    
    # Try ranks from 23 upward
    best_result = None
    
    for R in [23, 24, 25]:
        if time.time() > deadline:
            break
        
        # Initialize by taking first R columns of rank-26
        U_init = U26[:, :R].copy()
        V_init = V26[:, :R].copy()
        W_init = W26[:, :R].copy()
        
        # Add small random perturbation to break symmetry
        np.random.seed(42)
        U_init += np.random.randn(n*m, R) * 0.1
        V_init += np.random.randn(m*p, R) * 0.1
        W_init += np.random.randn(p*n, R) * 0.1
        
        result = try_rank(R, U_init, V_init, W_init)
        if result is None:
            continue
        
        U_opt, V_opt, W_opt = result
        U_r, V_r, W_r, valid = round_and_verify(U_opt, V_opt, W_opt, R)
        
        if valid:
            best_result = (R, U_r, V_r, W_r)
            break
    
    # If no low-rank solution found, try more random inits for rank 23
    if best_result is None and time.time() < deadline - 20:
        for R in [23, 24]:
            if time.time() > deadline:
                break
            for seed in range(5):
                if time.time() > deadline:
                    break
                np.random.seed(seed * 1000 + 7)
                U_init = U26[:, :R].copy() + np.random.randn(n*m, R) * 0.5
                V_init = V26[:, :R].copy() + np.random.randn(m*p, R) * 0.5
                W_init = W26[:, :R].copy() + np.random.randn(p*n, R) * 0.5
                
                result = try_rank(R, U_init, V_init, W_init, max_iters=3000)
                if result is None:
                    continue
                U_opt, V_opt, W_opt = result
                U_r, V_r, W_r, valid = round_and_verify(U_opt, V_opt, W_opt, R)
                if valid:
                    best_result = (R, U_r, V_r, W_r)
                    break
            if best_result is not None:
                break
    
    # Fallback: rank 26
    if best_result is None:
        R = 26
        U = U26
        V = V26
        W = W26
    else:
        R, U, V, W = best_result
    
    # Final verification
    T_recon = reconstruct(U, V, W, R)
    if not np.allclose(T_recon, target, atol=1e-9):
        # Ultimate fallback: schoolbook rank 27
        R = 27
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i,j), col] = 1
                    V[idx_v(j,k), col] = 1
                    W[idx_w(k,i), col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
