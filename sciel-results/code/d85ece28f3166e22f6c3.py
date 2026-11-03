import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Trivial schoolbook decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    # Build the target tensor
    def make_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T
    
    T = make_tensor(n, m, p)
    
    def verify(U, V, W, T, tol=1e-8):
        R = U.shape[1]
        try:
            Temp = np.einsum('ur,vr->uvr', U, V)
            T_approx = np.einsum('uvr,wr->uvw', Temp, W)
            return np.allclose(T_approx, T, atol=tol)
        except:
            return False
    
    # Try structural constructions with reduced rank
    # Key insight: For row i of A, variables A[i,0..2] multiply with all 6 B columns
    # We can share terms across B columns that use the same A-row
    
    # Attempt 1: Block-structured decomposition
    # For each i (row of A), create m terms for A[i,j] that cover all p B-columns
    # This gives n*m*p terms = 36, same as trivial
    # But we can try to merge terms where possible
    
    # Attempt 2: Use the fact that B has 3 rows, each with 6 columns
    # B[j,0..5] for j=0,1,2
    # A[i,j] * B[j,k] for all k
    # If we fix i and j, we have A[i,j] * [B[j,0], B[j,1], ..., B[j,5]]
    # This is a rank-1 tensor: e_{i,j}^A ⊗ (e_{j,0}+...+e_{j,5})^B ⊗ ...
    # No, that's not right. Each B[j,k] is a separate variable.
    
    # Let me think differently. The tensor is:
    # T = Σ_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the unit vector for A[i,j], etc.
    
    # Can we combine terms? For fixed i,j:
    # Σ_k e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # = e_{i,j} ⊗ (Σ_k e_{j,k}) ⊗ ... no, e_{k,i} depends on k
    
    # For fixed i:
    # Σ_{j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # The e_{k,i} term is the same for all j when k is fixed.
    # So: Σ_k (Σ_j e_{i,j} ⊗ e_{j,k}) ⊗ e_{k,i}
    # = Σ_k (row i of A projected onto column k of B) ⊗ e_{k,i}
    
    # This suggests a decomposition where for each (i,k), we have a term
    # involving the j-index summed out. But that's not rank-1.
    
    # Let's try a different approach: randomized search with structured initialization
    
    def try_rank(R, max_starts=20, time_per_start=2.0):
        """Try to find a rank-R decomposition"""
        nm, mp, pn = n*m, m*p, p*n
        start_time = time.time()
        
        for s in range(max_starts):
            if time.time() > deadline - 10:
                break
            if time.time() - start_time > time_per_start * max_starts:
                break
            
            # Structured initialization: start from trivial and try to merge
            # Use a sparse initialization
            U = np.random.uniform(-0.5, 0.5, (nm, R))
            V = np.random.uniform(-0.5, 0.5, (mp, R))
            W = np.random.uniform(-0.5, 0.5, (pn, R))
            
            # Local optimization using gradient descent
            lr = 0.01
            best_err = np.inf
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute error
                Temp = np.einsum('ur,vr->uvr', U, V)
                T_approx = np.einsum('uvr,wr->uvw', Temp, W)
                err = np.sum((T_approx - T)**2)
                
                if err < best_err:
                    best_err = err
                    U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    break
                
                # Compute gradients
                # dT_approx/dU = V^T ⊗ W (broadcasted)
                # dT_approx/dV = U^T ⊗ W
                # dT_approx/dW = (U⊗V)^T * (T_approx - T)
                
                # Gradient of W: for each r, dW[:,r] = sum_{u,v} (T_approx[u,v,r] - T[u,v]) * U[u,r] * V[v,r]
                # Actually: T_approx[u,v] = sum_r U[u,r]*V[v,r]*W[:,r]
                # d/dW[c,r] = sum_{u,v} (T_approx[u,v] - T[u,v]) * U[u,r] * V[v,r] * (c == k*n+i where...)
                
                # Let me use a simpler approach: compute via einsum
                # T_approx[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
                # dL/dW[w,r] = sum_{u,v} (T_approx[u,v,w] - T[u,v,w]) * U[u,r] * V[v,r]
                
                diff = T_approx - T  # shape (nm, mp, pn)
                
                # dW[w,r] = sum_{u,v} diff[u,v,w] * U[u,r] * V[v,r]
                # = einsum('uvw,ur,vr->wr', diff, U, V)
                dW = np.einsum('uvw,ur,vr->wr', diff, U, V)
                
                # dU[u,r] = sum_{v,w} diff[u,v,w] * V[v,r] * W[w,r]
                dU = np.einsum('uvw,vr,wr->ur', diff, V, W)
                
                # dV[v,r] = sum_{u,w} diff[u,v,w] * U[u,r] * W[w,r]
                dV = np.einsum('uvw,ur,wr->vr', diff, U, W)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if it % 20 == 19 and err > 1e-6:
                    lr *= 0.5
                
                if err < 1e-8:
                    break
            
            # Check if we found a valid solution
            if best_err < 1e-8:
                # Try to snap to integers or half-integers
                for factor in [1, 2, 4]:
                    U_round = np.round(U_best * factor) / factor
                    V_round = np.round(V_best * factor) / factor
                    W_round = np.round(W_best * factor) / factor
                    if verify(U_round, V_round, W_round, T, tol=1e-6):
                        return U_round, V_round, W_round
                
                if verify(U_best, V_best, W_best, T, tol=1e-6):
                    return U_best, V_best, W_best
        
        return None
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But the actual rank might be higher
    
    # Try some candidate ranks
    candidate_ranks = [18, 20, 22, 24, 26, 28, 30, 32, 34]
    
    for R in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        result = try_rank(R, max_starts=10, time_per_start=1.5)
        if result:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: return trivial decomposition
    return {
        "U": U_trivial,
        "V": V_trivial,
        "W": W_trivial
    }
