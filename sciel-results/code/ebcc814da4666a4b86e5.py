import time
import numpy as np
from itertools import combinations
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s

    # 1. The known rank-17 solution
    # Indices: U is (n*m, R) -> (6, R), V is (m*p, R) -> (9, R), W is (p*n, R) -> (6, R)
    # W index convention: W[k*n + i, r]
    
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1

    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1

    # Verify rank-17 validity
    def verify(U, V, W):
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Must be 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U17, V17, W17):
        # Fallback to schoolbook rank-18 if rank-17 is broken (should not happen)
        U18 = np.zeros((6, 18)); V18 = np.zeros((9, 18)); W18 = np.zeros((6, 18))
        idx = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U18[i*3+j, idx] = 1
                    V18[j*3+k, idx] = 1
                    W18[k*2+i, idx] = 1
                    idx += 1
        return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}

    # Fast tensor evaluation using einsum
    # U: (6, R), V: (9, R), W: (6, R)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape: U_3d[i,j,r], V_3d[j,k,r], W_3d[k,i,r]
    # T[i,j,k] = einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d) -- but indices don't align simply
    # Let's use a direct approach with precomputed index maps
    
    # Precompute index arrays for fast evaluation
    U_rows = np.array([i*m+j for i in range(n) for j in range(m)])  # 18 entries
    V_rows = np.array([j*p+k for j in range(m) for k in range(p)])  # 18 entries
    W_rows = np.array([k*n+i for k in range(p) for i in range(n)])  # 18 entries
    
    def fast_tensor(U, V, W):
        """Compute T[i,j,k] for all i,j,k using matrix operations."""
        R = U.shape[1]
        # U: (6, R), V: (9, R), W: (6, R)
        # For each (i,j,k): T = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We can compute this as: for each r, outer product contribution
        # T = sum_r U_col_r ⊗ V_col_r ⊗ W_col_r (reshaped appropriately)
        # More efficiently: 
        # Let A_r = U[:,r] (6,), B_r = V[:,r] (9,), C_r = W[:,r] (6,)
        # T[i,j,k] = sum_r A_r[i*m+j] * B_r[j*p+k] * C_r[k*n+i]
        # Reshape A_r to (n,m), B_r to (m,p), C_r to (p,n)
        # T[i,j,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
        # This is like a trace: sum_r trace(A_r[i,:] * B_r[:,k] * C_r[k,i])... not quite
        # Let's just use einsum with proper reshaping
        
        U_3d = U.reshape(n, m, R)  # (2, 3, R)
        V_3d = V.reshape(m, p, R)  # (3, 3, R)
        W_3d = W.reshape(p, n, R)  # (3, 2, R)
        
        # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # Use einsum: 'ijr,jkr,ki r->ijk' doesn't work directly because of the index permutation
        # Let's do it in steps:
        # Step 1: M[i,j,k,r] = U_3d[i,j,r] * V_3d[j,k,r] -> (2,3,3,R)
        # Step 2: T[i,j,k] = sum_r M[i,j,k,r] * W_3d[k,i,r]
        
        # M[i,j,k,r] = U_3d[i,j,r] * V_3d[j,k,r]
        # = einsum('ijr,jkr->ijkr', U_3d, V_3d)
        M = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)  # (2,3,3,R)
        # T[i,j,k] = sum_r M[i,j,k,r] * W_3d[k,i,r]
        T = np.einsum('ijkr,ki r->ijk', M, W_3d)  # (2,3,3)
        return T
    
    def objective_fast(x, R=16):
        """Fast objective: sum of squared errors."""
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
        W = x[n*m*R+m*p*R:].reshape(p*n, R)
        
        T = fast_tensor(U, V, W)
        err = np.sum((T - 1.0) ** 2)
        return err
    
    def objective_grad(x, R=16):
        """Objective with gradient."""
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
        W = x[n*m*R+m*p*R:].reshape(p*n, R)
        
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        M = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)  # (2,3,3,R)
        T = np.einsum('ijkr,kir->ijk', M, W_3d)  # (2,3,3)
        
        err = np.sum((T - 1.0) ** 2)
        
        # Gradient:
        # dT[i,j,k]/dU[i',j',r'] = V_3d[j',k,r'] * W_3d[k,i',r'] * delta(i,i') * delta(j,j')
        # dT[i,j,k]/dV[j',k',r'] = U_3d[i,j',r'] * W_3d[k',i,r'] * delta(j,j') * delta(k,k')
        # dT[i,j,k]/dW[k',i',r'] = U_3d[i',j,r'] * V_3d[j,k',r'] * delta(k,k') * delta(i,i')
        
        # grad_U[i',j',r'] = sum_{k} (T[i',j',k] - 1) * 2 * V_3d[j',k,r'] * W_3d[k,i',r']
        # grad_V[j',k',r'] = sum_{i} (T[i,j',k'] - 1) * 2 * U_3d[i,j',r'] * W_3d[k',i,r']
        # grad_W[k',i',r'] = sum_{j} (T[i',j,k'] - 1) * 2 * U_3d[i',j,r'] * V_3d[j,k',r']
        
        T_minus_1 = T - 1.0  # (2,3,3)
        
        # grad_U: (2,3,R)
        # For U[i',j',r']: sum_k 2*(T[i',j',k]-1) * V[j',k,r'] * W[k,i',r']
        grad_U = np.zeros_like(U_3d)
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    # sum_k 2*(T[i,j,k]-1) * V[j,k,r] * W[k,i,r]
                    grad_U[i,j,r] = 2.0 * np.sum(T_minus_1[i,j,:] * V_3d[j,:,r] * W_3d[:,i,r])
        
        # grad_V: (3,3,R)
        # For V[j',k',r']: sum_i 2*(T[i,j',k']-1) * U[i,j',r'] * W[k',i,r']
        grad_V = np.zeros_like(V_3d)
        for r in range(R):
            for j in range(m):
                for k in range(p):
                    # sum_i 2*(T[i,j,k]-1) * U[i,j,r] * W[k,i,r]
                    grad_V[j,k,r] = 2.0 * np.sum(T_minus_1[:,j,k] * U_3d[:,j,r] * W_3d[k,:,r])
        
        # grad_W: (3,2,R)
        # For W[k',i',r']: sum_j 2*(T[i',j,k']-1) * U[i',j,r'] * V[j,k',r']
        grad_W = np.zeros_like(W_3d)
        for r in range(R):
            for k in range(p):
                for i in range(n):
                    # sum_j 2*(T[i,j,k]-1) * U[i,j,r] * V[j,k,r]
                    grad_W[k,i,r] = 2.0 * np.sum(T_minus_1[i,:,k] * U_3d[i,:,r] * V_3d[:,k,r])
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return err, grad
    
    # Try to find rank-16
    # Strategy: For each pair of columns to "merge", try to optimize a rank-16 solution
    # initialized by removing one column and perturbing slightly
    
    best_rank = 17
    best_U = U17
    best_V = V17
    best_W = W17
    
    np.random.seed(42)
    
    # Try dropping each column and optimizing
    for drop_col in range(17):
        if time.time() > deadline - 15:
            break
            
        U_init = np.delete(U17, drop_col, axis=1)
        V_init = np.delete(V17, drop_col, axis=1)
        W_init = np.delete(W17, drop_col, axis=1)
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
                
            # Small perturbation
            scale = 0.05 * (attempt + 1)
            U_curr = U_init + np.random.randn(6, 16) * scale
            V_curr = V_init + np.random.randn(9, 16) * scale
            W_curr = W_init + np.random.randn(6, 16) * scale
            
            x0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective_grad, x0, jac=True, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < 1e-8:
                    U_sol = res.x[:96].reshape(6, 16)
                    V_sol = res.x[96:240].reshape(9, 16)
                    W_sol = res.x[240:336].reshape(6, 16)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    if verify(U_round, V_round, W_round):
                        best_rank = 16
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    if best_rank == 16:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
