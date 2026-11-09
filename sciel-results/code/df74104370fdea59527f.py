import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    def verify(U, V, W, n, m, p, tol=1e-6):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def standard_term(U, V, W, i, j, k, col):
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1

    def strassen_block(U, V, W, i1, i2, j1, j2, k1, k2, col):
        U[i1*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        U[i2*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1; W[k2*n+i2][col] += -1
        col += 1
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1; V[j2*p+k2][col] += -1
        W[k2*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1; V[j1*p+k1][col] += -1
        W[k1*n+i1][col] += 1; W[k1*n+i2][col] += 1
        col += 1
        U[i1*m+j1][col] += 1; U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += -1; W[k2*n+i1][col] += 1
        col += 1
        U[i2*m+j1][col] += 1; U[i1*m+j1][col] += -1
        V[j1*p+k1][col] += 1; V[j1*p+k2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        U[i1*m+j2][col] += 1; U[i2*m+j2][col] += -1
        V[j2*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        return col

    def build_rank92():
        U = np.zeros((nm, 92), dtype=np.float64)
        V = np.zeros((mp, 92), dtype=np.float64)
        W = np.zeros((pn, 92), dtype=np.float64)
        
        blocks = []
        for i_base in [0, 2]:
            for j_base in [0, 2]:
                for k_base in [0, 2]:
                    blocks.append((i_base, i_base+1, j_base, j_base+1, k_base, k_base+1))
        
        skip_set = set()
        for i1, i2, j1, j2, k1, k2 in blocks:
            for i in [i1, i2]:
                for j in [j1, j2]:
                    for k in [k1, k2]:
                        skip_set.add((i, j, k))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in skip_set:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        for i1, i2, j1, j2, k1, k2 in blocks:
            col = strassen_block(U.tolist(), V.tolist(), W.tolist(), i1, i2, j1, j2, k1, k2, col)
            U = np.array(U) if not isinstance(U, np.ndarray) else U
            V = np.array(V) if not isinstance(V, np.ndarray) else V
            W = np.array(W) if not isinstance(W, np.ndarray) else W
        
        return U, V, W, col

    # Build rank-92 solution
    U92, V92, W92, _ = build_rank92()
    
    # Verify rank-92
    if not verify(U92, V92, W92, n, m, p):
        # Fallback: standard rank-100
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-90 decomposition via optimization
    target_rank = 90
    U0 = U92[:, :target_rank].copy()
    V0 = V92[:, :target_rank].copy()
    W0 = W92[:, :target_rank].copy()
    
    # Flatten for optimization
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    nU = nm * target_rank
    nV = mp * target_rank
    nW = pn * target_rank
    
    def unpack(x):
        U = x[:nU].reshape(nm, target_rank)
        V = x[nU:nU+nV].reshape(mp, target_rank)
        W = x[nU+nV:].reshape(pn, target_rank)
        return U, V, W
    
    # Precompute index arrays for efficient tensor evaluation
    # Target tensor T[i,j,k] = 1 for all i,j,k
    # Decomposed tensor: D[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    def compute_tensor(U, V, W):
        """Compute the decomposed tensor D[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        D = np.zeros((n, m, p), dtype=np.float64)
        for r in range(target_rank):
            Ur = U[:, r]  # (nm,) = (n*m,)
            Vr = V[:, r]  # (mp,) = (m*p,)
            Wr = W[:, r]  # (pn,) = (p*n,)
            # Reshape
            Ur_mat = Ur.reshape(n, m)  # U[i*m+j] -> [i,j]
            Vr_mat = Vr.reshape(m, p)  # V[j*p+k] -> [j,k]
            Wr_mat = Wr.reshape(p, n)  # W[k*n+i] -> [k,i]
            # D[i,j,k] += Ur_mat[i,j] * Vr_mat[j,k] * Wr_mat[k,i]
            for i in range(n):
                for j in range(m):
                    uj = Ur_mat[i, j]
                    if uj == 0:
                        continue
                    for k in range(p):
                        D[i, j, k] += uj * Vr_mat[j, k] * Wr_mat[k, i]
        return D
    
    def objective_and_grad(x):
        U, V, W = unpack(x)
        
        # Compute D = sum_r U_r * V_r * W_r (outer product per rank)
        D = np.zeros((n, m, p), dtype=np.float64)
        for r in range(target_rank):
            Ur_mat = U[:, r].reshape(n, m)
            Vr_mat = V[:, r].reshape(m, p)
            Wr_mat = W[:, r].reshape(p, n)
            # D[i,j,k] += Ur_mat[i,j] * Vr_mat[j,k] * Wr_mat[k,i]
            # Efficient: for each r, D += einsum('ij,jk,ki->ijk', Ur_mat, Vr_mat, Wr_mat)
            D += np.einsum('ij,jk,ki->ijk', Ur_mat, Vr_mat, Wr_mat, optimize=True)
        
        # Residual
        R = D - 1.0  # target is all-ones tensor
        obj = 0.5 * np.sum(R * R)
        
        # Gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(target_rank):
            Ur_mat = U[:, r].reshape(n, m)
            Vr_mat = V[:, r].reshape(m, p)
            Wr_mat = W[:, r].reshape(p, n)
            
            # dD/dUr[i,j] = Vr_mat[j,k] * Wr_mat[k,i] summed over k
            # grad_U[r, i*m+j] = sum_k R[i,j,k] * Vr_mat[j,k] * Wr_mat[k,i]
            # = einsum('ijk,jk,ki->ij', R, Vr_mat, Wr_mat)
            gUr = np.einsum('ijk,jk,ki->ij', R, Vr_mat, Wr_mat, optimize=True)
            grad_U[:, r] = gUr.flatten()
            
            # dD/dVr[j,k] = Ur_mat[i,j] * Wr_mat[k,i] summed over i
            # grad_V[r, j*p+k] = sum_i R[i,j,k] * Ur_mat[i,j] * Wr_mat[k,i]
            gVr = np.einsum('ijk,ij,ki->jk', R, Ur_mat, Wr_mat, optimize=True)
            grad_V[:, r] = gVr.flatten()
            
            # dD/dWr[k,i] = Ur_mat[i,j] * Vr_mat[j,k] summed over j
            # grad_W[r, k*n+i] = sum_j R[i,j,k] * Ur_mat[i,j] * Vr_mat[j,k]
            gWr = np.einsum('ijk,ij,jk->ki', R, Ur_mat, Vr_mat, optimize=True)
            grad_W[:, r] = gWr.flatten()
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj, grad
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time for optimization, return rank-92
        return {"U": U92.tolist(), "V": V92.tolist(), "W": W92.tolist()}
    
    # Run optimization
    best_x = x0.copy()
    best_obj = objective_and_grad(x0)[0]
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
        
        # Check if we got close enough
        if best_obj < 1e-10:
            U_opt, V_opt, W_opt = unpack(best_x)
            if verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        
    except Exception:
        pass
    
    # If optimization didn't succeed, return rank-92
    return {"U": U92.tolist(), "V": V92.tolist(), "W": W92.tolist()}
