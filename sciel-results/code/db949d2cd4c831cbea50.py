import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    n, m, p = 3, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 12, 16, 12

    # Schoolbook baseline: rank 48
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.float64)
    V_school = np.zeros((mp, rank_school), dtype=np.float64)
    W_school = np.zeros((pn, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-9):
        """Verify T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for alpha in range(r):
                        s += U[a, alpha] * V[b, alpha] * W[c, alpha]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-9):
        """Fast verification using numpy broadcasting."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # U: (n,m,r), V: (m,p,r), W: (p,n,r)
        # Need: U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c
        # U_exp: (n, m, 1, r), V_exp: (1, m, p, r), W_exp: (n, 1, p, r) where W_exp[i,0,k,c]=W[k,i,c]
        Ue = U_r[:, :, np.newaxis, :]  # (n, m, 1, r)
        Ve = V_r[np.newaxis, :, :, :]  # (1, m, p, r)
        We = W_r.T[np.newaxis, :, :, :]  # (1, n, p, r) -> W_exp[i,0,k,c] = W[k,i,c]
        # We needs shape (n, 1, p, r): We[i, 0, k, c] = W_r[k, i, c] = W_r.T[i, k, c]
        We = W_r.T[:, np.newaxis, :, :]  # (n, 1, p, r)
        T = np.sum(Ue * Ve * We, axis=3)  # (n, m, p)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def to_half_integer(x):
        """Round to nearest half-integer."""
        return np.round(x * 2.0) / 2.0

    def try_rank(r, max_iters=200, seed=42):
        """Try to find a rank-r decomposition using gradient-based search with rounding."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        best_loss = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for attempt in range(min(max_iters, 30)):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random values
            U_init = rng.randn(nm, r) * 0.5
            V_init = rng.randn(mp, r) * 0.5
            W_init = rng.randn(pn, r) * 0.5
            
            # Gradient descent on the loss: sum_{i,j,k} (T[i,j,k] - 1)^2
            lr = 0.01
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            for it in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute T and gradients
                r_cur = r
                U_r = U_cur.reshape(n, m, r_cur)
                V_r = V_cur.reshape(m, p, r_cur)
                W_r = W_cur.reshape(p, n, r_cur)
                
                # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                Ue = U_r[:, :, np.newaxis, :]  # (n, m, 1, r)
                Ve = V_r[np.newaxis, :, :, :]  # (1, m, p, r)
                We = W_r.T[:, np.newaxis, :, :]  # (n, 1, p, r)
                T = np.sum(Ue * Ve * We, axis=3)  # (n, m, p)
                
                loss = np.sum((T - 1.0) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
                
                if loss < 1e-6:
                    break
                
                # Compute gradients
                # dT/dU[i,j,c] = V[j,k,c]*W[k,i,c] summed over k
                # dT/dV[j,k,c] = U[i,j,c]*W[k,i,c] summed over i
                # dT/dW[k,i,c] = U[i,j,c]*V[j,k,c] summed over j
                
                # For each c:
                dU = np.zeros_like(U_cur)
                dV = np.zeros_like(V_cur)
                dW = np.zeros_like(W_cur)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c_idx = k * n + i
                            err = T[i, j, k] - 1.0
                            for alpha in range(r_cur):
                                dU[a, alpha] += 2.0 * err * V_cur[b, alpha] * W_cur[c_idx, alpha]
                                dV[b, alpha] += 2.0 * err * U_cur[a, alpha] * W_cur[c_idx, alpha]
                                dW[c_idx, alpha] += 2.0 * err * U_cur[a, alpha] * V_cur[b, alpha]
                
                U_cur -= lr * dU
                V_cur -= lr * dV
                W_cur -= lr * dW
                
                # Decay learning rate
                lr *= 0.995
            
            # Try rounding to half-integers
            U_rounded = to_half_integer(best_U)
            V_rounded = to_half_integer(best_V)
            W_rounded = to_half_integer(best_W)
            
            if verify_fast(U_rounded, V_rounded, W_rounded, tol=1e-6):
                return U_rounded, V_rounded, W_rounded
        
        return None

    # Try to find a decomposition better than schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    # Try ranks from 47 down to 40
    for target_rank in range(rank_school - 1, max(16, rank_school - 10), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank, max_iters=20, seed=target_rank * 100 + 42)
        if result is not None:
            U_found, V_found, W_found = result
            if verify_fast(U_found, V_found, W_found, tol=1e-6):
                best_U = U_found
                best_V = V_found
                best_W = W_found
                best_rank = target_rank
                break  # Found a valid decomposition, stop searching
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
