"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    # Build the target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    triple_set = set(triples)
    
    # Precompute target tensor
    T_target = np.zeros((U_rows, V_rows, W_rows))
    for (a, b, c) in triples:
        T_target[a, b, c] = 1.0
    
    # Schoolbook solution
    rank_school = n * m * p  # 144
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.abs(T_recon - T_target)
        return np.max(diff) < tol
    
    def check_fast(U, V, W, tol=1e-4):
        """Faster approximate check."""
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        rng = np.random.RandomState(42)
        for _ in range(500):
            a = rng.randint(0, U_rows)
            b = rng.randint(0, V_rows)
            c = rng.randint(0, W_rows)
            if (a, b, c) in triple_set:
                continue
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val) > tol:
                return False
        return True
    
    def als_decompose(target_rank, max_iters=100, seed=42):
        """ALS for tensor decomposition."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(U_rows, target_rank) * 0.01
        V = rng.randn(V_rows, target_rank) * 0.01
        W = rng.randn(W_rows, target_rank) * 0.01
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            B_tensor = np.einsum('br,cr->bcr', V, W)
            B = B_tensor.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)
            G = B @ B.T + 1e-8 * np.eye(target_rank)
            try:
                G_inv = np.linalg.inv(G)
            except:
                G_inv = np.linalg.pinv(G)
            BT = B @ T_target.reshape(U_rows, -1).T
            U = (G_inv @ BT).T
            
            # Update V
            C_tensor = np.einsum('ar,cr->acr', U, W)
            C = C_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
            T_V = np.transpose(T_target, (1, 0, 2))
            Gv = C @ C.T + 1e-8 * np.eye(target_rank)
            try:
                Gv_inv = np.linalg.inv(Gv)
            except:
                Gv_inv = np.linalg.pinv(Gv)
            CT = C @ T_V.reshape(V_rows, -1).T
            V = (Gv_inv @ CT).T
            
            # Update W
            D_tensor = np.einsum('ar,br->abr', U, V)
            D = D_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
            T_W = np.transpose(T_target, (2, 0, 1))
            Gw = D @ D.T + 1e-8 * np.eye(target_rank)
            try:
                Gw_inv = np.linalg.inv(Gw)
            except:
                Gw_inv = np.linalg.pinv(Gw)
            DT = D @ T_W.reshape(W_rows, -1).T
            W = (Gw_inv @ DT).T
            
            # Error check
            if it % 10 == 0:
                err = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += (val - 1.0) ** 2
                if err < 1e-8:
                    break
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
        
        return U, V, W
    
    # Try decreasing ranks
    for target_rank in range(100, 35, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            
            result = als_decompose(target_rank, max_iters=80, seed=seed)
            if result is None:
                continue
            
            U, V, W = result
            
            if check_fast(U, V, W, tol=1e-4):
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_exact(U_r, V_r, W_r, tol=1e-9):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
