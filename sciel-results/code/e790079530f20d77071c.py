import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    dim1 = n * m  # 12
    dim2 = m * p  # 28
    dim3 = p * n  # 21
    
    # Build target tensor
    T = np.zeros((dim1, dim2, dim3))
    for a in range(dim1):
        i, j = divmod(a, m)
        for b in range(dim2):
            jp, k = divmod(b, p)
            if jp != j:
                continue
            for c in range(dim3):
                kp, ip = divmod(c, n)
                if kp == k and ip == i:
                    T[a, b, c] = 1.0
    
    # Naive fallback
    rank_naive = n * m * p
    U_naive = np.zeros((dim1, rank_naive))
    V_naive = np.zeros((dim2, rank_naive))
    W_naive = np.zeros((dim3, rank_naive))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_naive
    best_U, best_V, best_W = U_naive, V_naive, W_naive
    
    # Try to find lower rank
    for target_rank in range(75, 59, -1):
        if time.time() > deadline - 15:
            break
        
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(42 + target_rank * 100 + trial)
            U = rng.randn(dim1, target_rank)
            V = rng.randn(dim2, target_rank)
            W = rng.randn(dim3, target_rank)
            
            success = False
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U
                A = (V[:, :, None] * W[:, None, :]).reshape(target_rank, dim2 * dim3)
                Tf = T.reshape(dim1, dim2 * dim3)
                U_new, *_ = np.linalg.lstsq(A.T, Tf.T, rcond=None)
                U_new = U_new.T
                
                # Solve for V
                B = (U[:, :, None] * W[:, None, :]).reshape(target_rank, dim1 * dim3)
                Tf2 = T.transpose(1, 0, 2).reshape(dim2, dim1 * dim3)
                V_new, *_ = np.linalg.lstsq(B.T, Tf2.T, rcond=None)
                V_new = V_new.T
                
                # Solve for W
                C = (U[:, :, None] * V[:, None, :]).reshape(target_rank, dim1 * dim2)
                Tf3 = T.transpose(2, 0, 1).reshape(dim3, dim1 * dim2)
                W_new, *_ = np.linalg.lstsq(C.T, Tf3.T, rcond=None)
                W_new = W_new.T
                
                err = max(np.max(np.abs(U - U_new)), np.max(np.abs(V - V_new)), np.max(np.abs(W - W_new)))
                U, V, W = U_new, V_new, W_new
                if err < 1e-12:
                    break
            
            # Check
            T_recon = np.zeros((dim1, dim2, dim3))
            for r in range(target_rank):
                T_recon += U[:, r][:, None, None] * (V[:, r] @ W[:, r].T)[None, :, :]
            
            if np.max(np.abs(T_recon - T)) < 1e-8:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                success = True
                break
        
        if success:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
