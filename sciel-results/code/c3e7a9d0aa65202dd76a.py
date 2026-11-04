import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s margin before 150s kill

    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a, b, c correspond to a valid multiplication term
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Fallback: Schoolbook rank 56
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Verify a decomposition
    def verify(U, V, W):
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.allclose(result, T, atol=1e-8)

    # Start with schoolbook
    best_U, best_V, best_W, best_rank = schoolbook()

    # Try to find a lower rank decomposition using ALS
    # We know rank 50 is possible. Try 48, 49, etc.
    for target_rank in range(48, 51):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + restart + target_rank)
            U_try = np.random.randn(n*m, target_rank) * 0.1
            V_try = np.random.randn(m*p, target_rank) * 0.1
            W_try = np.random.randn(p*n, target_rank) * 0.1
            
            # ALS iterations
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                for c in range(target_rank):
                    residual = T.copy()
                    for c2 in range(target_rank):
                        if c2 != c:
                            residual -= np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                    
                    vw = V_try[:,c] * W_try[:,c]
                    denom = np.sum(vw * vw)
                    if abs(denom) > 1e-12:
                        U_try[:,c] = np.einsum('abk,bk->a', residual, vw) / denom
                    else:
                        U_try[:,c] = 0
                
                # Update V
                for c in range(target_rank):
                    residual = T.copy()
                    for c2 in range(target_rank):
                        if c2 != c:
                            residual -= np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                    
                    uw = U_try[:,c] * W_try[:,c]
                    denom = np.sum(uw * uw)
                    if abs(denom) > 1e-12:
                        V_try[:,c] = np.einsum('abk,ak->b', residual, uw) / denom
                    else:
                        V_try[:,c] = 0
                
                # Update W
                for c in range(target_rank):
                    residual = T.copy()
                    for c2 in range(target_rank):
                        if c2 != c:
                            residual -= np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                    
                    uv = U_try[:,c] * V_try[:,c]
                    denom = np.sum(uv * uv)
                    if abs(denom) > 1e-12:
                        W_try[:,c] = np.einsum('abk,ab->k', residual, uv) / denom
                    else:
                        W_try[:,c] = 0
                
                # Check convergence
                result = np.zeros((n*m, m*p, p*n))
                for c in range(target_rank):
                    result += np.einsum('i,j,k->ijk', U_try[:,c], V_try[:,c], W_try[:,c])
                if np.linalg.norm(result - T) < 1e-8:
                    break
            
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                found = True
                break
        
        if found:
            break

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
