import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    target_tensor = np.zeros((n*m, m*p, p*n))
    # Construct the matrix multiplication tensor
    # T[(i,j), (j,k), (k,i)] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[i*m + j, j*p + k, k*n + i] = 1.0

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Store schoolbook as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school

    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # Try to find lower rank
    for rank in range(best_rank - 1, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts for this rank
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iterations
            for _ in range(50):
                # Update U
                for c in range(rank):
                    # Contract T with V[:, c] and W[:, c]
                    # T is (A, B, C)
                    # V_c is (B,)
                    # W_c is (C,)
                    # Result is (A,)
                    U[:, c] = np.einsum('abc,b,c->a', target_tensor, V[:, c], W[:, c])
                    # Normalize
                    norm_v = np.dot(V[:, c], V[:, c])
                    norm_w = np.dot(W[:, c], W[:, c])
                    if norm_v * norm_w > 1e-10:
                        U[:, c] /= (norm_v * norm_w)
                    else:
                        U[:, c] = 0
                
                # Update V
                for c in range(rank):
                    V[:, c] = np.einsum('abc,a,c->b', target_tensor, U[:, c], W[:, c])
                    norm_u = np.dot(U[:, c], U[:, c])
                    norm_w = np.dot(W[:, c], W[:, c])
                    if norm_u * norm_w > 1e-10:
                        V[:, c] /= (norm_u * norm_w)
                    else:
                        V[:, c] = 0
                        
                # Update W
                for c in range(rank):
                    W[:, c] = np.einsum('abc,a,b->c', target_tensor, U[:, c], V[:, c])
                    norm_u = np.dot(U[:, c], U[:, c])
                    norm_v = np.dot(V[:, c], V[:, c])
                    if norm_u * norm_v > 1e-10:
                        W[:, c] /= (norm_u * norm_v)
                    else:
                        W[:, c] = 0
            
            # Check residual
            # Reconstruct T_approx
            # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(target_tensor - T_approx)
            
            if residual < 1e-6:
                # Try to round to integers or half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_approx_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                residual_rounded = np.linalg.norm(target_tensor - T_approx_rounded)
                
                if residual_rounded < 1e-9:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break # Try next rank
        
    # Return the best found
    # Convert to lists of lists as required by the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
