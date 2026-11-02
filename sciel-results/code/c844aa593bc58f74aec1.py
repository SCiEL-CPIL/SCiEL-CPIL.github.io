import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _check_decomposition(U, V, W, n, m, p):
    """
    Checks if U, V, W form a valid decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Tensor T[i, j, k] should be 1 if indices match, 0 otherwise?
    # No, the condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid triplet?
    # Actually, the tensor M has M[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'.
    # So we need to check for all i, i', j, j', k, k':
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[k'*n+i', c] = delta(i,i') delta(j,j') delta(k,k')
    
    # We can construct the full tensor and compare.
    # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # This T should be 1 if the indices are consistent?
    # Wait, the indices in U, V, W are linked.
    # U row i*m+j corresponds to A_ij.
    # V row j*p+k corresponds to B_jk.
    # W row k*n+i corresponds to C_ik.
    # The product term is A_ij B_jk. This contributes to C_ik.
    # So for a specific (i,j,k), the term U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] 
    # should sum to 1.
    # For any other combination of indices, it should sum to 0?
    # No, the tensor is defined on the product space.
    # The condition is that the reconstructed tensor matches the matrix multiplication tensor.
    
    # Let's build the tensor T of shape (n*m, m*p, p*n)
    # T[idx_u, idx_v, idx_w] = sum_c U[idx_u, c] V[idx_v, c] W[idx_w, c]
    # We need T[idx_u, idx_v, idx_w] = 1 if idx_u, idx_v, idx_w correspond to (i,j), (j,k), (i,k)
    # i.e., if (idx_u // m == idx_w % n) and (idx_u % m == idx_v // p) and (idx_v % p == idx_w // n)
    # Let's verify indices:
    # idx_u = i*m + j
    # idx_v = j*p + k
    # idx_w = k*n + i
    
    # Condition:
    # i = idx_u // m, j = idx_u % m
    # j' = idx_v // p, k = idx_v % p
    # k' = idx_w // n, i' = idx_w % n
    # We need i == i', j == j', k == k'.
    
    T = np.einsum('uc,vc,wc->uvw', U, V, W)
    
    # Construct target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                target[idx_u, idx_v, idx_w] = 1
                
    return np.allclose(T, target, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=int)
    V_sb = np.zeros((m*p, rank_sb), dtype=int)
    W_sb = np.zeros((p*n, rank_sb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank
    # We'll try ranks from 19 to best_rank - 1
    for r in range(19, best_rank):
        if time.time() > deadline:
            break
            
        # Random search
        for _ in range(1000): # Limit iterations per rank
            if time.time() > deadline:
                break
                
            # Random integers -2 to 2
            U = np.random.randint(-2, 3, (n*m, r))
            V = np.random.randint(-2, 3, (m*p, r))
            W = np.random.randint(-2, 3, (p*n, r))
            
            if _check_decomposition(U, V, W, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break # Found one for this rank, try lower
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
