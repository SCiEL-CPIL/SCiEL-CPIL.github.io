import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    total_dim = N * M * P
    num_terms = n * m * p
    
    # Construct the 120 basis tensors flattened
    # T_r corresponds to (i, j, k)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # Flatten index: (u * M + v) * P + w? No, just linear index in N*M*P space.
    # We form a matrix A of shape (total_dim, num_terms)
    
    A = np.zeros((total_dim, num_terms), dtype=np.float64)
    
    term_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Flatten index in the tensor space
                # The tensor is in R^N x R^M x R^P
                # We can flatten it to a vector of length N*M*P
                # Index = (u_idx * M + v_idx) * P + w_idx
                flat_idx = (u_idx * M + v_idx) * P + w_idx
                A[flat_idx, term_idx] = 1.0
                term_idx += 1
                
    # Check rank of A
    # Use SVD to find rank and null space
    U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)
    
    # Determine rank
    tol = 1e-9
    rank = np.sum(s > tol)
    
    if rank < num_terms:
        # Find null space vector
        # The last column of Vt_svd.T is the null space vector if rank < num_terms
        # Actually, the null space is spanned by the last (num_terms - rank) rows of Vt_svd
        # We just need one non-trivial vector
        c = Vt_svd[-1, :]
        
        # Find an index k where c[k] is significant
        # We want to eliminate term k
        # We need c[k] != 0
        # If all are 0, pick another vector from null space
        # But for generic tensors, null space is 1D if rank = num_terms - 1
        
        # Find k with max |c[k]| to avoid division by small numbers
        k = np.argmax(np.abs(c))
        
        if abs(c[k]) < tol:
            # Fallback to schoolbook
            rank_out = n * m * p
            U = [[0] * rank_out for _ in range(N)]
            V = [[0] * rank_out for _ in range(M)]
            W = [[0] * rank_out for _ in range(P)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for kk in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + kk][column] = 1
                        W[kk * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
            
        # Construct new decomposition of rank num_terms - 1
        new_rank = num_terms - 1
        U_new = [[0] * new_rank for _ in range(N)]
        V_new = [[0] * new_rank for _ in range(M)]
        W_new = [[0] * new_rank for _ in range(P)]
        
        # Map old term indices to new term indices
        # We remove term k
        # For each old term r != k, assign to new index
        new_idx = 0
        for r in range(num_terms):
            if r == k:
                continue
                
            # Coefficient for T_r in the new sum
            # T = sum_{r != k} (1 - c[r]/c[k]) T_r
            coeff = 1.0 - c[r] / c[k]
            
            # Extract i, j, kk for term r
            # r is the index in the loop i, j, kk
            # We need to recover i, j, kk from r
            # r = i * m * p + j * p + kk
            kk = r % p
            j = (r // p) % m
            i = r // (m * p)
            
            u_idx = i * m + j
            v_idx = j * p + kk
            w_idx = kk * n + i
            
            U_new[u_idx][new_idx] = coeff
            V_new[v_idx][new_idx] = 1.0
            W_new[w_idx][new_idx] = 1.0
            
            new_idx += 1
            
        # Verify the solution
        # Check if sum_r U_new[i,j,r] V_new[j,k,r] W_new[k,n+i,r] == 1 for all i,j,k
        # This is expensive, but we can do a quick check
        # Or just trust the linear algebra
        
        return {"U": U_new, "V": V_new, "W": W_new}
    else:
        # Rank is full, return schoolbook
        rank_out = n * m * p
        U = [[0] * rank_out for _ in range(N)]
        V = [[0] * rank_out for _ in range(M)]
        W = [[0] * rank_out for _ in range(P)]
        column = 0
        for i in range(n):
            for j in range(m):
                for kk in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + kk][column] = 1
                    W[kk * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
