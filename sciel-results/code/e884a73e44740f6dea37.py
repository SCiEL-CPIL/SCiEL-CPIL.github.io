import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 23
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Laderman's 23 terms definitions
    # Each term is a tuple (u_indices, v_indices)
    # u_indices: list of (i, j) with coefficient 1
    # v_indices: list of (j, k) with coefficient 1
    
    terms = [
        # 1
        ([(0,0), (1,1), (2,2)], [(0,0), (1,1), (2,2)]),
        # 2
        ([(0,0), (1,1)], [(0,0), (1,1)]),
        # 3
        ([(1,1), (2,2)], [(1,1), (2,2)]),
        # 4
        ([(2,2), (0,0)], [(2,2), (0,0)]),
        # 5
        ([(0,0), (2,2)], [(0,0), (2,2)]),
        # 6
        ([(1,1), (0,0)], [(1,1), (0,0)]),
        # 7
        ([(0,1), (1,2)], [(1,0), (2,1)]),
        # 8
        ([(1,2), (2,0)], [(2,1), (0,2)]),
        # 9
        ([(2,0), (0,1)], [(0,2), (1,0)]),
        # 10
        ([(0,1), (1,2), (2,0)], [(1,0), (2,1), (0,2)]),
        # 11
        ([(0,2), (1,0)], [(2,0), (0,1)]),
        # 12
        ([(1,0), (2,1)], [(0,1), (1,2)]),
        # 13
        ([(2,1), (0,2)], [(1,2), (2,0)]),
        # 14
        ([(0,2), (1,0), (2,1)], [(2,0), (0,1), (1,2)]),
        # 15
        ([(0,0), (0,1)], [(1,0), (1,1)]),
        # 16
        ([(1,1), (1,2)], [(2,1), (2,2)]),
        # 17
        ([(2,2), (2,0)], [(0,2), (0,0)]),
        # 18
        ([(0,0), (0,1), (0,2)], [(1,0), (1,1), (1,2)]),
        # 19
        ([(1,0), (1,1), (1,2)], [(2,0), (2,1), (2,2)]),
        # 20
        ([(2,0), (2,1), (2,2)], [(0,0), (0,1), (0,2)]),
        # 21
        ([(0,0), (1,0)], [(0,1), (1,1)]),
        # 22
        ([(1,1), (2,1)], [(1,2), (2,2)]),
        # 23
        ([(2,2), (0,2)], [(2,0), (0,0)]),
    ]
    
    # Fill U and V
    for r in range(rank):
        u_indices, v_indices = terms[r]
        for (i, j) in u_indices:
            U[i * m + j, r] = 1
        for (j, k) in v_indices:
            V[j * p + k, r] = 1
            
    # Solve for W
    # For each i, k, we solve M @ w_ik = b_ik
    # M is 81x23, b_ik is 81x1
    for i in range(n):
        for k in range(p):
            # Construct b_ik
            b = np.zeros(n * m * m * p) # 3*3*3*3 = 81
            # Indexing for b: (a, b_idx, c, d) -> a*m*m*p + b_idx*m*p + c*p + d
            # Condition: a=i, d=k, b_idx=c
            for j in range(m):
                # a=i, b_idx=j, c=j, d=k
                idx = i * (m * m * p) + j * (m * p) + j * p + k
                b[idx] = 1
                
            # Construct M
            # M[row, r] = U[a, b_idx, r] * V[c, d, r]
            # row index: (a, b_idx, c, d)
            M = np.zeros((n * m * m * p, rank))
            for a in range(n):
                for b_idx in range(m):
                    for c in range(m):
                        for d in range(p):
                            row_idx = a * (m * m * p) + b_idx * (m * p) + c * p + d
                            u_val = U[a * m + b_idx, :]
                            v_val = V[c * p + d, :]
                            M[row_idx, :] = u_val * v_val
                            
            # Solve M @ w = b
            # Use least squares
            w_ik, residuals, rank_m, s = np.linalg.lstsq(M, b, rcond=None)
            
            # Store in W
            # W index: (k, i) -> k * n + i
            W[k * n + i, :] = w_ik
            
    # Verification
    # Check if sum_r U[i,j,r] V[j,k,r] W[i,k,r] == 1 for all i,j,k
    # And 0 otherwise?
    # Actually, we just need to check the final product.
    # Let's do a quick sanity check with random matrices
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_direct = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            # C[i, k] = sum_r W[k, i, r] * (sum_j U[i, j, r] A[i, j]) * (sum_j V[j, k, r] B[j, k])
            # Note: W is stored as W[k*n+i, r]
            w_vec = W[k * n + i, :]
            u_vec = U[i * m : (i+1) * m, :] # Shape (m, rank)
            v_vec = V[:m * (k+1), :] # This is wrong. V is (m*p, rank).
            # V[j, k, r] is at index j*p + k in V.
            # For fixed k, we need V[j, k, r] for all j.
            # V indices are j*p + k.
            # So v_vec[j, r] = V[j*p + k, r]
            
            # Let's compute the scalar M_r for this i, k? No, M_r depends on all A and B.
            # M_r = (sum_{a,b} U[a,b,r] A[a,b]) * (sum_{c,d} V[c,d,r] B[c,d])
            # This is expensive to do inside the loop.
            # Better:
            # Compute all M_r first.
            pass
            
    # Efficient verification
    M_r = np.zeros(rank)
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        # u_col is (n*m, 1), A is (n, m). Flatten A to (n*m, 1)
        a_flat = A.flatten()
        b_flat = B.flatten()
        M_r[r] = np.dot(u_col, a_flat) * np.dot(v_col, b_flat)
        
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            w_vec = W[k * n + i, :]
            C_decomp[i, k] = np.dot(w_vec, M_r)
            
    if not np.allclose(C_direct, C_decomp):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
