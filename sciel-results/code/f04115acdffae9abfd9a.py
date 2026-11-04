import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    start_time = time.time()
    deadline = start_time + 140

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    # Try structured decompositions based on the tensor's linear index structure
    # T[a,b,c] = 1 if c = 2*(b-(a%3)*9) + a//3 and 0 <= b-(a%3)*9 < 9
    
    # Approach: use the fact that for each (i,j), the (b,c) pairs form a line
    # T[i*3+j, j*9+k, 2*k+i] = 1 for k=0..8
    
    # This is a sum over (i,j) of rank-9 tensors (one for each k)
    # Total rank: 2*3*9 = 54 (schoolbook)
    
    # To do better, we need to share rank across (i,j) pairs.
    
    # Try: for each k, define a rank-1 tensor that covers all (i,j)
    # T[i*3+j, j*9+k, 2*k+i] = 1 for all i,j
    # This requires: U[i*3+j, r_k] * V[j*9+k, r_k] * W[2*k+i, r_k] = 1 for all i,j
    # U[i*3+j] * V[j*9+k] * W[2*k+i] = 1
    # This is not separable in i and j because V depends on j and W depends on i.
    
    # Let me try a different factorization.
    # T = sum_{k} sum_{i,j} e_{i*3+j} ⊗ e_{j*9+k} ⊗ e_{2*k+i}
    
    # = sum_{k} sum_{i} sum_{j} e_{i*3+j} ⊗ e_{j*9+k} ⊗ e_{2*k+i}
    
    # For fixed k, this is a 6x9x2 tensor (i,j in the first, j in the second, i in the third)
    # T_k[i*3+j, j*9+k, 2*k+i] = 1
    
    # Let me try to find a low-rank decomposition of T_k.
    # T_k is 6x9x2 with 18 non-zero entries.
    
    # For fixed k, T_k[a, b, c] = 1 if a=i*3+j, b=j*9+k, c=2*k+i
    # b = j*9+k => j = (b-k)//9, and b-k must be divisible by 9 and in 0..18
    # a = i*3+j => i = (a-j)//3, and a-j must be divisible by 3 and in 0..3
    # c = 2*k+i => i = c-2*k
    
    # So for fixed k, T_k[a,b,c] = 1 if:
    # j = (b-k)//9 is integer in 0..2
    # i = c-2*k is in 0..1
    # a = i*3+j
    
    # This gives at most 3*2 = 6 non-zero entries per k? No, for each valid (i,j), there's one entry.
    # i in 0..1, j in 0..2, so 6 entries per k.
    # Total: 9 k's * 6 = 54 entries. Correct.
    
    # For fixed k, T_k is a 6x9x2 tensor with 6 non-zero entries.
    # Can we decompose it with rank < 6?
    
    # T_k[i*3+j, j*9+k, 2*k+i] = 1 for i in 0..1, j in 0..2
    
    # = sum_{i=0}^{1} sum_{j=0}^{2} e_{i*3+j} ⊗ e_{j*9+k} ⊗ e_{2*k+i}
    
    # This is a sum of 6 rank-1 tensors. Can we do better?
    
    # Let me try to see if there's a rank-3 decomposition.
    # We need: sum_{r=1}^{3} U_k[a,r] * V_k[b,r] * W_k[c,r] = T_k[a,b,c]
    
    # This is 6*9*2 = 108 equations in 6*3+9*3+2*3 = 51 unknowns.
    # Overdetermined, but the tensor is sparse so maybe it's solvable.
    
    # Let me just try random search for each k and see if rank 3 or 4 works.
    
    # Actually, let me try a completely different approach: use the fact that the problem is small enough
    # to do an exhaustive or semi-exhaustive search.
    
    # For now, let me just try random search for ranks 20-40 with more trials.
    
    for r in range(20, 41):
        if time.time() > deadline:
            break
        if r >= best_rank:
            break
            
        max_trials = 30 if r <= 30 else 10
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            np.random.seed(200000 + r * 1000 + trial)
            
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            M = np.zeros((mp * pn, r))
            for r_idx in range(r):
                M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx]).flatten()
            
            try:
                M_pinv = np.linalg.pinv(M)
                
                U = np.zeros((nm, r))
                for a in range(nm):
                    target = T[a, :, :].flatten()
                    U[a, :] = M_pinv @ target
                
                if verify_solution(U, V, W, T):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    break
            except:
                continue
        
        if best_rank == r:
            break
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
