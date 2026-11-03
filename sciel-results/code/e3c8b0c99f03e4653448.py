"""Matrix multiplication 2x3x6 tensor decomposition with structured rank reduction."""
DIMENSIONS = [2, 3, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Trivial (schoolbook) decomposition: rank = n*m*p = 36
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial), dtype=float)
    V_trivial = np.zeros((m * p, rank_trivial), dtype=float)
    W_trivial = np.zeros((p * n, rank_trivial), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    
    def verify(U, V, W, tol=1e-8):
        """Verify that U, V, W form a valid decomposition."""
        r = U.shape[1]
        # Check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                    # Determine target
                    i_a, j_a = divmod(a, m)
                    j_b, k_b = divmod(b, p)
                    k_c, i_c = divmod(c, n)
                    target = 1.0 if (i_a == i_c and j_a == j_b and j_b == k_c) else 0.0
                    if abs(val - target) > tol:
                        return False
        return True
    
    # Strategy: For each k, the sub-computation is a 2x3 * 3x1 = 2x1 multiplication.
    # The sub-tensor for fixed k has rank at most 3 (since it's two dot products of length 3).
    # Actually, the minimal rank of the 6x3x2 tensor where T[i*3+j, j, i] = 1 is 3.
    # Let's verify this and construct an explicit decomposition.
    
    # For the sub-tensor (fixed k), we want U_sub (6x3), V_sub (3x3), W_sub (2x3)
    # such that sum_l U_sub[i*3+j, l] * V_sub[j, l] * W_sub[i, l] = 1 if (i,j,i) else 0
    
    # Try: 
    #   W_sub[0, :] = [1, 0, 0]
    #   W_sub[1, :] = [0, 1, 0]
    # Then for i=0: sum_l U_sub[3+j, l] * V_sub[j, l] * W_sub[0, l] = U_sub[3+j, 0] * V_sub[j, 0] = 1
    # For i=1: U_sub[3+j, 1] * V_sub[j, 1] = 1
    # And for the zero entries, we need U_sub[3+j', l] * V_sub[j, l] * W_sub[i, l] = 0 for (i,j) != (i',j')
    
    # Let's try:
    #   V_sub[j, 0] = 1 for all j, V_sub[j, 1] = 0 for all j, V_sub[j, 2] = 0 for all j
    #   Then U_sub[3+j, 0] = 1 for all j, U_sub[3+j, 1] = 0, U_sub[3+j, 2] = 0
    #   But then for i=1, we get U_sub[3+j, 1] * V_sub[j, 1] = 0, not 1.
    
    # Let's try a different approach. Use:
    #   W_sub[0, :] = [1, 1, 1]
    #   W_sub[1, :] = [1, 1, 1]
    # Then sum_l U_sub[i*3+j, l] * V_sub[j, l] * W_sub[i, l] = sum_l U_sub[i*3+j, l] * V_sub[j, l]
    # We need this to be 1 if (i,j,i) else 0.
    # But this doesn't distinguish between i=0 and i=1.
    
    # Let's try:
    #   W_sub[0, :] = [1, 0, 0]
    #   W_sub[1, :] = [0, 1, 0]
    #   V_sub = [[1,0,0], [0,1,0], [0,0,1]] (identity)
    #   Then for i=0: U_sub[3+j, 0] * V_sub[j, 0] = U_sub[3+j, 0] * (1 if j==0 else 0)
    #   So U_sub[3+0, 0] = 1, U_sub[3+1, 0] = 0, U_sub[3+2, 0] = 0
    #   But we need U_sub[3+j, 0] * V_sub[j, 0] = 1 for all j, which requires V_sub[j,0] != 0 for all j.
    
    # I think the minimal rank of the sub-tensor is actually 3, but I need to find the right decomposition.
    # Let me just try to solve it numerically for rank 3.
    
    def find_subtensor_decomp(r=3, max_attempts=500):
        """Find a rank-r decomposition of the 6x3x2 sub-tensor."""
        T = np.zeros((6, 3, 2))
        for i in range(2):
            for j in range(3):
                T[i*3+j, j, i] = 1
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            U = np.random.randn(6, r)
            V = np.random.randn(3, r)
            W = np.random.randn(2, r)
            
            # Simple local search
            best_err = np.inf
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                # Compute error
                err = 0.0
                for a in range(6):
                    for b in range(3):
                        for c in range(2):
                            val = np.sum(U[a,:] * V[b,:] * W[c,:])
                            err += (val - T[a,b,c]) ** 2
                
                if err < 1e-10:
                    return U, V, W
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Perturb one factor
                factor = random.choice(['U', 'V', 'W'])
                step = 0.1
                if factor == 'U':
                    idx = random.randint(0, 5)
                    U[idx,:] += step * np.random.randn(r)
                elif factor == 'V':
                    idx = random.randint(0, 2)
                    V[idx,:] += step * np.random.randn(r)
                else:
                    idx = random.randint(0, 1)
                    W[idx,:] += step * np.random.randn(r)
        
        if best_err < 1e-8:
            return best_U, best_V, best_W
        return None, None, None
    
    sub_U, sub_V, sub_W = find_subtensor_decomp(r=3)
    
    if sub_U is not None:
        # Build full decomposition
        total_rank = 3 * p  # 3 * 6 = 18
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        for k in range(p):
            offset = k * 3
            for i in range(n):
                for j in range(m):
                    for l in range(3):
                        U_full[i*m+j, offset+l] = sub_U[i*3+j, l]
            for j in range(m):
                for l in range(3):
                    V_full[j*p+k, offset+l] = sub_V[j, l]
            for i in range(n):
                for l in range(3):
                    W_full[k*n+i, offset+l] = sub_W[i, l]
        
        if verify(U_full, V_full, W_full):
            best_rank = total_rank
            best_U, best_V, best_W = U_full, V_full, W_full
    
    # If we still have the trivial decomposition, return it
    # Otherwise return the improved one
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
