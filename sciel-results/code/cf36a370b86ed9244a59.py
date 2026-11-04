import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Start with the known rank-55 decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2x2
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    Wc = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Efficient check: for each column r, check if removing it still gives valid decomposition
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Without column r: T'[a,b,c] = T[a,b,c] - U[a,r]*V[b,r]*W[c,r]
    # We need T'[a,b,c] = 0 for all a,b,c where T[a,b,c] = 0, and T'[a,b,c] = 1 where T[a,b,c] = 1
    
    # More efficiently: check if U[:,r]*V[:,r]*W[:,r] is in the span of the other rank-1 tensors
    # This is equivalent to checking if the outer product is in the column space
    
    # For each r, compute the contribution and check if it's redundant
    def is_redundant(r, U_m, V_m, W_m):
        """Check if column r is redundant (its rank-1 tensor is in the span of others)."""
        # Compute the rank-1 tensor for column r
        contrib = np.einsum('a,b,c->abc', U_m[:, r], V_m[:, r], W_m[:, r])
        
        # The remaining tensor should equal T - contrib
        # Check if T - contrib can be decomposed with the remaining columns
        # This is equivalent to checking if contrib is in the span of the other rank-1 tensors
        
        # Efficient check: for each (a,b,c) where T[a,b,c]=1, 
        # we need sum_{r'!=r} U[a,r']*V[b,r']*W[c,r'] = 1 - contrib[a,b,c]
        # And for each (a,b,c) where T[a,b,c]=0,
        # we need sum_{r'!=r} U[a,r']*V[b,r']*W[c,r'] = -contrib[a,b,c]
        
        # Use einsum to compute the sum over all columns except r
        mask = np.ones(rank, dtype=bool)
        mask[r] = False
        U_rem = U_m[:, mask]
        V_rem = V_m[:, mask]
        W_rem = W_m[:, mask]
        
        T_rem = np.einsum('ar,br,cr->abc', U_rem, V_rem, W_rem)
        
        # Check if T_rem == T - contrib
        return np.allclose(T_rem, T - contrib, atol=1e-9)
    
    # Try removing each column
    best_rank = rank
    for r in range(rank):
        if time.time() > deadline:
            break
        if is_redundant(r, U_np, V_np, W_np):
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            rank -= 1
            # After removing, restart the check from the beginning
            r = -1  # restart
            if time.time() > deadline:
                break
    
    # Convert back to lists with integer/half-integer entries
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
    
    return {"U": U_out, "V": V_out, "W": W_out}
