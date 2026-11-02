"""Rewritten program for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    result = np.einsum('ijr,jkr,ki r->ijk', U3, V3, W3)
    return np.max(np.abs(result - 1.0)) < tol

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s limit
    
    # Schoolbook rank: n*m*p = 48
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank), dtype=float)
    best_V = np.zeros((m*p, best_rank), dtype=float)
    best_W = np.zeros((p*n, best_rank), dtype=float)
    
    # Build schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
    
    # Strategy: Use the known result that rank(2x3x8) can be reduced
    # by exploiting structure. The key insight is that 2x3x8 can be 
    # decomposed using a combination of smaller tensors.
    
    # Known lower bounds and upper bounds:
    # - Schoolbook: 48
    # - Strassen-like approaches can reduce this
    
    # Let's try a systematic approach:
    # The tensor for 2x3x8 multiplication has the property that
    # we can group terms to reduce the rank.
    
    # One approach: view 2x3x8 as related to 2x3x4 x 2 (Kronecker)
    # If we can find a rank-r decomposition for 2x3x4, we get rank 2r for 2x3x8
    
    # Known: rank(2x3x4) <= 11 (there are known decompositions)
    # So rank(2x3x8) <= 22
    
    # Let me try to construct a rank-11 decomposition for 2x3x4
    # and then use it to get rank-22 for 2x3x8.
    
    # Actually, let me think about this more carefully.
    # The 2x3x4 tensor: n=2, m=3, p=4, schoolbook rank = 24
    # We want to find a decomposition with rank < 24.
    
    # Let me try a different approach: direct search with structured random matrices.
    
    # For each target rank, try to find a decomposition.
    # Use the fact that entries can be from a small set.
    
    def try_random_decomposition(target_rank, n_iters=1000, seed=None):
        """Try to find a rank decomposition using random entries."""
        if seed is not None:
            np.random.seed(seed)
        
        for it in range(n_iters):
            if time.time() > deadline - 10:
                break
            
            # Use entries from {-1, 0, 1} for simplicity
            U = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
        
        return None
    
    # Try ranks from 47 down, but be smart about it
    # Focus on ranks that are likely to work based on known results
    
    # First, let's try some structured approaches
    
    # Approach 1: Block decomposition
    # Split the 8 columns into two groups of 4
    # This gives two independent 2x3x4 problems
    
    # If we can find a rank-r decomposition for 2x3x4,
    # then rank(2x3x8) <= 2r
    
    # Let's try to find a good decomposition for 2x3x4 first
    
    def build_2x3x4_decomposition(target_rank, n_iters=500):
        """Try to find a rank decomposition for 2x3x4."""
        nn, mm, pp = 2, 3, 4
        
        def verify_234(U_t, V_t, W_t, tol=1e-9):
            U_t = np.asarray(U_t, dtype=float)
            V_t = np.asarray(V_t, dtype=float)
            W_t = np.asarray(W_t, dtype=float)
            rank = U_t.shape[1]
            U3 = U_t.reshape(nn, mm, rank)
            V3 = V_t.reshape(mm, pp, rank)
            W3 = W_t.reshape(pp, nn, rank)
            result = np.einsum('ijr,jkr,ki r->ijk', U3, V3, W3)
            return np.max(np.abs(result - 1.0)) < tol
        
        for it in range(n_iters):
            if time.time() > deadline - 15:
                break
            
            U_t = np.random.choice([-1, 0, 1], size=(nn*mm, target_rank))
            V_t = np.random.choice([-1, 0, 1], size=(mm*pp, target_rank))
            W_t = np.random.choice([-1, 0, 1], size=(pp*nn, target_rank))
            
            if verify_234(U_t, V_t, W_t):
                return U_t, V_t, W_t
        
        return None
    
    # Try to find a 2x3x4 decomposition with rank < 24
    # Then use it to get a 2x3x8 decomposition
    
    for r234 in range(23, 10, -1):
        if time.time() > deadline - 20:
            break
        
        # If we find rank r234 for 2x3x4, we get rank 2*r234 for 2x3x8
        target_rank_238 = 2 * r234
        if target_rank_238 >= best_rank:
            continue
        
        result_234 = build_2x3x4_decomposition(r234, n_iters=200)
        if result_234 is not None:
            U_234, V_234, W_234 = result_234
            
            # Build 2x3x8 from two copies of 2x3x4
            U_238 = np.zeros((n*m, target_rank_238), dtype=float)
            V_238 = np.zeros((m*p, target_rank_238), dtype=float)
            W_238 = np.zeros((p*n, target_rank_238), dtype=float)
            
            for r_idx in range(r234):
                # First copy: columns 0-3
                U_238[:, r_idx] = U_234[:, r_idx]
                for j in range(m):
                    for k in range(4):
                        V_238[j*8 + k, r_idx] = V_234[j*4 + k, r_idx]
                for k in range(4):
                    for i in range(n):
                        W_238[k*2 + i, r_idx] = W_234[k*2 + i, r_idx]
                
                # Second copy: columns 4-7
                r2 = r_idx + r234
                U_238[:, r2] = U_234[:, r_idx]
                for j in range(m):
                    for k in range(4):
                        V_238[j*8 + (k+4), r2] = V_234[j*4 + k, r_idx]
                for k in range(4):
                    for i in range(n):
                        W_238[(k+4)*2 + i, r2] = W_234[k*2 + i, r_idx]
            
            if _verify_decomposition(U_238, V_238, W_238, n, m, p):
                best_rank = target_rank_238
                best_U, best_V, best_W = U_238, V_238, W_238
                break
    
    # Try direct random search for 2x3x8 at various ranks
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 15:
            break
        
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            result = try_random_decomposition(target_rank, n_iters=10, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
