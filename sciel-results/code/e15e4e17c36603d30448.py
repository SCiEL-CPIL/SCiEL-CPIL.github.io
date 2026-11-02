# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s kill

    n, m, p = 4, 4, 8

    # Schoolbook decomposition as fallback (rank 128)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return mat.tolist()

    best_U = to_list(U_school)
    best_V = to_list(V_school)
    best_W = to_list(W_school)
    best_rank = rank_school

    # Strategy: Exploit the product structure T[i,j,k] = 1
    # Since p = 8 = 4*2, we can write k = 2*k2 + k1 where k2 in {0,1}, k1 in {0,1,2,3}
    # Actually, let's think differently.
    #
    # The matmul tensor T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,8)
    # This is a rank-1 tensor in the (i,j,k) space! But wait, that's not right.
    # T[i,j,k] = 1 means the tensor is all ones, which IS rank 1 as a 3-tensor.
    #
    # But the constraint is that the decomposition must be of the form:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # The indices are coupled: U depends on (i,j), V on (j,k), W on (k,i).
    # This is NOT a free 3-tensor decomposition - it's a specific tensor network.
    #
    # Let's try a structured approach:
    # For the 4x4x4 case, we know the matmul tensor can be decomposed.
    # For 4x4x8, we can use the structure.
    
    # Approach: Try to find a rank-64 solution using the block structure
    # k ranges over 8 values. Let's try to pair up k values.
    
    # Actually, let's try a direct constructive approach:
    # If we can find U, V, W with rank r such that the tensor equation holds,
    # we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # For rank 64, let's try:
    # U[i,j,r] = a[i,j,r1] * b[j,r2] where r = r1*4 + r2, r1 in {0,1,2,3}, r2 in {0,1,2,3}
    # This is getting complicated. Let's try a simpler structured ansatz.
    
    # Simple structured attempt: 
    # Let rank = 64 = 4*4*4
    # U[i,j, (a,b,c)] = delta_{i,a} * delta_{j,b}  (only 16 nonzero per row pattern)
    # No, let's just try to verify if a rank-64 solution exists by construction.
    
    # Alternative: use the fact that for square matrices, there are known
    # sub-schoolbook decompositions. For 4x4, the known best is around 49.
    # For 4x4x8, we might get rank = 4 * 49 / 4 * 2 = ... not straightforward.
    
    # Let me try a different approach: random search with structure.
    # Initialize with a pattern that respects the symmetry.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Check if U,V,W exactly decompose the matmul tensor."""
        rank = len(W[0])
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T_rec, 1.0, atol=1e-10)
    
    # Try to construct a rank-64 solution using block structure
    # The key insight: T[i,j,k] = 1 can be written as sum of products
    # if we use the structure of the index space.
    
    # Let's try: for k in 0..7, split into two groups of 4
    # Group 1: k = 0,1,2,3 ; Group 2: k = 4,5,6,7
    # If we can find a 4x4x4 decomposition of rank r, then 4x4x8 might be rank 2r
    # by duplicating the structure for each group.
    
    # For 4x4x4 matmul tensor T[i,j,k]=1, the schoolbook is rank 64.
    # Can we do better? The tensor is all-ones, which as a free 3-tensor has rank 1,
    # but with the coupled index structure, it's different.
    
    # Let's just try to find ANY lower rank solution via optimization.
    # Use a smarter initialization.
    
    def try_constructive_rank(rank, max_iter=200, n_restarts=10):
        """Try to find a decomposition of given rank using ALS with good init."""
        if rank >= best_rank:
            return None
        if time.time() > deadline - 15:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                return None
            
            # Structured initialization
            if restart == 0:
                # Use a block-structured init
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # For rank = 64, use 4x4x4 block structure
                if rank == 64:
                    for i in range(n):
                        for j in range(m):
                            for a in range(4):
                                for b in range(4):
                                    r = a * 16 + b * 4 + i
                                    U[i * m + j, r] = 1.0 / 4.0
                                    
                                    for k in range(p):
                                        # V[j, k, r]
                                        V[j * p + k, r] = 1.0 / 4.0
                                        
                                        # W[k, i, r]  
                                        W[k * n + i, r] = 1.0 / 4.0
                else:
                    rng = np.random.RandomState(restart * 1000 + rank)
                    U = rng.randn(n * m, rank) * 0.1
                    V = rng.randn(m * p, rank) * 0.1
                    W = rng.randn(p * n, rank) * 0.1
            else:
                rng = np.random.RandomState(restart * 7919 + rank + 12345)
                U = rng.randn(n * m, rank)
                V = rng.randn(m * p, rank)
                W = rng.randn(p * n, rank)
            
            # ALS iterations
            prev_error = float('inf')
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        V_slice = V[j * p: j * p + p, :]  # (p, rank)
                        W_slice = W[i::n, :]  # (p, rank)
                        A = V_slice * W_slice  # (p, rank)
                        b = np.ones(p)
                        try:
                            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U[i * m + j, :] = u
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        U_slice = U[j::m, :]  # (n, rank)
                        W_slice = W[k * n: k * n + n, :]  # (n, rank)
                        B = U_slice * W_slice  # (n, rank)
                        c = np.ones(n)
                        try:
                            v, _, _, _ = np.linalg.lstsq(B, c, rcond=None)
                            V[j * p + k, :] = v
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        U_slice = U[i * m: i * m + m, :]  # (m, rank)
                        V_slice = V[k::p, :]  # (m, rank)
                        C = U_slice * V_slice  # (m, rank)
                        d = np.ones(m)
                        try:
                            w, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
                            W[k * n + i, :] = w
                        except:
                            pass
                
                # Check error
                U_reshaped = U.reshape(n, m, rank)
                V_reshaped = V.reshape(m, p, rank)
                W_reshaped = W.reshape(p, n, rank)
                
                try:
                    T_rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                    error = np.sum((T_rec - 1.0)**2)
                except:
                    error = float('inf')
                
                if error < 1e-8:
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    U_r = U_rounded.reshape(n, m, rank)
                    V_r = V_rounded.reshape(m, p, rank)
                    W_r = W_rounded.reshape(p, n, rank)
                    
                    try:
                        T_rec_r = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                        error_r = np.sum((T_rec_r - 1.0)**2)
                    except:
                        error_r = float('inf')
                    
                    if error_r < 1e-6:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    else:
                        # Return unrounded if it's exact enough
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                
                if abs(prev_error - error) < 1e-12:
                    break
                prev_error = error
        
        return None
    
    # Try rank 64 first (half of schoolbook)
    if time.time() < deadline - 20:
        res = try_constructive_rank(64, max_iter=300, n_restarts=5)
        if res:
            best_U = res["U"]
            best_V = res["V"]
            best_W = res["W"]
            best_rank = 64
    
    # Try rank 60
    if best_rank > 60 and time.time() < deadline - 20:
        res = try_constructive_rank(60, max_iter=300, n_restarts=5)
        if res:
            best_U = res["U"]
            best_V = res["V"]
            best_W = res["W"]
            best_rank = 60

    # Try rank 56
    if best_rank > 56 and time.time() < deadline - 20:
        res = try_constructive_rank(56, max_iter=300, n_restarts=5)
        if res:
            best_U = res["U"]
            best_V = res["V"]
            best_W = res["W"]
            best_rank = 56

    # Try a few more ranks
    for r in [52, 48, 44, 40]:
        if best_rank > r and time.time() < deadline - 15:
            res = try_constructive_rank(r, max_iter=200, n_restarts=3)
            if res:
                best_U = res["U"]
                best_V = res["V"]
                best_W = res["W"]
                best_rank = r

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
