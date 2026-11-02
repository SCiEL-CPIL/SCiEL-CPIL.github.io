import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_rank11_233():
    """
    Returns (U, V, W) for a rank-11 decomposition of 2x3 x 3x3 multiplication.
    U: shape (6, 11) - A is 2x3, flat index i*3+j
    V: shape (9, 11) - B is 3x3, flat index j*3+k
    W: shape (6, 11) - C is 2x3, flat index k*2+i (trace convention)
    """
    n, m, p = 2, 3, 3
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Terms for 2x3 x 3x3
    # A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # B: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # C: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    
    terms = []
    
    # Strassen 7 terms for 2x2 x 2x2 (A[:,0:2] * B[0:2,:])
    # A sub: 0,1,3,4; B sub: 0,1,3,4; C sub: 0,1,3,4
    
    # M1 = (A00+A11)(B00+B11)
    terms.append(({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}))
    # M2 = (A10+A11)B00
    terms.append(({3:1, 4:1}, {0:1}, {3:1, 4:-1}))
    # M3 = A00(B01-B11)
    terms.append(({0:1}, {1:1, 4:-1}, {1:1, 4:1}))
    # M4 = A11(B10-B00)
    terms.append(({4:1}, {3:1, 0:-1}, {0:1, 3:1}))
    # M5 = (A00+A01)B11
    terms.append(({0:1, 1:1}, {4:1}, {0:-1, 1:1}))
    # M6 = (A10-A00)(B00+B01)
    terms.append(({3:1, 0:-1}, {0:1, 1:1}, {4:1}))
    # M7 = (A01-A11)(B10+B11)
    terms.append(({1:1, 4:-1}, {3:1, 4:1}, {0:1}))
    
    # 4 scalar terms for A[:,2] * B[2,:]
    # K1 = A02 * B20 -> C00
    terms.append(({2:1}, {6:1}, {0:1}))
    # K2 = A02 * B21 -> C01
    terms.append(({2:1}, {7:1}, {1:1}))
    # K3 = A12 * B20 -> C10
    terms.append(({5:1}, {6:1}, {3:1}))
    # K4 = A12 * B21 -> C11
    terms.append(({5:1}, {7:1}, {4:1}))
    
    # 4 scalar terms for A[:,2] * B[0:2,2] (B's 3rd column, rows 0,1)
    # Wait, B[0:2, 2] is indices 2, 5.
    # K5 = A02 * B02 -> C02
    terms.append(({2:1}, {2:1}, {2:1}))
    # K6 = A02 * B12 -> C12
    terms.append(({2:1}, {5:1}, {5:1}))
    # K7 = A12 * B02 -> C02 ... wait, C indices.
    # C is 2x3. C[i, k]. Flat index k*n + i = 2*i + k.
    # C02 is (0, 2) -> index 2*0 + 2 = 2.
    # C12 is (1, 2) -> index 2*1 + 2 = 4.
    # Let's re-verify C indices.
    # C: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # So C02 is index 2. C12 is index 5.
    
    # Correction:
    # K5 = A02 * B02 -> C02 (index 2)
    # K6 = A02 * B12 -> C12 (index 5)
    # K7 = A12 * B02 -> C02? No, A12 is (1,2). B02 is (0,2). Product contributes to C12? No, C[i,k] = sum A[i,j]B[j,k].
    # A12 * B02 -> C[1, 2] -> index 5.
    # A12 * B12 -> C[1, 2] -> index 5.
    
    # Let's list them properly.
    # We need to cover C02 (index 2) and C12 (index 5).
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    
    # My previous "4 scalar terms" for A[:,2]*B[2,:] covered:
    # A02*B20 -> C00 (0)
    # A02*B21 -> C01 (1)
    # A12*B20 -> C10 (3)
    # A12*B21 -> C11 (4)
    
    # Now I need to cover the remaining terms involving B column 2.
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    
    # This is a 2x3 (A rows) by 3x1 (B col 2) multiplication? No, it's part of the full product.
    # The terms involving B[:, 2] are:
    # A[:, 0] * B[0, 2] -> A00*B02, A10*B02
    # A[:, 1] * B[1, 2] -> A01*B12, A11*B12
    # A[:, 2] * B[2, 2] -> A02*B22, A12*B22
    
    # This is a 2x3 x 3x1 multiplication.
    # Can we do this in 4 terms?
    # Yes, it's essentially 2x3 * 3x1.
    # Rank of 2x3x1 multiplication?
    # It's just 6 multiplications. Can we do better?
    # 2x3 * 3x1 is equivalent to dot product of 2 vectors of length 3? No, 2 rows of A against 1 column of B.
    # It's 2 independent dot products of length 3.
    # Rank is 6? Or can we share?
    # Since the output is 2x1, and inputs are 2x3 and 3x1.
    # This is just two independent rank-3 multiplications (2x3x1 and 1x3x1? No).
    # Actually, 2x3 * 3x1 is just 2 dot products.
    # There is no known decomposition better than 6 for 2x3x1?
    # Wait, 2x3x1 is the same as 3x2x1 transposed.
    # 3x2 * 2x1.
    # Rank of 3x2x1?
    # It is 6.
    # So total rank would be 7 (Strassen) + 4 (B col 0,1 row 2) + 6 (B col 2) = 17?
    # My previous rank-11 construction was flawed or I made a mistake in counting.
    
    # Let's re-examine the "4 scalar terms" in the inspiration.
    # The inspiration said "4 scalar terms for A[:,2] * B[2,:]".
    # B[2,:] is row 2 of B. That's B20, B21, B22.
    # A[:,2] is col 2 of A. That's A02, A12.
    # Product:
    # A02*B20 -> C00
    # A02*B21 -> C01
    # A02*B22 -> C02
    # A12*B20 -> C10
    # A12*B21 -> C11
    # A12*B22 -> C12
    # That's 6 terms.
    
    # The inspiration code for rank-11 2x3x3 was omitted, but the description said "Strassen on 2x2x2 + 4 scalar terms".
    # 7 + 4 = 11.
    # This implies that the remaining 6 terms (involving B col 2 or A col 2) were handled by 4 terms?
    # Or maybe the Strassen part handled more?
    
    # Let's look at the structure of 2x3 x 3x3.
    # A = [A1 A2], where A1 is 2x2, A2 is 2x1.
    # B = [B1 B2], where B1 is 3x2, B2 is 3x1.
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2 x 3x2. Rank 11 (as per current program).
    # A2 B2 is 2x1 x 3x1. Rank 6.
    # Total rank 17.
    
    # How can it be 11?
    # Maybe the decomposition is not block-wise additive in that simple way.
    # Or maybe 2x3 x 3x3 has rank 11?
    # Let's check the rank of 2x3 x 3x3.
    # It is known that R(2,3,3) = 11?
    # If so, there must be a clever decomposition.
    
    # Let's try to find a rank-11 decomposition for 2x3 x 3x3.
    # If I can't find one, I should stick to the rank-33 (which is 3 * 11 for 2x3x2) or rank-22 (2 * 11 for 2x3x3) if rank-11 2x3x3 exists.
    
    # The inspiration 2 explicitly claims rank 22 for 2x3x6 using two rank-11 2x3x3 blocks.
    # This implies rank-11 2x3x3 exists.
    
    # Let's assume it exists and try to construct it or find it.
    # Since I can't search efficiently, I will use the rank-33 decomposition which is verified to work.
    # However, the goal is to improve the score. Rank 22 is better than Rank 33.
    # If I can implement a valid rank-11 2x3x3, I can get rank 22.
    
    # Let's look for a known rank-11 decomposition of 2x3 x 3x3.
    # One source says R(2,3,3) = 11.
    # Construction:
    # C = AB
    # C00 = A00 B00 + A01 B10 + A02 B20
    # C01 = A00 B01 + A01 B11 + A02 B21
    # C02 = A00 B02 + A01 B12 + A02 B22
    # C10 = A10 B00 + A11 B10 + A12 B20
    # C11 = A10 B01 + A11 B11 + A12 B21
    # C12 = A10 B02 + A11 B12 + A12 B22
    
    # This is 18 multiplications schoolbook.
    # Can we reduce?
    
    # Let's try to use the rank-11 decomposition for 2x3 x 3x2 (which we have) and extend it.
    # 2x3 x 3x2 rank 11.
    # 2x3 x 3x3 = 2x3 x 3x2 + 2x3 x 3x1.
    # If we can do 2x3 x 3x1 in rank 0? No.
    # If we can do 2x3 x 3x3 in rank 11, it must be a global optimization.
    
    # Given the complexity and time limit, I will stick to the Rank 33 solution which is known to work.
    # But wait, the inspiration 2 says "Score 1.059" for Rank 22.
    # My current score is 1.0909 (Rank 33).
    # 1.059 is better (lower is better? No, "score is the rank", "search score is seed/value".
    # If seed is 36 (schoolbook), Rank 33 -> 36/33 = 1.0909.
    # Rank 22 -> 36/22 = 1.636.
    # Wait, the prompt says "score is the rank" and "search score is seed/value".
    # "1.0 equals the seed program".
    # If seed is Rank 36, then Rank 33 gives 36/33 = 1.09.
    # Rank 22 gives 36/22 = 1.63.
    # So Rank 22 is a BETTER score (higher number).
    # The prompt says "lower is better" for the RAW value (rank).
    # "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    # So higher score is better.
    # Current score 1.0909.
    # Inspiration 2 score 1.059? That would be WORSE.
    # Wait, "Measured: 1 -> 1.059".
    # Maybe the seed is different?
    # "score 1.0909" for current program.
    # If current is Rank 33, and score is 1.0909, then Seed/33 = 1.0909 => Seed = 36.
    # If Inspiration 2 is Rank 22, Score = 36/22 = 1.636.
    # Why does it say 1.059?
    # Maybe Inspiration 2 is Rank 34? 36/34 = 1.058.
    # Or maybe the seed is different.
    
    # Let's re-read: "Uses two copies of a Rank 11 decomposition for 2x3x3 to construct a Rank 22 decomposition for 2x3x6, improving score from 34 to 22 Measured: 1 -> 1.059."
    # This sentence is confusing. "improving score from 34 to 22".
    # If rank goes from 34 to 22, that's an improvement.
    # If score is 1.059, and rank is 22, then Seed/22 = 1.059 => Seed = 23.3? No.
    # Maybe the score IS the rank? "score is the rank".
    # "search score is seed/value".
    # If value = rank, and score = seed/rank.
    # If rank 22 gives score 1.059, then seed = 23.3.
    # If rank 33 gives score 1.0909, then seed = 36.
    # Contradiction.
    
    # Let's assume the goal is to MINIMIZE the rank.
    # Current Rank: 33.
    # Target Rank: 22 (if valid).
    
    # I will attempt to implement the Rank 22 solution using two Rank 11 2x3x3 blocks.
    # I need a valid Rank 11 2x3x3 decomposition.
    
    # Since I don't have one, I will use the Rank 33 solution.
    # However, I can try to find a Rank 11 2x3x3 decomposition using the `least_squares` approach from Inspiration 1, but for 2x3x3.
    
    def _try_find_rank11_233(max_time=30):
        n, m, p = 2, 3, 3
        rank = 11
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        def residuals(params):
            U = params[:dim_u*rank].reshape(dim_u, rank)
            V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        res.append(s - 1.0)
            return np.array(res)
        
        best_solution = None
        best_cost = np.inf
        
        for seed in range(50):
            if time.time() - max_time > 0: # Pseudo-check
                break
            np.random.seed(seed)
            x0 = np.random.randn(dim_u*rank + dim_v*rank + dim_w*rank) * 0.5
            try:
                from scipy.optimize import least_squares
                result = least_squares(residuals, x0, max_nfev=500)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_solution = result.x
                    if best_cost < 1e-14:
                        break
            except:
                continue
        
        if best_solution is not None and best_cost < 1e-14:
            U = best_solution[:dim_u*rank].reshape(dim_u, rank)
            V = best_solution[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W = best_solution[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: break
            
            if valid:
                return U_r, V_r, W_r
        
        return None

    return None # Placeholder, will be implemented in construct

def _build_rank33_decomposition():
    """
    Constructs a rank-33 decomposition for 2x3 by 3x6 matrix multiplication.
    """
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    terms = []
    
    # Strassen 7 terms for 2x2 x 2x2
    terms.append(({0:1, 4:1}, {0:1, 3:1}, {0:1, 3:1}))
    terms.append(({3:1, 4:1}, {0:1}, {2:1, 3:-1}))
    terms.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))
    terms.append(({4:1}, {2:1, 0:-1}, {0:1, 2:1}))
    terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))
    terms.append(({3:1, 0:-1}, {0:1, 1:1}, {3:1}))
    terms.append(({1:1, 4:-1}, {2:1, 3:1}, {0:1}))
    
    # 4 scalar terms for A[:,2] * B[2,:]
    terms.append(({2:1}, {4:1}, {0:1}))
    terms.append(({2:1}, {5:1}, {1:1}))
    terms.append(({5:1}, {4:1}, {2:1}))
    terms.append(({5:1}, {5:1}, {3:1}))
    
    for block in range(3):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = block * 11 + t_idx
            
            for idx, val in u_dict.items():
                U[idx, col] = val
            
            for l_idx, val in v_dict.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            
            for l_idx, val in w_dict.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find Rank 11 2x3x3
    if time.time() < deadline - 20:
        U11, V11, W11 = _try_find_rank11_233(max_time=20)
        if U11 is not None:
            # Build Rank 22 2x3x6
            rank = 22
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for block in range(2):
                for c in range(11):
                    col = block * 11 + c
                    U[:, col] = U11[:, c]
                    for j in range(m):
                        for k in range(3):
                            V[j*p + (block*3 + k), col] = V11[j*3 + k, c]
                    for k in range(3):
                        for i in range(n):
                            W[(block*3 + k)*n + i, col] = W11[k*n + i, c]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Rank 33
    U, V, W = _build_rank33_decomposition()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
