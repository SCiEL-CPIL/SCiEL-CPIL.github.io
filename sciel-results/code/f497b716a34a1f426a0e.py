"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 115  # Leave 35s safety margin

    best_rank = rank_school
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Create target tensor
    # Target tensor T has 1s at (i*m+j, j*p+k, k*n+i)
    # Shape: (n*m, m*p, p*n)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    def compute_error(U, V, W):
        T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.sum((T_rec - T_target) ** 2)

    def gradient(U, V, W):
        T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        diff = T_rec - T_target
        # grad_U[u, r] = 2 * sum_{v, w} diff[u, v, w] * V[v, r] * W[w, r]
        gU = 2.0 * np.einsum('uvw,vr,wr->ur', diff, V, W)
        gV = 2.0 * np.einsum('uvw,ur,wr->vr', diff, U, W)
        gW = 2.0 * np.einsum('uvw,ur,vr->wr', diff, U, V)
        return gU, gV, gW

    def optimize_rank(R, max_iter=2000, lr=0.01, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Simple Adam optimizer
        mU, vU = np.zeros_like(U), np.zeros_like(U)
        mV, vV = np.zeros_like(V), np.zeros_like(V)
        mW, vW = np.zeros_like(W), np.zeros_like(W)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            t += 1
            err = compute_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-12:
                break
                
            gU, gV, gW = gradient(U, V, W)
            
            # Update U
            mU = b1 * mU + (1 - b1) * gU
            vU = b2 * vU + (1 - b2) * (gU ** 2)
            mU_hat = mU / (1 - b1 ** t)
            vU_hat = vU / (1 - b2 ** t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            # Update V
            mV = b1 * mV + (1 - b1) * gV
            vV = b2 * vV + (1 - b2) * (gV ** 2)
            mV_hat = mV / (1 - b1 ** t)
            vV_hat = vV / (1 - b2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            # Update W
            mW = b1 * mW + (1 - b1) * gW
            vW = b2 * vW + (1 - b2) * (gW ** 2)
            mW_hat = mW / (1 - b1 ** t)
            vW_hat = vW / (1 - b2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
        return best_U, best_V, best_W, best_err

    # Try ranks from lower bound up to schoolbook
    # Lower bound is 25. Realistic lower bound for 3x5x5 is likely > 30.
    # We will try a range of ranks.
    # To maximize chance, we try ranks that are "likely" to work first?
    # Or just sweep?
    # Let's sweep from 35 to 75.
    
    ranks_to_try = list(range(35, rank_school + 1))
    # Also add some specific lower ranks just in case
    for r in [30, 32, 34]:
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    ranks_to_try = sorted(ranks_to_try)
    
    # We have limited time. We should try fewer ranks with more iterations or more ranks with fewer?
    # Gradient descent is robust. Let's try 5 restarts per rank.
    # Time per rank: ~1-2 seconds per 1000 iterations?
    # 115s / (5 restarts * 2s) = 11 ranks?
    # We need to be smarter.
    
    # Strategy:
    # 1. Try rank 75 (guaranteed to work, but we want lower).
    # 2. Try ranks in descending order from 74 down to 35.
    #    Stop if we find a valid rank? No, we want the *minimal* rank.
    #    So we must search low ranks first?
    #    But low ranks are harder to find.
    #    If we start at 40 and fail, we might waste time.
    #    If we start at 70 and succeed, we know 70 works, but maybe 60 works too.
    
    # Heuristic: The rank is likely around 40-50.
    # Let's prioritize 40-50.
    
    priority_ranks = list(range(40, 51)) + list(range(35, 40)) + list(range(51, 76))
    # Remove duplicates and sort? No, keep priority order.
    # Actually, let's just iterate from 35 to 75.
    # But give more time/budget to the middle range?
    
    # Let's allocate time dynamically.
    # Total time 115s.
    # Number of ranks ~ 40.
    # ~2.8s per rank.
    # 1 restart per rank? Or 2?
    
    max_restarts_per_rank = 2
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(max_restarts_per_rank):
            if time.time() > deadline:
                break
                
            # Adjust learning rate based on rank?
            # Smaller ranks might need smaller LR to avoid diverging?
            # Or larger LR to escape?
            lr = 0.01 if R > 50 else 0.005
            
            U_curr, V_curr, W_curr, err = optimize_rank(R, max_iter=1000, lr=lr, seed=1000*restart + R)
            
            if err < 1e-10:
                # Found a valid decomposition for this rank
                # Polish it?
                # Just accept it
                best_rank = R
                best_U, best_V, best_W = U_curr, V_curr, W_curr
                found = True
                break
        
        # If we found a rank, we should continue to lower ranks to see if we can do better.
        # If we didn't find, we continue to higher ranks?
        # The loop is sorted ascending.
        # If we find R=45, we record it.
        # Then we try R=46... which is worse.
        # We should stop if we start failing at lower ranks?
        # No, non-convex. R=45 might fail, R=46 might succeed.
        # But we want the MINIMAL rank.
        # So if we find R=45, we should try R=44, 43...
        # My loop is ascending.
        # So if I find R=45, I have already tried 35-44 and failed.
        # So 45 is the best so far.
        # I should continue to 46? No, 46 is worse.
        # I should stop searching for *lower* ranks?
        # I am searching ascending.
        # So if I find R=45, I know 35-44 failed.
        # I can stop?
        # Yes, if I am confident 35-44 failed.
        # But maybe 35 failed due to bad init, but works with more tries?
        # Given time constraints, if we find a valid rank R, we can assume it's the minimum
        # provided we tried lower ranks sufficiently.
        
        if found:
            # We found a rank. Since we are iterating ascending, this is the smallest found so far.
            # We can break if we are confident lower ranks don't work.
            # To be safe, we might do a quick check of lower ranks with more restarts?
            # Or just break to save time for polishing?
            # Let's break to ensure we return a valid result within time.
            # The score is the rank. Lower is better.
            # If we find 45, and we tried 35-44 with 2 restarts each, it's a reasonable bet.
            break

    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
