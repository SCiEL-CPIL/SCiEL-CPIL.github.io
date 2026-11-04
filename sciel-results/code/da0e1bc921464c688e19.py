import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
START_TIME = time.time()
DEADLINE = START_TIME + 115  # Safety margin

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _search_rank(rank, seed, time_budget):
    """
    Search for integer solution U', V', W' such that sum_r U'[i,j,r] V'[j,k,r] W'[k,i,r] = 8.
    Returns U, V, W (halves of U', V', W') if found.
    """
    np.random.seed(seed)
    n, m, p = N, M, P
    
    # Initialize with small random integers
    # Range [-3, 3] is a good start
    U_p = np.random.randint(-3, 4, (n * m, rank)).astype(np.int64)
    V_p = np.random.randint(-3, 4, (m * p, rank)).astype(np.int64)
    W_p = np.random.randint(-3, 4, (p * n, rank)).astype(np.int64)
    
    U3 = U_p.reshape(n, m, rank)
    V3 = V_p.reshape(m, p, rank)
    W3 = W_p.reshape(p, n, rank)
    
    # Precompute S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is O(n*m*p*rank)
    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    target = 8
    # Error E = sum (S - target)^2
    # We don't need E explicitly if we compute delta E, but useful for tracking
    E = np.sum((S - target) ** 2)
    
    if E == 0:
        return U_p / 2.0, V_p / 2.0, W_p / 2.0

    # Precompute products for fast delta E
    # P_U[i,j,r,k] = V[j,k,r] * W[k,i,r]
    # P_V[j,k,r,i] = U[i,j,r] * W[k,i,r]  -- Note: W index is k,i
    # P_W[k,i,r,j] = U[i,j,r] * V[j,k,r]
    
    # To save memory and time, we can compute these on the fly or precompute partials.
    # Since n,m,p are small (3,4,5), precomputing is fine.
    # P_U shape: (n, m, rank, p)
    P_U = np.einsum('jkr,kir->ijrk', V3, W3) # Wait, V3 is (m,p,rank), W3 is (p,n,rank)
    # V3[j,k,r] * W3[k,i,r]
    # Indices: j (m), k (p), r (rank), i (n)
    # Result: i (n), j (m), r (rank), k (p)
    P_U = np.einsum('jkr,kir->ijrk', V3, W3)
    
    # P_V[j,k,r,i] = U[i,j,r] * W[k,i,r]
    # U3[i,j,r] * W3[k,i,r]
    # Result: j (m), k (p), r (rank), i (n)
    P_V = np.einsum('ijr,kir->jkri', U3, W3)
    
    # P_W[k,i,r,j] = U[i,j,r] * V[j,k,r]
    # U3[i,j,r] * V3[j,k,r]
    # Result: k (p), i (n), r (rank), j (m)
    P_W = np.einsum('ijr,jkr->kirj', U3, V3)
    
    def get_delta_E_U(i, j, r, delta):
        # Change U[i,j,r] by delta
        # Affects S[i,j,k] for all k
        # Delta S_k = delta * V[j,k,r] * W[k,i,r] = delta * P_U[i,j,r,k]
        # Delta E = sum_k (S[i,j,k] + Delta S_k - target)^2 - (S[i,j,k] - target)^2
        S_ijk = S[i, j, :] # shape (p,)
        delta_S = delta * P_U[i, j, r, :] # shape (p,)
        new_S = S_ijk + delta_S
        old_err = (S_ijk - target) ** 2
        new_err = (new_S - target) ** 2
        return np.sum(new_err - old_err)

    def get_delta_E_V(j, k, r, delta):
        # Change V[j,k,r] by delta
        # Affects S[i,j,k] for all i
        # Delta S_i = delta * U[i,j,r] * W[k,i,r] = delta * P_V[j,k,r,i]
        S_ijk = S[:, j, k] # shape (n,)
        delta_S = delta * P_V[j, k, r, :] # shape (n,)
        new_S = S_ijk + delta_S
        old_err = (S_ijk - target) ** 2
        new_err = (new_S - target) ** 2
        return np.sum(new_err - old_err)

    def get_delta_E_W(k, i, r, delta):
        # Change W[k,i,r] by delta
        # Affects S[i,j,k] for all j
        # Delta S_j = delta * U[i,j,r] * V[j,k,r] = delta * P_W[k,i,r,j]
        S_ijk = S[i, :, k] # shape (m,)
        delta_S = delta * P_W[k, i, r, :] # shape (m,)
        new_S = S_ijk + delta_S
        old_err = (S_ijk - target) ** 2
        new_err = (new_S - target) ** 2
        return np.sum(new_err - old_err)

    def apply_U(i, j, r, delta):
        U3[i, j, r] += delta
        U_p[i*m + j, r] += delta
        # Update S
        S[i, j, :] += delta * P_U[i, j, r, :]
        # Update P_V and P_W which depend on U
        # P_V[j,k,r,i] = U[i,j,r] * W[k,i,r]
        # P_V[:, :, r, i] += delta * W3[:, i, r]
        P_V[:, :, r, i] += delta * W3[:, i, r]
        # P_W[k,i,r,j] = U[i,j,r] * V[j,k,r]
        # P_W[:, i, r, j] += delta * V3[j, :, r]
        P_W[:, i, r, j] += delta * V3[j, :, r]
        # P_U does not depend on U

    def apply_V(j, k, r, delta):
        V3[j, k, r] += delta
        V_p[j*p + k, r] += delta
        S[:, j, k] += delta * P_V[j, k, r, :]
        # P_U[i,j,r,k] = V[j,k,r] * W[k,i,r]
        # P_U[:, j, r, k] += delta * W3[k, :, r]
        P_U[:, j, r, k] += delta * W3[k, :, r]
        # P_W[k,i,r,j] = U[i,j,r] * V[j,k,r]
        # P_W[k, :, r, j] += delta * U3[:, j, r]
        P_W[k, :, r, j] += delta * U3[:, j, r]
        # P_V does not depend on V

    def apply_W(k, i, r, delta):
        W3[k, i, r] += delta
        W_p[k*n + i, r] += delta
        S[i, :, k] += delta * P_W[k, i, r, :]
        # P_U[i,j,r,k] = V[j,k,r] * W[k,i,r]
        # P_U[i, :, r, k] += delta * V3[:, k, r]
        P_U[i, :, r, k] += delta * V3[:, k, r]
        # P_V[j,k,r,i] = U[i,j,r] * W[k,i,r]
        # P_V[:, k, r, i] += delta * U3[i, :, r]
        P_V[:, k, r, i] += delta * U3[i, :, r]
        # P_W does not depend on W

    deltas = [-2, -1, 1, 2]
    stuck_count = 0
    max_stuck = 1000
    
    while _time_left() > 1.0:
        if E == 0:
            return U_p / 2.0, V_p / 2.0, W_p / 2.0
            
        # Pick random move
        choice = np.random.randint(3)
        best_delta = 0
        best_delta_E = 0
        best_apply = None
        
        if choice == 0:
            i = np.random.randint(n)
            j = np.random.randint(m)
            r = np.random.randint(rank)
            for d in deltas:
                dE = get_delta_E_U(i, j, r, d)
                if dE < best_delta_E:
                    best_delta_E = dE
                    best_delta = d
            if best_delta != 0:
                best_apply = lambda: apply_U(i, j, r, best_delta)
        elif choice == 1:
            j = np.random.randint(m)
            k = np.random.randint(p)
            r = np.random.randint(rank)
            for d in deltas:
                dE = get_delta_E_V(j, k, r, d)
                if dE < best_delta_E:
                    best_delta_E = dE
                    best_delta = d
            if best_delta != 0:
                best_apply = lambda: apply_V(j, k, r, best_delta)
        else:
            k = np.random.randint(p)
            i = np.random.randint(n)
            r = np.random.randint(rank)
            for d in deltas:
                dE = get_delta_E_W(k, i, r, d)
                if dE < best_delta_E:
                    best_delta_E = dE
                    best_delta = d
            if best_delta != 0:
                best_apply = lambda: apply_W(k, i, r, best_delta)
                
        if best_apply:
            best_apply()
            E += best_delta_E
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > max_stuck:
                # Kick: random changes
                for _ in range(5):
                    choice = np.random.randint(3)
                    d = np.random.choice(deltas)
                    if choice == 0:
                        i = np.random.randint(n)
                        j = np.random.randint(m)
                        r = np.random.randint(rank)
                        apply_U(i, j, r, d)
                    elif choice == 1:
                        j = np.random.randint(m)
                        k = np.random.randint(p)
                        r = np.random.randint(rank)
                        apply_V(j, k, r, d)
                    else:
                        k = np.random.randint(p)
                        i = np.random.randint(n)
                        r = np.random.randint(rank)
                        apply_W(k, i, r, d)
                    # Recalculate E after kick? Or just update incrementally?
                    # Incremental is safer but we didn't track dE for kicks.
                    # Let's just recompute S and E after a kick to be safe.
                    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                    E = np.sum((S - target) ** 2)
                    # Rebuild P arrays
                    P_U = np.einsum('jkr,kir->ijrk', V3, W3)
                    P_V = np.einsum('ijr,kir->jkri', U3, W3)
                    P_W = np.einsum('ijr,jkr->kirj', U3, V3)
                    stuck_count = 0

    if E == 0:
        return U_p / 2.0, V_p / 2.0, W_p / 2.0
    return None

def construct():
    """Search for rank-minimal exact half-integer tensor decomposition."""
    n, m, p = N, M, P
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook

    # Search for lower ranks
    # Lower bound is 20.
    # We try ranks from 40 downwards? No, we want MINIMUM.
    # So we try 20, 21, ...
    # But if 20 is impossible, we waste time.
    # Let's try a few random seeds for each rank.
    
    # Heuristic: Start from a slightly lower rank than schoolbook?
    # Actually, just iterate.
    for rank in range(20, best_rank + 1):
        if _time_left() < 10:
            break
            
        # Time budget per rank
        # If we have 100s left and 40 ranks, ~2.5s each.
        # Let's give 2s per rank, with 5 seeds.
        found = False
        for seed in range(5):
            if _time_left() < 5:
                break
            res = _search_rank(rank, seed, _time_left() - 2)
            if res is not None:
                U, V, W = res
                if _verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # If not found, continue to next rank
    return best_result
