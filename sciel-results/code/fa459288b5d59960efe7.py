import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 118

    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    best = make_schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for vectorized computation
    # For each (i,j,k) triple, we need U[i*m+j], V[j*p+k], W[k*n+i]
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can compute this as: for each c, outer product of U[:,c], V[:,c], W[:,c] reshaped
    # More efficiently: use einsum or manual broadcasting

    # Precompute the index mapping
    idx_U = np.array([i*m + j for i in range(n) for j in range(m)])  # shape (n*m,)
    idx_V = np.array([j*p + k for j in range(m) for k in range(p)])  # shape (m*p,)
    idx_W = np.array([k*n + i for k in range(p) for i in range(n)])  # shape (p*n,)

    # For each (i,j,k), the indices into U, V, W are:
    # U_idx[i,j,k] = i*m+j, V_idx[i,j,k] = j*p+k, W_idx[i,j,k] = k*n+i
    I, J, K = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    U_idx = (I * m + J).ravel()  # shape (n*m*p,)
    V_idx = (J * p + K).ravel()  # shape (n*m*p,)
    W_idx = (K * n + I).ravel()  # shape (n*m*p,)

    def get_smooth_error_np(U_np, V_np, W_np):
        """Vectorized smooth error using NumPy"""
        # U_np shape: (n*m, rank)
        # V_np shape: (m*p, rank)
        # W_np shape: (p*n, rank)
        # For each (i,j,k) and each c: U[U_idx,c] * V[V_idx,c] * W[W_idx,c]
        # Sum over c, then compute (sum - 1)^2 and sum over all (i,j,k)
        # U_sel: (n*m*p, rank)
        U_sel = U_np[U_idx, :]  # (80, rank)
        V_sel = V_np[V_idx, :]  # (80, rank)
        W_sel = W_np[W_idx, :]  # (80, rank)
        # Product: (80, rank)
        prod = U_sel * V_sel * W_sel
        # Sum over rank: (80,)
        sums = prod.sum(axis=1)
        # Squared error
        return float(np.sum((sums - 1.0) ** 2))

    def get_error_count_np(U_np, V_np, W_np, tol=1e-9):
        """Count entries with |sum - 1| > tol"""
        U_sel = U_np[U_idx, :]
        V_sel = V_np[V_idx, :]
        W_sel = W_np[W_idx, :]
        prod = U_sel * V_sel * W_sel
        sums = prod.sum(axis=1)
        return int(np.sum(np.abs(sums - 1.0) > tol))

    def local_search_smooth(rank, U, V, W, seed, max_steps=200):
        """Local search using smooth error as objective, then verify with exact count"""
        rng = random.Random(seed)
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)

        cur_smooth = get_smooth_error_np(U_np, V_np, W_np)
        cur_err = get_error_count_np(U_np, V_np, W_np)
        if cur_err == 0:
            return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}

        best_err = cur_err
        no_improve = 0

        # Candidate values for each entry
        candidate_vals = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25, 1.5, -1.5]

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if cur_err == 0:
                break

            # Choose a random factor and entry to modify
            choice = rng.random()
            if choice < 0.34:
                idx = rng.randrange(n*m)
                c = rng.randrange(rank)
                old = U_np[idx, c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if abs(val - old) < 1e-12:
                        continue
                    U_np[idx, c] = val
                    new_smooth = get_smooth_error_np(U_np, V_np, W_np)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    U_np[idx, c] = best_candidate
                    new_err = get_error_count_np(U_np, V_np, W_np)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        U_np[idx, c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            elif choice < 0.67:
                idx = rng.randrange(m*p)
                c = rng.randrange(rank)
                old = V_np[idx, c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if abs(val - old) < 1e-12:
                        continue
                    V_np[idx, c] = val
                    new_smooth = get_smooth_error_np(U_np, V_np, W_np)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    V_np[idx, c] = best_candidate
                    new_err = get_error_count_np(U_np, V_np, W_np)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        V_np[idx, c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                idx = rng.randrange(p*n)
                c = rng.randrange(rank)
                old = W_np[idx, c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if abs(val - old) < 1e-12:
                        continue
                    W_np[idx, c] = val
                    new_smooth = get_smooth_error_np(U_np, V_np, W_np)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    W_np[idx, c] = best_candidate
                    new_err = get_error_count_np(U_np, V_np, W_np)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        W_np[idx, c] = old
                        no_improve += 1
                else:
                    no_improve += 1

            if no_improve > 60:
                break

        if cur_err == 0:
            return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
        return None

    def structured_init(rank, rng):
        """Initialize with structure based on matrix multiplication patterns"""
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]

        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    if rng.random() < 0.5:
                        U[i*m+j][c] = float(rng.choice([1, -1, 0.5, -0.5]))
            for j in range(m):
                for k in range(p):
                    if rng.random() < 0.5:
                        V[j*p+k][c] = float(rng.choice([1, -1, 0.5, -0.5]))
            for k in range(p):
                for i in range(n):
                    if rng.random() < 0.5:
                        W[k*n+i][c] = float(rng.choice([1, -1, 0.5, -0.5]))

        return U, V, W

    def schoolbook_init(rank, rng):
        """Initialize from schoolbook (rank 80) by keeping first 'rank' columns and randomizing the rest"""
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]

        # Copy first min(rank, 80) columns from schoolbook
        n_copy = min(rank, n*m*p)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if c < n_copy:
                        U[i*m+j][c] = 1.0
                        V[j*p+k][c] = 1.0
                        W[k*n+i][c] = 1.0
                    c += 1

        # For remaining columns, add random small perturbations
        for c in range(n_copy, rank):
            for i in range(n):
                for j in range(m):
                    if rng.random() < 0.3:
                        U[i*m+j][c] = float(rng.choice([1, -1, 0.5, -0.5]))
            for j in range(m):
                for k in range(p):
                    if rng.random() < 0.3:
                        V[j*p+k][c] = float(rng.choice([1, -1, 0.5, -0.5]))
            for k in range(p):
                for i in range(n):
                    if rng.random() < 0.3:
                        W[k*n+i][c] = float(rng.choice([1, -1, 0.5, -0.5]))

        return U, V, W

    # Try ranks from 50 upward
    for rank in range(50, min(n*m*p, 66)):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break

        num_restarts = 25 if rank <= 55 else 15
        found = None

        for restart in range(num_restarts):
            if time.time() > deadline - 4:
                break
            seed = rank * 1000 + restart
            rng = random.Random(seed)

            # Alternate between different initialization strategies
            if restart % 3 == 0:
                U, V, W = schoolbook_init(rank, rng)
            elif restart % 3 == 1:
                U, V, W = structured_init(rank, rng)
            else:
                # Pure random
                U, V, W = structured_init(rank, random.Random(seed + 99999))

            result = local_search_smooth(rank, U, V, W, seed + 7777, max_steps=200)
            if result:
                found = result
                break

        if found:
            best = found
            best_rank = rank
            break

    return best
