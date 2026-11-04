"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36

    # Build the target tensor: T[a,b,c] = 1 iff (a,b,c) is a valid triple
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))

    triple_set = set(triples)
    num_triples = len(triples)  # 144

    # Build target tensor as a flat vector
    T = np.zeros(U_rows * V_rows * W_rows)
    for (a, b, c) in triples:
        idx = (a * V_rows + b) * W_rows + c
        T[idx] = 1.0

    # Schoolbook solution (rank 144)
    rank_school = n * m * p
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def check_exact(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the tensor."""
        # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # Target tensor
        T_target = np.zeros((U_rows, V_rows, W_rows))
        for (a, b, c) in triples:
            T_target[a, b, c] = 1.0
        return np.max(np.abs(T_recon - T_target)) < tol

    def round_and_check(U, V, W, scales=None):
        """Try rounding to various scales and check exactness."""
        if scales is None:
            scales = [1, 2, 4, 8, 16, 32]
        for scale in scales:
            inv = 1.0 / scale
            U_r = np.round(U * scale) * inv
            V_r = np.round(V * scale) * inv
            W_r = np.round(W * scale) * inv
            if check_exact(U_r, V_r, W_r, tol=1e-8):
                return U_r, V_r, W_r
        return None

    def als_decompose(target_rank, seed, max_iters=30, tol=1e-7,
                      init_U=None, init_V=None, init_W=None):
        """ALS decomposition of the matmul tensor."""
        rng = np.random.RandomState(seed)
        
        if init_U is not None:
            U = init_U.astype(float).copy()
        else:
            U = rng.randn(U_rows, target_rank) * 0.01
        if init_V is not None:
            V = init_V.astype(float).copy()
        else:
            V = rng.randn(V_rows, target_rank) * 0.01
        if init_W is not None:
            W = init_W.astype(float).copy()
        else:
            W = rng.randn(W_rows, target_rank) * 0.01

        # Precompute flat targets
        T_U_flat = np.zeros((U_rows, V_rows * W_rows))
        T_V_flat = np.zeros((V_rows, U_rows * W_rows))
        T_W_flat = np.zeros((W_rows, U_rows * V_rows))
        for (a, b, c) in triples:
            T_U_flat[a, b * W_rows + c] = 1.0
            T_V_flat[b, a * W_rows + c] = 1.0
            T_W_flat[c, a * V_rows + b] = 1.0

        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Update U
            VW = np.einsum('br,cr->bcr', V, W)  # (V_rows, W_rows, rank)
            B = VW.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)
            G = B @ B.T + 1e-10 * np.eye(target_rank)
            try:
                U = np.linalg.solve(G, B @ T_U_flat.T).T
            except:
                U = (np.linalg.pinv(G) @ (B @ T_U_flat.T)).T

            # Update V
            UW = np.einsum('ar,cr->acr', U, W)  # (U_rows, W_rows, rank)
            C = UW.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
            Gv = C @ C.T + 1e-10 * np.eye(target_rank)
            try:
                V = np.linalg.solve(Gv, C @ T_V_flat.T).T
            except:
                V = (np.linalg.pinv(Gv) @ (C @ T_V_flat.T)).T

            # Update W
            UV = np.einsum('ar,br->abr', U, V)  # (U_rows, V_rows, rank)
            D = UV.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
            Gw = D @ D.T + 1e-10 * np.eye(target_rank)
            try:
                W = np.linalg.solve(Gw, D @ T_W_flat.T).T
            except:
                W = (np.linalg.pinv(Gw) @ (D @ T_W_flat.T)).T

            # Check error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            T_target = np.zeros((U_rows, V_rows, W_rows))
            for (a, b, c) in triples:
                T_target[a, b, c] = 1.0
            err = np.max(np.abs(T_recon - T_target))
            if err < tol:
                break
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err

        return U, V, W, err

    # Strategy: Try to find low-rank decomposition via ALS with various initializations
    # The matmul tensor for 4x4*4x9 has a natural block structure.
    # We can try block-structured initializations.
    
    # First, try a simple approach: use ALS with random init at decreasing ranks
    # Focus on ranks that might work based on known bounds
    # For 4x4*4x4, Strassen gives rank 49 (vs 64 schoolbook)
    # For 4x4*4x9 = 4x4*4x4 + 4x4*4x4 + 4x4*4x1
    # Best case: 49 + 49 + 16 = 114, but shared structure might help
    
    # Try ranks from low to high, with multiple seeds
    # Use structured initialization when possible
    
    # Precompute for ALS
    T_U_flat = np.zeros((U_rows, V_rows * W_rows))
    T_V_flat = np.zeros((V_rows, U_rows * W_rows))
    T_W_flat = np.zeros((W_rows, U_rows * V_rows))
    for (a, b, c) in triples:
        T_U_flat[a, b * W_rows + c] = 1.0
        T_V_flat[b, a * W_rows + c] = 1.0
        T_W_flat[c, a * V_rows + b] = 1.0

    # Try structured ranks
    # Theoretical lower bound: max(16, 36, 36) = 36, but realistically much higher
    # For 4x4*4x4, best known is 49 (Strassen)
    # For 4x4*4x9, we can do no better than ~114 with block decomposition
    # But maybe there's a smarter decomposition
    
    candidate_ranks = [114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142]
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break

        found = False
        for seed in range(8):
            if time.time() > deadline - 10:
                break
            if found:
                break

            # Use schoolbook as init (first target_rank columns)
            init_U = U_school[:, :target_rank]
            init_V = V_school[:, :target_rank]
            init_W = W_school[:, :target_rank]
            
            U, V, W, err = als_decompose(target_rank, seed, max_iters=25, tol=1e-8,
                                         init_U=init_U, init_V=init_V, init_W=init_W)
            
            if err < 1e-6:
                rounded = round_and_check(U, V, W)
                if rounded is not None:
                    best_U, best_V, best_W = rounded
                    best_rank = target_rank
                    found = True
                    break

            # Also try with random init
            if time.time() < deadline - 12:
                U, V, W, err = als_decompose(target_rank, seed + 1000, max_iters=20, tol=1e-7)
                if err < 1e-6:
                    rounded = round_and_check(U, V, W)
                    if rounded is not None:
                        best_U, best_V, best_W = rounded
                        best_rank = target_rank
                        found = True
                        break

        if found:
            # Try to go lower
            continue

    # If we improved, try to push even lower
    if best_rank < rank_school and time.time() < deadline - 30:
        for target_rank in range(best_rank - 2, max(100, best_rank - 15), -1):
            if time.time() > deadline - 15:
                break

            for seed in range(4):
                if time.time() > deadline - 10:
                    break

                U, V, W, err = als_decompose(target_rank, seed + 5000, max_iters=20, tol=1e-7,
                                             init_U=U_school[:, :target_rank] if target_rank <= rank_school else None,
                                             init_V=V_school[:, :target_rank] if target_rank <= rank_school else None,
                                             init_W=W_school[:, :target_rank] if target_rank <= rank_school else None)
                
                if err < 1e-6:
                    rounded = round_and_check(U, V, W)
                    if rounded is not None:
                        best_U, best_V, best_W = rounded
                        best_rank = target_rank
                        break
            else:
                continue
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
