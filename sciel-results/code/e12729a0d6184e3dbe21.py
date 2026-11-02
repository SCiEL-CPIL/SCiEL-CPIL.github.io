import time

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # We are constructing a rank 32 decomposition.
    # This is composed of:
    # 4 blocks of M_222 (rank 7) for columns 0-1, 2-3, 4-5, 6-7
    # 1 block of M_221 (rank 4) for column 8
    # Total rank = 4*7 + 4 = 32.

    rank = 32

    U = [[0] * rank for _ in range(n * m)] # 4 x 32
    V = [[0] * rank for _ in range(m * p)] # 18 x 32
    W = [[0] * rank for _ in range(p * n)] # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M2
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        # M3
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M4
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        # M5
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        # M6
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Cols 0-1
    add_strassen(0, 1)
    # Cols 2-3
    add_strassen(2, 3)
    # Cols 4-5
    add_strassen(4, 5)
    # Cols 6-7
    add_strassen(6, 7)
    # Col 8
    add_trivial(8)

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Try to find a rank-31 decomposition via local search
    # The matrix multiplication tensor T[i,j,k] = delta_{i,k} for the (n,m,p) = (2,2,9) case
    # has entries T[i,j,k] = 1 if i==k (output row) and j is the column index in B.
    # Actually, the tensor for C = A*B where A is 2x2, B is 2x9:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) where we need A[i,j]*B[j,k] to contribute to C[i,k].
    # The tensor T has shape (n*m, m*p, p*n) = (4, 18, 18)
    # T[a, b, c] where a = i*m+j (row of A, col of A), b = j*p+k (row of B, col of B), c = k*n+i (row of C, col of C)
    # T[a,b,c] = 1 if the indices are consistent: i from a, j from a, j from b, k from b, k from c, i from c
    # i = a // m, j_a = a % m, j_b = b // p, k = b % p, k_c = c // n, i_c = c % n
    # T = 1 if j_a == j_b and k == k_c and i == i_c

    # Build the tensor as a flat list of (a,b,c) triples where T=1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))

    # Rank-31 search: try to find U', V', W' of shape (4,31), (18,31), (18,31)
    # such that sum_r U'[a,r]*V'[b,r]*W'[c,r] = T[a,b,c] for all (a,b,c) in tensor_entries

    # Use a simple random restart + gradient-free local search
    # Objective: sum over all tensor entries of (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2

    import random
    import math

    target_rank = 31
    search_start = time.time()
    max_search_time = 60.0  # Leave 50s margin

    # Initialize with the rank-32 solution, dropping one column
    # Try dropping each of the 32 columns and see which gives the best residual
    best_residual = float('inf')
    best_init = None

    # Pre-compute the tensor as numpy arrays for speed
    try:
        import numpy as np
        use_numpy = True
        tensor_a = np.array([e[0] for e in tensor_entries])
        tensor_b = np.array([e[1] for e in tensor_entries])
        tensor_c = np.array([e[2] for e in tensor_entries])
        tensor_val = np.ones(len(tensor_entries))
    except ImportError:
        use_numpy = False

    if use_numpy:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Try dropping each column
        for drop_col in range(rank):
            if time.time() - search_start > max_search_time:
                break
            U_test = np.delete(U_np, drop_col, axis=1)
            V_test = np.delete(V_np, drop_col, axis=1)
            W_test = np.delete(W_np, drop_col, axis=1)

            # Compute residual
            # For each tensor entry (a,b,c), compute sum_r U[a,r]*V[b,r]*W[c,r]
            # Use broadcasting
            # U[tensor_a, :] has shape (N, 31)
            # V[tensor_b, :] has shape (N, 31)
            # W[tensor_c, :] has shape (N, 31)
            prod = U_np[tensor_a, :] * V_np[tensor_b, :] * W_np[tensor_c, :]
            # Drop the column
            prod = np.delete(prod, drop_col, axis=1)
            vals = prod.sum(axis=1)
            residual = np.sum((vals - tensor_val) ** 2)

            if residual < best_residual:
                best_residual = residual
                best_init = (U_test.copy(), V_test.copy(), W_test.copy())

        if best_init is not None and best_residual < 1e-10:
            # Found an exact rank-31 decomposition!
            best_U = best_init[0].tolist()
            best_V = best_init[1].tolist()
            best_W = best_init[2].tolist()
            best_rank = target_rank
            return {"U": best_U, "V": best_V, "W": best_W}

        # If no exact solution found by dropping a column, try a local search
        # Start from the best drop-column initialization
        if best_init is not None:
            U_cur = best_init[0].copy()
            V_cur = best_init[1].copy()
            W_cur = best_init[2].copy()

            # Vectorized gradient descent
            # Precompute indices for fast access
            # We want to compute gradients for all U, V, W entries efficiently
            
            # Residual R = sum_{entries} (Pred - 1)^2
            # Pred_e = sum_r U[a_e, r] V[b_e, r] W[c_e, r]
            
            lr = 0.05
            best_search_residual = best_residual
            
            for iteration in range(1000):
                if time.time() - search_start > max_search_time:
                    break

                # Compute predictions for all entries
                # U[tensor_a, :] shape (N, R)
                # V[tensor_b, :] shape (N, R)
                # W[tensor_c, :] shape (N, R)
                UA = U_cur[tensor_a, :]
                VB = V_cur[tensor_b, :]
                WC = W_cur[tensor_c, :]
                
                # Prod shape (N, R)
                Prod = UA * VB * WC
                Pred = Prod.sum(axis=1)
                Err = Pred - 1.0
                
                # Gradient w.r.t U[a, r]
                # dR/dU[a,r] = 2 * sum_{e: a_e=a} Err_e * V[b_e, r] * W[c_e, r]
                # We can accumulate this using bincount or similar, but simple loops over N=36 are fast enough
                # Or use np.einsum?
                
                # Let's just compute the gradient terms for the active indices
                # Since N is small (36), we can loop over entries and accumulate gradients
                # But vectorizing the update is better.
                
                # grad_U[a, r] += 2 * Err_e * VB[e, r] * WC[e, r] for all e where a_e = a
                # This is hard to vectorize directly without advanced indexing accumulation.
                # However, we can compute the gradient for the specific rows involved.
                
                # Alternative: Stochastic Gradient Descent with vectorized inner product
                # Pick a batch or single sample
                idx = random.randint(0, len(tensor_entries) - 1)
                a, b, c = tensor_entries[idx]
                
                # Compute pred for this specific entry
                pred_val = np.dot(np.dot(U_cur[a, :], V_cur[b, :]), W_cur[c, :])
                err_val = pred_val - 1.0
                
                # Gradient for U[a, :]
                grad_U = err_val * V_cur[b, :] * W_cur[c, :]
                # Gradient for V[b, :]
                grad_V = err_val * U_cur[a, :] * W_cur[c, :]
                # Gradient for W[c, :]
                grad_W = err_val * U_cur[a, :] * V_cur[b, :]
                
                # Update
                U_cur[a, :] -= lr * grad_U
                V_cur[b, :] -= lr * grad_V
                W_cur[c, :] -= lr * grad_W
                
                # Check residual every 20 iterations
                if iteration % 20 == 0:
                    total_res = np.sum((Pred - 1.0)**2) # Re-use Pred from last full compute? No, Pred was for all entries but we only updated one.
                    # Recompute residual for all entries to be safe
                    UA = U_cur[tensor_a, :]
                    VB = V_cur[tensor_b, :]
                    WC = W_cur[tensor_c, :]
                    Pred = (UA * VB * WC).sum(axis=1)
                    total_res = np.sum((Pred - 1.0)**2)
                    
                    if total_res < best_search_residual:
                        best_search_residual = total_res
                        if total_res < 1e-10:
                            best_U = U_cur.tolist()
                            best_V = V_cur.tolist()
                            best_W = W_cur.tolist()
                            best_rank = target_rank
                            return {"U": best_U, "V": best_V, "W": best_W}
                        
                        # Anneal learning rate if we are improving
                        if total_res < 1.0:
                            lr *= 0.9

    # If no rank-31 found, return the rank-32 solution
    return {"U": best_U, "V": best_V, "W": best_W}
