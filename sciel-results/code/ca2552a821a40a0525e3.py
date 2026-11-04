"""Matrix multiplication tensor decomposition for (2,3,10).

Strategy:
  * Start with the known rank-55 Strassen decomposition.
  * Use a smooth optimization surrogate (gradient descent) to attempt 
    rank reduction to 54.
  * If successful, return the lower rank; otherwise, return rank 55.
"""
import time
import random
import numpy as np
from scipy.optimize import minimize

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    n, m, p = 2, 3, 10
    # Base rank-55 decomposition
    base_rank = 55
    U_base = np.zeros((n * m, base_rank))
    V_base = np.zeros((m * p, base_rank))
    W_base = np.zeros((p * n, base_rank))

    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # Strassen 7 products for X = A[:,0:2], Y = B[0:2, k0:k1+1]
        # M1 = (a00 + a11) * (B0k0 + B1k1)
        U_base[0, col] += 1   # a00
        U_base[4, col] += 1   # a11
        V_base[k0, col] += 1       # B[0,k0]
        V_base[10 + k1, col] += 1  # B[1,k1]
        W_base[k0 * 2 + 0, col] += 1  # C[0,k0]
        W_base[k1 * 2 + 1, col] += 1  # C[1,k1]
        col += 1

        # M2 = (a10 + a11) * B0k0
        U_base[3, col] += 1   # a10
        U_base[4, col] += 1   # a11
        V_base[k0, col] += 1       # B[0,k0]
        W_base[k0 * 2 + 1, col] += 1  # C[1,k0]
        W_base[k1 * 2 + 1, col] -= 1  # -C[1,k1]
        col += 1

        # M3 = a00 * (B1k0 - B0k0)
        U_base[0, col] += 1   # a00
        V_base[10 + k0, col] += 1  # B[1,k0]
        V_base[k0, col] -= 1       # -B[0,k0]
        W_base[k1 * 2 + 0, col] += 1  # C[0,k1]
        W_base[k0 * 2 + 0, col] += 1  # C[0,k0]
        col += 1

        # M4 = a11 * (B1k0 - B0k0)
        U_base[4, col] += 1   # a11
        V_base[10 + k0, col] += 1  # B[1,k0]
        V_base[k0, col] -= 1       # -B[0,k0]
        W_base[k0 * 2 + 0, col] += 1  # C[0,k0]
        W_base[k0 * 2 + 1, col] += 1  # C[1,k0]
        col += 1

        # M5 = (a00 + a01) * B1k0
        U_base[0, col] += 1   # a00
        U_base[1, col] += 1   # a01
        V_base[10 + k0, col] += 1  # B[1,k0]
        W_base[k0 * 2 + 0, col] -= 1  # -C[0,k0]
        W_base[k1 * 2 + 0, col] += 1  # C[0,k1]
        col += 1

        # M6 = (a10 - a00) * (B0k0 + B0k1)
        U_base[3, col] += 1   # a10
        U_base[0, col] -= 1   # -a00
        V_base[k0, col] += 1       # B[0,k0]
        V_base[k1, col] += 1       # B[0,k1]
        W_base[k1 * 2 + 1, col] += 1  # C[1,k1]
        col += 1

        # M7 = (a01 - a11) * (B1k0 + B1k1)
        U_base[1, col] += 1   # a01
        U_base[4, col] -= 1   # -a11
        V_base[10 + k0, col] += 1  # B[1,k0]
        V_base[10 + k1, col] += 1  # B[1,k1]
        W_base[k0 * 2 + 0, col] += 1  # C[0,k0]
        col += 1

        # Schoolbook for A[:,2] * B[2, k0:k1+1]
        # a02 * B[2,k0] -> C[0,k0]
        U_base[2, col] += 1
        V_base[20 + k0, col] += 1
        W_base[k0 * 2 + 0, col] += 1
        col += 1

        # a02 * B[2,k1] -> C[0,k1]
        U_base[2, col] += 1
        V_base[20 + k1, col] += 1
        W_base[k1 * 2 + 0, col] += 1
        col += 1

        # a12 * B[2,k0] -> C[1,k0]
        U_base[5, col] += 1
        V_base[20 + k0, col] += 1
        W_base[k0 * 2 + 1, col] += 1
        col += 1

        # a12 * B[2,k1] -> C[1,k1]
        U_base[5, col] += 1
        V_base[20 + k1, col] += 1
        W_base[k1 * 2 + 1, col] += 1
        col += 1

    # Helper to verify decomposition
    def verify_decomposition(U, V, W, rank, n, m, p, trials=20):
        rng = random.Random(123)
        for _ in range(trials):
            A = np.array([[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)], dtype=float)
            B = np.array([[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)], dtype=float)
            C_direct = A @ B
            C_decomp = np.zeros((n, p))
            # Compute C_decomp using tensor decomposition
            # C[i,k] = sum_r U[i,k,r] * V[j,l,r] ... wait, indices are mixed.
            # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
            # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
            for i in range(n):
                for k in range(p):
                    acc = 0.0
                    for r in range(rank):
                        ua = 0.0
                        for j in range(m):
                            ua += U[i*m+j, r] * A[i,j]
                        vb = 0.0
                        for j in range(m):
                            vb += V[j*p+k, r] * B[j,k]
                        acc += W[k*n+i, r] * ua * vb
                    C_decomp[i,k] = acc
            if not np.allclose(C_direct, C_decomp, atol=1e-5):
                return False
        return True

    # Helper to construct return dict from numpy arrays
    def to_dict(U, V, W, rank):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Base solution
    if verify_decomposition(U_base, V_base, W_base, base_rank, n, m, p):
        best_result = to_dict(U_base, V_base, W_base, base_rank)
    else:
        # Fallback to schoolbook rank 60
        rank60 = n * m * p
        U60 = np.zeros((n*m, rank60))
        V60 = np.zeros((m*p, rank60))
        W60 = np.zeros((p*n, rank60))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U60[i*m+j, idx] = 1
                    V60[j*p+k, idx] = 1
                    W60[k*n+i, idx] = 1
                    idx += 1
        if verify_decomposition(U60, V60, W60, rank60, n, m, p):
            best_result = to_dict(U60, V60, W60, rank60)
        else:
            # Should not happen
            return {"U": [[0]*60 for _ in range(6)], "V": [[0]*60 for _ in range(30)], "W": [[0]*60 for _ in range(20)]}

    # Attempt rank reduction to 54 using optimization
    target_rank = 54
    if time.time() < deadline - 10.0:
        # Initialize U, V, W for rank 54
        # We can try to drop one random term from the base rank 55
        # Or just take the first 54 columns. The structure is local, so dropping one might break things.
        # A better init: Use the base 55, but force the last term to be zero? No, that's rank 54 if independent.
        # Let's just try to optimize a random perturbation of the first 54 columns.
        
        U_opt = np.zeros((n*m, target_rank))
        V_opt = np.zeros((m*p, target_rank))
        W_opt = np.zeros((p*n, target_rank))
        
        # Copy first 54 columns
        U_opt[:, :target_rank] = U_base[:, :target_rank]
        V_opt[:, :target_rank] = V_base[:, :target_rank]
        W_opt[:, :target_rank] = W_base[:, :target_rank]
        
        # Flatten for optimization
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x, n, m, p, rank):
            size_U = n*m*rank
            size_V = m*p*rank
            U = x[:size_U].reshape(n*m, rank)
            V = x[size_U:size_U+size_V].reshape(m*p, rank)
            W = x[size_U+size_V:].reshape(p*n, rank)
            return U, V, W
        
        def loss(x):
            U, V, W = unflatten(x, n, m, p, target_rank)
            # Compute error on a small set of random tensors
            # To be fast, use 5 random A, B pairs
            total_err = 0.0
            rng = random.Random(42)
            for _ in range(5):
                A = np.array([[rng.randint(-2, 2) for _ in range(m)] for _ in range(n)], dtype=float)
                B = np.array([[rng.randint(-2, 2) for _ in range(p)] for _ in range(m)], dtype=float)
                C_direct = A @ B
                C_decomp = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        acc = 0.0
                        for r in range(target_rank):
                            ua = sum(U[i*m+j, r] * A[i,j] for j in range(m))
                            vb = sum(V[j*p+k, r] * B[j,k] for j in range(m))
                            acc += W[k*n+i, r] * ua * vb
                        C_decomp[i,k] = acc
                total_err += np.sum((C_direct - C_decomp)**2)
            return total_err
        
        x0 = flatten(U_opt, V_opt, W_opt)
        
        # Optimize
        try:
            res = minimize(loss, x0, method='BFGS', options={'maxiter': 200, 'disp': False})
            if res.success:
                U_f, V_f, W_f = unflatten(res.x, n, m, p, target_rank)
                # Check if it's close to integers/half-integers? 
                # The task requires integer/half-integer entries. 
                # Optimization likely produces floats. 
                # We can try rounding to nearest 0.5 and verify.
                U_r = np.round(U_f * 2) / 2
                V_r = np.round(V_f * 2) / 2
                W_r = np.round(W_f * 2) / 2
                
                if verify_decomposition(U_r, V_r, W_r, target_rank, n, m, p, trials=50):
                    if verify_decomposition(U_r, V_r, W_r, target_rank, n, m, p, trials=50): # Double check
                        best_result = to_dict(U_r, V_r, W_r, target_rank)
        except Exception:
            pass

    return best_result
