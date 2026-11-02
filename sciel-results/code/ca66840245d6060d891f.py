"""Matrix multiplication 4x4 x 4x6 — Global Randomized Algebraic Search."""
import time
import sys
import numpy as np
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n, m, p = 4, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 16, 24, 24

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U3 = np.array(U).reshape(n, m, rank)
        V3 = np.array(V).reshape(m, p, rank)
        W3 = np.array(W).reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=1e-9)

    # Fallback: Build the known rank-81 solution (Strassen 4x4x4 + Schoolbook 4x4x2)
    def build_fallback_rank81():
        def build_strassen_444():
            A_c = [
                {(0,0): 1, (1,1): 1},
                {(1,0): 1, (1,1): 1},
                {(0,0): 1},
                {(1,1): 1},
                {(0,0): 1, (0,1): 1},
                {(1,0): 1, (0,0): -1},
                {(0,1): 1, (1,1): -1}
            ]
            B_c = [
                {(0,0): 1, (1,1): 1},
                {(0,0): 1},
                {(0,1): 1, (1,1): -1},
                {(1,0): 1, (0,0): -1},
                {(1,1): 1},
                {(0,0): 1, (0,1): 1},
                {(1,0): 1, (1,1): 1}
            ]
            C_c = [
                [[1, 0, 0, -1, 1, 0, -1],
                 [0, 0, 1, 0, 1, 0, 0]],
                [[0, 1, 0, 1, 0, 0, 0],
                 [1, -1, 1, 0, 0, 1, 0]]
            ]
            block_A = [
                [[1, 0], [0, 1]],
                [[0, 0], [1, 1]],
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, -1]],
                [[0, 1], [0, -1]]
            ]
            block_B = [
                [[1, 0], [0, 1]],
                [[1, 0], [0, 0]],
                [[0, 1], [0, -1]],
                [[0, -1], [1, 0]],
                [[0, 0], [0, 1]],
                [[1, 1], [0, 0]],
                [[0, 1], [1, 1]]
            ]

            r = 49
            U4 = [[0] * r for _ in range(16)]
            V4 = [[0] * r for _ in range(16)]
            W4 = [[0] * r for _ in range(16)]

            for beta in range(7):
                for alpha in range(7):
                    col = beta * 7 + alpha
                    for i in range(4):
                        for j in range(4):
                            bi, bj = i // 2, j // 2
                            coeff = 0
                            for rr in range(2):
                                for ss in range(2):
                                    if (rr, ss) in A_c[alpha]:
                                        ac = A_c[alpha][(rr, ss)]
                                        bc = block_A[beta][rr][ss]
                                        if rr == bi and ss == bj:
                                            coeff += ac * bc
                            U4[i * 4 + j][col] = coeff

                    for j in range(4):
                        for k in range(4):
                            bj, bk = j // 2, k // 2
                            coeff = 0
                            for rr in range(2):
                                for ss in range(2):
                                    if (rr, ss) in B_c[alpha]:
                                        bc = B_c[alpha][(rr, ss)]
                                        ac = block_B[beta][rr][ss]
                                        if rr == bj and ss == bk:
                                            coeff += bc * ac
                            V4[j * 4 + k][col] = coeff

                    for bi in range(2):
                        for bk in range(2):
                            for pp in range(2):
                                for qq in range(2):
                                    ci = 2 * bi + pp
                                    ck = 2 * bk + qq
                                    coeff = C_c[bi][bk][beta] * C_c[pp][qq][alpha]
                                    W4[ck * 4 + ci][col] += coeff

            return U4, V4, W4

        def build_schoolbook_442():
            r = 32
            U42 = [[0] * r for _ in range(16)]
            V42 = [[0] * r for _ in range(8)]
            W42 = [[0] * r for _ in range(8)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(2):
                        U42[i * 4 + j][col] = 1
                        V42[j * 2 + k][col] = 1
                        W42[k * 4 + i][col] = 1
                        col += 1
            return U42, V42, W42

        U4, V4, W4 = build_strassen_444()
        U42, V42, W42 = build_schoolbook_442()

        total_rank = 49 + 32
        U_tot = [[0] * total_rank for _ in range(16)]
        V_tot = [[0] * total_rank for _ in range(24)]
        W_tot = [[0] * total_rank for _ in range(24)]

        for i in range(16):
            for c in range(49):
                U_tot[i][c] = U4[i][c]
                V_tot[i][c] = V4[i][c]
                W_tot[i][c] = W4[i][c]

        for i in range(16):
            for c in range(32):
                U_tot[i][49 + c] = U42[i][c]
        for j in range(8):
            for c in range(32):
                V_tot[j][49 + c] = V42[j][c]
        for k in range(8):
            for c in range(32):
                W_tot[k][49 + c] = W42[k][c]

        return U_tot, V_tot, W_tot, total_rank

    # Pre-compute fallback
    best_U, best_V, best_W, best_rank = build_fallback_rank81()
    
    if not verify_fast(best_U, best_V, best_W, best_rank):
        # Should not happen, but fallback to schoolbook rank 96
        rank0 = n * m * p
        U = [[0] * rank0 for _ in range(nm)]
        V = [[0] * rank0 for _ in range(mp)]
        W = [[0] * rank0 for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}

    # Strategy: Try to find a rank < 81 decomposition via random search
    # We will search for ranks 80, 79, ... down to a minimum (e.g., 60)
    # If found, return it. Else return rank 81.
    
    # Objective function: || T(U,V,W) - 1 ||_F^2
    # T is (n, m, p) = (4, 4, 6) = 96 elements.
    # U: (16, r), V: (24, r), W: (24, r)
    
    def compute_error_sq(U, V, W):
        """Compute squared Frobenius norm of error."""
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = T - 1.0
        return np.sum(err * err)

    def random_init(r):
        # Initialize with small random values
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        return U, V, W

    def local_search(U, V, W, r, max_steps=500, tol=1e-6):
        """
        Perform gradient-free local search (Nelder-Mead style or simple perturbations).
        Given time constraints, we use a simple random perturbation hill-climbing approach
        on a subset of parameters to avoid high computational cost.
        """
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_err = compute_error_sq(current_U, current_V, current_W)
        
        if current_err < 1e-9:
            return current_U, current_V, current_W, current_err

        step_size = 0.1
        best_err = current_err
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        
        # We don't have time for full Nelder-Mead on ~ (16+24+24)*r params.
        # Instead, we do a few random restarts and limited coordinate descent-like moves.
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
                
            # Pick a random matrix and a random entry to perturb
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                idx = np.random.randint(nm * r)
                i, c = divmod(idx, r)
                # Perturb U[i, c]
                new_val = current_U[i, c] + np.random.randn() * step_size
                old_val = current_U[i, c]
                current_U[i, c] = new_val
            elif mat_choice == 1:
                idx = np.random.randint(mp * r)
                j, c = divmod(idx, r)
                new_val = current_V[j, c] + np.random.randn() * step_size
                old_val = current_V[j, c]
                current_V[j, c] = new_val
            else:
                idx = np.random.randint(pn * r)
                k, c = divmod(idx, r)
                new_val = current_W[k, c] + np.random.randn() * step_size
                old_val = current_W[k, c]
                current_W[k, c] = new_val

            new_err = compute_error_sq(current_U, current_V, current_W)
            
            if new_err < current_err:
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                # Accept move
            else:
                # Revert
                if mat_choice == 0:
                    current_U[i, c] = old_val
                elif mat_choice == 1:
                    current_V[j, c] = old_val
                else:
                    current_W[k, c] = old_val
            
            # Occasionally adjust step size
            if step % 50 == 0:
                step_size *= 0.9

        return best_U, best_V, best_W, best_err

    def try_find_decomposition(r, num_restarts=5):
        """Try to find a valid decomposition of rank r."""
        best_U_global, best_V_global, best_W_global = None, None, None
        best_err_global = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
            
            U, V, W = random_init(r)
            U, V, W, err = local_search(U, V, W, r, max_steps=200)
            
            if err < best_err_global:
                best_err_global = err
                best_U_global = U
                best_V_global = V
                best_W_global = W
                
            if err < 1e-6:
                # Found a very low error solution. Try to round to integers/half-integers.
                # This is a heuristic: if the continuous solution is close to integer, round it.
                # However, tensor decomposition coefficients are often integers.
                # Let's check if rounding works.
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                
                err_r = compute_error_sq(U_r, V_r, W_r)
                if err_r < 1e-9:
                    # Convert to lists of floats (or ints if exact)
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    # Verify again with lists
                    if verify_fast(U_list, V_list, W_list, r):
                        return U_list, V_list, W_list, r
                
                # Also try pure integer rounding
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                err_i = compute_error_sq(U_i, V_i, W_i)
                if err_i < 1e-9:
                    U_list = U_i.tolist()
                    V_list = V_i.tolist()
                    W_list = W_i.tolist()
                    if verify_fast(U_list, V_list, W_list, r):
                        return U_list, V_list, W_list, r
                        
        return None, None, None, best_err_global

    # Search for ranks lower than 81
    # We know 81 is valid. Try 80, 79, ...
    # Note: Random search is unlikely to succeed for small r without strong priors,
    # but we try a few ranks.
    
    # Given the difficulty, we might only find 81.
    # But let's try 80 and 79 with limited effort.
    
    for r in [80, 79, 78]:
        if time.time() > deadline - 30:
            break
            
        U_sol, V_sol, W_sol, found_rank = try_find_decomposition(r, num_restarts=3)
        if U_sol is not None:
            best_U, best_V, best_W, best_rank = U_sol, V_sol, W_sol, found_rank
            break

    # Final check
    if verify_fast(best_U, best_V, best_W, best_rank):
        return {'U': best_U, 'V': best_V, 'W': best_W}
    else:
        # Return fallback
        return {'U': best_U, 'V': best_V, 'W': best_W}

# EVOLVE-BLOCK-END
