import time
import numpy as np
from scipy.optimize import least_squares

DEADLINE_S = 110

def construct():
    start_time = time.time()
    n, m, p = 3, 3, 8
    nm, mp, pn = n*m, m*p, p*n

    def time_left():
        return DEADLINE_S - (time.time() - start_time)

    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, tol=1e-10):
        """
        Verifies if U, V, W decompose the matrix multiplication tensor for 3x3x2.
        U: (9, r), V: (6, r), W: (6, r)
        T[i,j,k] = sum_c U[i*3+j, c] * V[j*2+k, c] * W[k*3+i, c]
        """
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        r = U.shape[1]
        T_computed = np.zeros((3, 3, 2))
        
        # Pre-reshape for speed
        Us = U.reshape(3, 3, r)
        Vs = V.reshape(3, 2, r)
        Ws = W.reshape(2, 3, r)
        
        for i in range(3):
            for j in range(3):
                # T[i,j,k] = sum_c Us[i,j,c] * Vs[j,k,c] * Ws[k,i,c]
                # Vectorized over c and k
                # Us[i,j,:] * (Vs[j,:,c] * Ws[:,i,c]) summed over c
                # Let's just loop over c for clarity and moderate speed, or use einsum
                # T[i,j,:] += np.einsum('c,kc,c->k', Us[i,j,:], Vs[j,:,:], Ws[:,i,:])
                T_computed[i, j, :] += np.einsum('c,ck,ck->k', Us[i,j,:], Vs[j,:,:], Ws[:,i,:])
        
        target = np.ones((3, 3, 2))
        return np.max(np.abs(T_computed - target)) < tol

    def build_rank_17_block():
        """
        Constructs a rank-17 decomposition for 3x3x2.
        This uses 7 Strassen terms for a 2x2x2 sub-block and 10 schoolbook terms for the rest.
        """
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part for i in {0,1}, j in {0,1}, k in {0,1}
        # Indices for 2x2 block:
        # U: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
        # V: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # W: (0,0)->0, (0,1)->3, (1,0)->1, (1,1)->4  (Note: W is k*n+i, so (0,0)=0, (0,1)=1? No, W index is k*3+i)
        # W indices for k in {0,1}, i in {0,1}:
        # (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
        
        # Strassen Coefficients (Standard)
        # U
        U_s = [
            {0: 1, 4: 1}, # (0,0) + (1,1)
            {3: 1, 4: 1}, # (1,0) + (1,1)
            {0: 1},       # (0,0)
            {4: 1},       # (1,1)
            {0: 1, 1: 1}, # (0,0) + (0,1)
            {3: 1, 0: -1},# (1,0) - (0,0)
            {1: 1, 4: -1} # (0,1) - (1,1)
        ]
        # V
        V_s = [
            {0: 1, 3: 1}, # (0,0) + (1,1)
            {0: 1},       # (0,0)
            {1: 1, 3: -1},# (0,1) - (1,1)
            {2: 1, 0: -1},# (1,0) - (0,0)
            {3: 1},       # (1,1)
            {0: 1, 1: 1}, # (0,0) + (0,1)
            {2: 1, 3: 1}  # (1,0) + (1,1)
        ]
        # W
        W_s = [
            {0: 1, 4: 1}, # (0,0) + (1,1)  (k=0,i=0) + (k=1,i=1)
            {3: 1, 4: -1},# (1,0) - (1,1)  (k=1,i=0) - (k=1,i=1)
            {1: 1, 4: 1}, # (0,1) + (1,1)  (k=0,i=1) + (k=1,i=1)
            {0: 1, 3: 1}, # (0,0) + (1,0)  (k=0,i=0) + (k=1,i=0)
            {0: -1, 1: 1},# -(0,0) + (0,1) (k=0,i=0) + (k=0,i=1)
            {4: 1},       # (1,1)          (k=1,i=1)
            {0: 1}        # (0,0)          (k=0,i=0)
        ]
        
        for l in range(7):
            for idx, val in U_s[l].items():
                U[idx, col] += val
            for idx, val in V_s[l].items():
                V[idx, col] += val
            for idx, val in W_s[l].items():
                W[idx, col] += val
            col += 1
        
        # Schoolbook part for remaining (i,j,k)
        # Covered: i in {0,1}, j in {0,1}, k in {0,1}
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i*3+j, col] = 1.0
                    V[j*2+k, col] = 1.0
                    W[k*3+i, col] = 1.0
                    col += 1
                    
        return U, V, W

    def try_find_rank_16_block():
        """
        Attempts to find a rank-16 decomposition for 3x3x2.
        Uses optimization to collapse one dimension of the rank-17 solution.
        """
        U17, V17, W17 = build_rank_17_block()
        
        # Strategy: Initialize with a rank-16 parameterization derived from rank-17.
        # We assume the last term is linearly dependent or can be absorbed.
        # Let's try to optimize a rank-16 tensor directly, initializing near the rank-17 solution.
        
        target_T = np.ones((3, 3, 2))
        
        def compute_residual(x):
            """
            x contains flattened U (9x16), V (6x16), W (6x16)
            """
            U = x[:9*16].reshape(9, 16)
            V = x[9*16:9*16+6*16].reshape(6, 16)
            W = x[9*16+6*16:].reshape(6, 16)
            
            Us = U.reshape(3, 3, 16)
            Vs = V.reshape(3, 2, 16)
            Ws = W.reshape(2, 3, 16)
            
            T_comp = np.zeros((3, 3, 2))
            for i in range(3):
                for j in range(3):
                    T_comp[i, j, :] += np.einsum('c,ck,ck->k', Us[i,j,:], Vs[j,:,:], Ws[:,i,:])
            
            return (T_comp - target_T).flatten()
        
        def round_and_verify(U, V, W):
            """
            Tries to round the continuous solution to half-integers or integers.
            """
            scales = [1, 2, 4]
            for s in scales:
                Ur = np.round(U * s) / s
                Vr = np.round(V * s) / s
                Wr = np.round(W * s) / s
                if verify(Ur, Vr, Wr):
                    return Ur, Vr, Wr
            return None

        best_solution = None
        best_error = np.inf
        
        # Try multiple initializations
        rng = np.random.default_rng(123)
        
        for init_idx in range(10):
            if time_left() < 10:
                break
                
            # Initialize U, V, W
            U0 = U17[:, :16].copy()
            V0 = V17[:, :16].copy()
            W0 = W17[:, :16].copy()
            
            # Add noise to escape local minima
            noise_scale = 0.1 if init_idx > 0 else 0.0
            if noise_scale > 0:
                U0 += rng.normal(0, noise_scale, U0.shape)
                V0 += rng.normal(0, noise_scale, V0.shape)
                W0 += rng.normal(0, noise_scale, W0.shape)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # Use LM solver
                res = least_squares(compute_residual, x0, method='lm', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                error = np.max(np.abs(res.fun))
                
                if error < best_error:
                    best_error = error
                    best_solution = res.x
                    
                    if error < 1e-10:
                        U_opt = res.x[:9*16].reshape(9, 16)
                        V_opt = res.x[9*16:9*16+6*16].reshape(6, 16)
                        W_opt = res.x[9*16+6*16:].reshape(6, 16)
                        
                        rounded = round_and_verify(U_opt, V_opt, W_opt)
                        if rounded is not None:
                            return rounded[0], rounded[1], rounded[2]
                            
            except Exception:
                pass
        
        # If we found a continuous solution with very low error, try harder rounding
        if best_solution is not None and best_error < 1e-8:
            U_opt = best_solution[:9*16].reshape(9, 16)
            V_opt = best_solution[9*16:9*16+6*16].reshape(6, 16)
            W_opt = best_solution[9*16+6*16:].reshape(6, 16)
            
            # Try a grid search around the continuous solution for small integer/half-integer adjustments
            # This is expensive, so we only do it if the continuous error is tiny.
            # Just try standard rounding first.
            rounded = round_and_verify(U_opt, V_opt, W_opt)
            if rounded is not None:
                return rounded[0], rounded[1], rounded[2]
                
        return None

    # 1. Try to find a rank-16 block
    block_16 = None
    if time_left() > 20:
        block_16 = try_find_rank_16_block()
    
    if block_16 is not None:
        U_block, V_block, W_block = block_16
        rank_block = 16
    else:
        U_block, V_block, W_block = build_rank_17_block()
        rank_block = 17

    # 2. Assemble full 3x3x8 decomposition
    # The problem is 3x3 * 3x8.
    # We can view this as 4 independent 3x3 * 3x2 multiplications.
    # The tensor T_full[i, j, k] for i in [0,3), j in [0,3), k in [0,8)
    # T_full[i, j, k] = 1 if k corresponds to the same column in A and B?
    # Matrix Mult: C = A B. A is 3x3, B is 3x8.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T[i, j, k] = 1.
    
    rank_total = 4 * rank_block
    U_full = np.zeros((9, rank_total))
    V_full = np.zeros((24, rank_total)) # 3*8
    W_full = np.zeros((24, rank_total)) # 8*3
    
    for g in range(4):
        # g corresponds to the block of 2 columns in B (and C)
        # k_global in {2g, 2g+1}
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # U is the same for all blocks (A is shared)
        U_full[:, col_start:col_end] = U_block
        
        # V: Rows are j*8 + k_global. Block V rows are j*2 + k_local.
        for j in range(3):
            for k_local in range(2):
                k_global = 2 * g + k_local
                V_full[j*8 + k_global, col_start:col_end] = V_block[j*2 + k_local, :]
                
        # W: Rows are k_global*3 + i. Block W rows are k_local*3 + i.
        for k_local in range(2):
            k_global = 2 * g + k_local
            for i in range(3):
                W_full[k_global*3 + i, col_start:col_end] = W_block[k_local*3 + i, :]

    # 3. Verify and Return
    if verify(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback: Check if rounding fixes it (shouldn't be necessary for deterministic constructions)
        U_r = np.round(U_full * 2) / 2
        V_r = np.round(V_full * 2) / 2
        W_r = np.round(W_full * 2) / 2
        if verify(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Last resort: Schoolbook
        return schoolbook()

if __name__ == "__main__":
    # Basic sanity check
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("U shape:", (len(res['U']), len(res['U'][0])))
    print("V shape:", (len(res['V']), len(res['V'][0])))
    print("W shape:", (len(res['W']), len(res['W'][0])))
