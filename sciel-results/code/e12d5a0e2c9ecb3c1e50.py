import numpy as np
import time

# Safety margin: The evaluator kills after 150s. We aim to finish in < 100s.
_DEADLINE = time.time() + 100

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate the target tensor T
    # T[i_A, i_B, i_C] = 1 if i_A = i*m+j, i_B = j*p+k, i_C = k*n+i
    # Dimensions: (n*m, m*p, p*n) -> (16, 28, 28)
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0

    # 2. Define the fallback (Current Best: Rank 93)
    # We reuse the logic from the current program to generate the fallback
    # This ensures we always have a valid answer if the search fails.
    fallback_result = generate_fallback_decomposition(n, m, p)
    fallback_rank = fallback_result['rank']
    
    # 3. Search for a lower rank decomposition
    # Strategy: Try ranks from (fallback_rank - 1) down to a lower bound.
    # We use a simple iterative improvement method (Gaussian elimination on the error)
    # or random restarts with gradient descent.
    
    # Given the complexity, a full gradient search might be too slow in Python.
    # We will try a "Null Space" approach on a small number of random initializations.
    
    best_rank_found = fallback_rank
    best_decomp = None

    # Try a few ranks below the fallback
    # Theoretical lower bound is ~50-60. Let's try down to 60.
    # To save time, we only try a few random seeds for each rank.
    
    search_ranks = range(fallback_rank - 1, max(50, fallback_rank - 15), -1)
    
    for R in search_ranks:
        if time.time() > _DEADLINE - 20: # Stop if running out of time
            break
            
        # Try a limited number of random initializations for this rank
        max_restarts = 5 if R > 70 else 10
        
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 20:
                break
                
            # Random initialization
            np.random.seed(42 + R + restart)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Iterative improvement (Simplified Landsberg-like step)
            # We want to minimize || T - sum(U_r V_r W_r) ||^2
            # This is equivalent to finding a low-rank tensor approximation.
            # Since we need EXACT match, we are looking for a zero-error solution.
            
            # Simple Gradient Descent on the error
            # Error E = T - einsum('ir,jr,kr->ijk', U, V, W)
            # dE/dU[i,r] = - sum_{j,k} V[j,r] W[k,r] * (T[i,j,k] - recon[i,j,k]) ... complex
            # 
            # Instead, let's use a "Block Relaxation" method.
            # Fix V and W, solve for U (least squares).
            # Fix U and W, solve for V.
            # Fix U and V, solve for W.
            
            # Reshape T to matrix form for least squares
            # For U: T_ij,k = sum_r U_ir V_jr W_kr
            # Let M_U be a matrix where row is i, col is (j,k).
            # T_flat[i, j*p*n + k] = sum_r U[i,r] * (V[j,r] * W[k,r])
            # This is not linear in U if V,W are fixed? Yes it is.
            # T[i,j,k] = sum_r U[i,r] * Z[j,k,r] where Z[j,k,r] = V[j,r]W[k,r]
            # So T_mat[i, :] = U[i, :] @ Z_mat.T ?
            # Let's construct the linear system.
            
            # Pre-compute V_W products? No, V and W change.
            
            # Let's try 50 iterations of alternating least squares
            # This might be too slow for R~90 in Python loops.
            # We need vectorized operations.
            
            # Reshape T to (n*m, m*p*p*n) ? No, that's huge.
            # T is 16 x 28 x 28.
            
            # Iteration 1: Solve for U
            # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
            # For fixed j, k: T[i, j, k] is linear in U[i, :].
            # We can solve for each column of U? No, U has rows i.
            # For a fixed i, U[i, :] is a vector of length R.
            # The equation is: T[i, :, :] = sum_r U[i, r] * outer(V[:, r], W[:, r])
            # Let O_r = outer(V[:, r], W[:, r]). Shape (m*p, p*n).
            # Then T[i, :, :] = sum_r U[i, r] * O_r.
            # This is a linear system: T_i = U_i @ O
            # Where T_i is T[i, :, :].reshape(1, -1) ? No.
            # T[i, j, k] = sum_r U[i, r] * O_r[j, k]
            # So T_i (vector of length m*p*p*n) = U_i (vector of length R) @ O_mat (R x m*p*p*n)
            # O_mat[r, j*p*n + k] = V[j, r] * W[k, r]
            
            # We can solve U = T @ O^+ (Pseudo-inverse)
            # But O is R x (m*p*p*n). m*p*p*n = 28*28 = 784.
            # R is ~90.
            # This is solvable with np.linalg.lstsq.
            
            # Let's implement one step of this.
            # If the error becomes 0, we are done.
            
            # To speed up, we only do this for a few iterations.
            
            success = False
            for step in range(20):
                if time.time() > _DEADLINE - 20:
                    break
                    
                # Build O_mat for U
                # O_mat[r, idx] = V[j, r] * W[k, r] where idx = j*p*n + k
                # V shape: (m*p, R) -> (28, R)
                # W shape: (p*n, R) -> (28, R)
                # Outer product of V and W for each r?
                # V_T = V.T (R, 28)
                # W_T = W.T (R, 28)
                # We need a matrix O (R, 28*28) such that O[r, j*28+k] = V[j,r]*W[k,r]
                # This is the Khatri-Rao product? Or just vectorized outer product.
                # V_r = V[:, r] (28, 1)
                # W_r = W[:, r] (28, 1)
                # outer(V_r, W_r) -> (28, 28)
                # Stack them -> (R, 28, 28) -> reshape (R, 784)
                
                O_mat = np.einsum('jr,kr->rjk', V, W).reshape(R, -1)
                
                # T_mat: (n*m, 784)
                T_mat = T_target.reshape(n*m, -1)
                
                # Solve T_mat = U @ O_mat
                # U = T_mat @ O_mat^T (O_mat O_mat^T)^-1
                # Use lstsq for stability
                try:
                    U_new, residuals, rank, sv = np.linalg.lstsq(O_mat, T_mat.T, rcond=None)
                    U = U_new.T
                except:
                    continue
                
                # Check error
                recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.max(np.abs(recon - T_target))
                if err < 1e-9:
                    success = True
                    break
                
                # Solve for V
                # T[i, j, k] = sum_r V[j, r] * (U[i, r] * W[k, r])
                # O_mat_V[r, i*p*n + k] = U[i, r] * W[k, r]
                O_mat_V = np.einsum('ir,kr->rik', U, W).reshape(R, -1)
                T_mat_V = T_target.transpose(1, 0, 2).reshape(m*p, -1) # (j, i, k) -> (j, i*p*n+k)
                try:
                    V_new, residuals, rank, sv = np.linalg.lstsq(O_mat_V, T_mat_V.T, rcond=None)
                    V = V_new.T
                except:
                    continue
                
                # Solve for W
                # T[i, j, k] = sum_r W[k, r] * (U[i, r] * V[j, r])
                # O_mat_W[r, i*m*p + j] = U[i, r] * V[j, r]
                O_mat_W = np.einsum('ir,jr->rij', U, V).reshape(R, -1)
                T_mat_W = T_target.transpose(2, 0, 1).reshape(p*n, -1) # (k, i, j) -> (k, i*m*p+j)
                try:
                    W_new, residuals, rank, sv = np.linalg.lstsq(O_mat_W, T_mat_W.T, rcond=None)
                    W = W_new.T
                except:
                    continue
                
                # Check error again
                recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.max(np.abs(recon - T_target))
                if err < 1e-9:
                    success = True
                    break

            if success:
                best_rank_found = R
                best_decomp = {"U": U, "V": V, "W": W}
                break # Found a decomposition for this rank, try lower?
                # Actually, if we found rank R, we can try R-1 immediately.
                # But the loop structure handles that.

    if best_decomp is not None:
        return {
            "U": best_decomp["U"].tolist(),
            "V": best_decomp["V"].tolist(),
            "W": best_decomp["W"].tolist()
        }
    else:
        # Return fallback
        return {
            "U": fallback_result["U"],
            "V": fallback_result["V"],
            "W": fallback_result["W"]
        }

def generate_fallback_decomposition(n, m, p):
    # This is the exact logic from the current program, extracted to ensure validity
    total_rank = 49 + 44
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    current_rank = 0
    
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c
    
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank
        
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj
                V[i * p + j, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj
                W[k_final * n + i_final, r] += cc
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0
        
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0

    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)
    
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": current_rank}
