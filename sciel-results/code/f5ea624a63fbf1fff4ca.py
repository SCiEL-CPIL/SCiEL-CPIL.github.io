"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Build rank-33 via Strassen blocks, then search for rank-32 by
    exploiting non-uniformity of middle dimension to merge one output column
    via linear combination.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Block b handles columns 2b, 2b+1 of B.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                
                col += 1
        
        return U, V, W

    # Start with best known
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = schoolbook()
        best_rank = 36

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Search for rank 32 using structured perturbation of the rank-33 solution
    # and random restarts informed by the structure.
    
    def try_find_rank32():
        """
        Attempt to find a rank-32 decomposition.
        Strategy 1: Structured search - perturb the standard parts of the rank-33
                    solution and try to absorb one term.
        Strategy 2: Random search with constraints from the tensor structure.
        """
        r = 32
        
        # Strategy 1: Try to merge two of the 12 standard terms by modifying
        # them and adjusting nearby Strassen terms.
        # The standard terms in block b are at columns 7*b+7 .. 7*b+10 (0-indexed).
        # Term indices within block: 0,1,2,3 correspond to (i,k) = (0,0),(0,1),(1,0),(1,1)
        
        # Idea: Can we replace two standard terms (i1,k1) and (i2,k2) in the same
        # block with a single term that, combined with a modified Strassen term,
        # gives the same result?
        
        # This is complex. Let's try a more direct approach:
        # Random search with structure-aware initialization.
        
        np.random.seed(42)
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with structure: mostly based on rank-33,
            # but with one column removed and others perturbed.
            
            # Start from rank-33 and try to "fold" one column into others.
            U33, V33, W33 = strassen_block_decomposition()
            
            # Pick a column to eliminate (prefer standard columns as they're simpler)
            # Standard columns are at positions: 7,8,9,10, 14,15,16,17, 21,22,23,24
            std_cols = [7,8,9,10, 14,15,16,17, 21,22,23,24]
            
            if attempt < 12:
                # Try eliminating each standard column
                elim_col = std_cols[attempt]
            else:
                # Try random columns
                elim_col = np.random.randint(0, 33)
            
            # Create rank-32 matrices by removing column elim_col
            mask = np.ones(33, dtype=bool)
            mask[elim_col] = False
            U32 = U33[:, mask].copy()
            V32 = V33[:, mask].copy()
            W32 = W33[:, mask].copy()
            
            # The difference tensor is:
            # D = U33[:,elim_col] * V33[:,elim_col] * W33[:,elim_col]
            # We need to absorb this into the remaining 32 terms.
            
            # This means we need to find vectors u', v', w' such that:
            # sum_{c=0}^{31} (U32[:,c] + du_c) * (V32[:,c] + dv_c) * (W32[:,c] + dw_c) = T
            # where du_c, dv_c, dw_c are small perturbations.
            
            # Linearizing: sum_c [U32[:,c]*dv_c*W32[:,c] + U32[:,c]*V32[:,c]*dw_c + du_c*V32[:,c]*W32[:,c]] = D
            # (ignoring second-order terms)
            
            # This is a linear system in du, dv, dw (3*32*... unknowns, 1296 equations).
            # Too many unknowns. Let's restrict perturbations to specific columns.
            
            # Try perturbing only 1-3 columns.
            if attempt % 3 == 0:
                # Perturb 1 column
                pert_cols = [np.random.randint(0, 32)]
            elif attempt % 3 == 1:
                # Perturb 2 columns
                pert_cols = list(np.random.choice(32, 2, replace=False))
            else:
                # Perturb 3 columns
                pert_cols = list(np.random.choice(32, 3, replace=False))
            
            # For simplicity, try adding a new rank-1 term that approximates D
            # by finding the best rank-1 approximation of D.
            
            D = np.einsum('i,j,k->ijk', U33[:,elim_col], V33[:,elim_col], W33[:,elim_col])
            
            # Find best rank-1 approximation of D: min ||D - u*v*w||_F
            # This is non-trivial. Use SVD-based approach:
            # D1 = D.reshape(dim_a, dim_b*dim_c)
            # U_svd, S_svd, Vt_svd = np.linalg.svd(D1, full_matrices=False)
            # u = U_svd[:,0] * S_svd[0]
            # vw = Vt_svd[0].reshape(dim_b, dim_c)
            # Then factor vw = v*w (another SVD)
            
            D1 = D.reshape(dim_a, dim_b * dim_c)
            U_svd, S_svd, Vt_svd = np.linalg.svd(D1, full_matrices=False)
            if S_svd[0] < 1e-10:
                continue
            u_approx = U_svd[:, 0] * S_svd[0]
            vw_approx = Vt_svd[0].reshape(dim_b, dim_c)
            
            # Factor vw_approx = v_approx * w_approx
            U2_svd, S2_svd, V2t_svd = np.linalg.svd(vw_approx, full_matrices=False)
            if S2_svd[0] < 1e-10:
                continue
            v_approx = U2_svd[:, 0] * S2_svd[0]
            w_approx = V2t_svd[0]
            
            # Add this as a new column? No, we need to REPLACE, not add.
            # So we need to modify existing columns to absorb D.
            
            # Try: add u_approx, v_approx, w_approx to one existing column
            # and check if the result is still valid.
            # This won't work directly because we'd be adding, not replacing.
            
            # Alternative: try random search from scratch with small entries.
            if attempt >= 12:
                U = np.random.randint(-1, 2, size=(dim_a, r))
                V = np.random.randint(-1, 2, size=(dim_b, r))
                W = np.random.randint(-1, 2, size=(dim_c, r))
                
                if verify_decomposition(U, V, W):
                    return U, V, W
        
        return None

    if time.time() < deadline - 20:
        result = try_find_rank32()
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 32

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
