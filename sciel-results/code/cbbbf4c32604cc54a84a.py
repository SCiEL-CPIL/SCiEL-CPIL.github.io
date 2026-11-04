import time
import numpy as np

# Configuration
N, M, P = 2, 3, 9
START_TIME = time.time()
TIME_LIMIT = 115.0

# EVOLVE-BLOCK-START
def construct():
    # Target: Find rank-minimal decomposition for 2x3x9
    # Strategy: Find rank-8 decomposition for 2x3x3, replicate 3 times.
    
    # 1. Build 2x3x3 Target Tensor
    n2, m2, p2 = 2, 3, 3
    T_233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T_233[i*m2+j, j*p2+k, k*n2+i] = 1.0

    # 2. Search for low-rank decomposition of T_233
    def find_decomposition(T, n, m, p, target_rank, max_trials=50):
        """
        Attempts to find a decomposition of T with rank target_rank.
        Returns (U, V, W) as integer matrices or None.
        """
        nm, mp, pn = n*m, m*p, p*n
        # T_flat shapes
        # U: (nm, r), V: (mp, r), W: (pn, r)
        
        for trial in range(max_trials):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            np.random.seed(42 + trial)
            r = target_rank
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            prev_error = np.inf
            converged = False
            
            # ALS iterations
            for it in range(500):
                if time.time() - START_TIME > TIME_LIMIT - 10:
                    break
                
                # Update U
                # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix V, W. Solve for U.
                # T[a, b, c] = U[a, :] dot (V[:, r] * W[:, r] for each r)
                # M_VW[b, c, r] = V[b, r] * W[c, r]
                M_VW = np.einsum('br,cr->bcr', V, W)
                # T_flat[a, (b,c)]
                T_flat = T.reshape(nm, mp*pn)
                M_VW_flat = M_VW.reshape(mp*pn, r)
                # We want U[a, :] such that U[a, :] @ M_VW_flat.T approx T_flat[a, :]
                # Actually: sum_r U[a,r] M_VW_flat[bc, r] = T_flat[a, bc]
                # So U[a, :] is the solution to U[a, :] @ M_VW_flat.T = T_flat[a, :]
                # U = T_flat @ pinv(M_VW_flat.T) ?
                # Let's use lstsq on the transposed system or direct.
                # System: M_VW_flat.T @ U.T = T_flat.T
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_VW_flat.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    break

                # Update V
                M_UW = np.einsum('ar,cr->acr', U, W)
                T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm*pn)
                M_UW_flat = M_UW.reshape(nm*pn, r)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_UW_flat.T, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    break

                # Update W
                M_UV = np.einsum('ar,br->abr', U, V)
                T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm*mp)
                M_UV_flat = M_UV.reshape(nm*mp, r)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_UV_flat.T, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    break

                # Check error
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.sum((T_rec - T)**2)
                
                if error < 1e-10:
                    converged = True
                    break
                elif error > prev_error * 1.5 and it > 10:
                    break
                prev_error = error
            
            if not converged:
                continue
            
            # Try to snap to rationals
            # Try denominators 1, 2, 3, 4, 6, 8, 12, 16, 24
            for D in [1, 2, 3, 4, 6, 8, 12, 16, 24]:
                U_int = np.round(U * D)
                V_int = np.round(V * D)
                W_int = np.round(W * D)
                
                # Check if close to integers
                if not (np.allclose(U, U_int/D) and np.allclose(V, V_int/D) and np.allclose(W, W_int/D)):
                    continue
                
                # Verify exact decomposition with integers
                # T_rec_int = sum_r U_int[:,r] V_int[:,r] W_int[:,r]
                # Should equal D^3 * T
                T_rec_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                target_int = (D**3) * T
                
                if np.allclose(T_rec_int, target_int, atol=1e-6):
                    # Success! Return integer matrices
                    return U_int, V_int, W_int, D

        return None

    # Try rank 8 first
    decomp_233 = find_decomposition(T_233, n2, m2, p2, 8, max_trials=100)
    
    if decomp_233:
        U233, V233, W233, D = decomp_233
        rank_233 = 8
        # Construct 2x3x9 decomposition
        rank_total = rank_233 * 3
        U = np.zeros((N*M, rank_total))
        V = np.zeros((M*P, rank_total))
        W = np.zeros((P*N, rank_total))
        
        for b in range(3):
            col_start = b * rank_233
            # U is same for all blocks
            U[:, col_start:col_start+rank_233] = U233
            
            # V: rows j*P + k
            for j in range(M):
                for k_local in range(3):
                    k = 3*b + k_local
                    v_row = j*P + k
                    v2_row = j*3 + k_local
                    V[v_row, col_start:col_start+rank_233] = V233[v2_row, :]
                    
            # W: rows k*N + i
            for k_local in range(3):
                k = 3*b + k_local
                for i in range(N):
                    w_row = k*N + i
                    w2_row = k_local*N + i
                    W[w_row, col_start:col_start+rank_233] = W233[w2_row, :]
        
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to Schoolbook Rank 54
        RANK = N * M * P
        U = np.zeros((N*M, RANK))
        V = np.zeros((M*P, RANK))
        W = np.zeros((P*N, RANK))
        
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i*M+j, col] = 1
                    V[j*P+k, col] = 1
                    W[k*N+i, col] = 1
                    col += 1
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

# EVOLVE-BLOCK-END
