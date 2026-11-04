import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Construct the target tensor T
    # T shape: (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
                
    # 2. Try to find a lower rank decomposition using ALS
    # We know rank 36 works. We try 32, 33, 34, 35.
    # If we find 32, we stop.
    
    best_rank = nm * mp
    best_factors = None
    
    # Generate schoolbook factors as fallback
    U_school = [[0] * best_rank for _ in range(nm)]
    V_school = [[0] * best_rank for _ in range(mp)]
    W_school = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_factors = {'U': U_school, 'V': V_school, 'W': W_school}
    
    # Search for lower ranks
    for target_rank in [32, 33, 34, 35]:
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random restarts
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # ALS iterations
            for it in range(100):
                # Update U
                # T_u = T[:, :, u] flattened?
                # T is (nm, mp, pn).
                # We want to minimize || T - sum_r U[:,r] V[:,r] W[:,r] ||^2
                # This is a linear least squares problem for U given V, W.
                # Residual R = T - sum_{s != r} ... is hard.
                # Standard ALS:
                # For each r, solve for U[:,r].
                # But we can update all U at once if we fix V and W?
                # No, the model is sum_r U_r V_r W_r.
                # This is not linear in U if V,W are fixed?
                # It IS linear in U if we treat the columns of V and W as fixed.
                # Let M_u be the tensor reshaped.
                # T = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]
                # This is a sum of rank-1 tensors.
                # We can use the standard ALS update:
                # Fix V, W. Solve for U.
                # The term is sum_r U[:,r] (V[:,r] \otimes W[:,r])^T?
                # Let's flatten V and W into a single vector for each r.
                # v_w_r = V[:,r] * W[:,r] (element-wise? No, Kronecker product).
                # The tensor T can be seen as a matrix T_mat of size (nm, mp*pn).
                # T_mat[a, b+c] = T[a, b, c].
                # Then T_mat = U @ (V \otimes W)^T ?
                # (V \otimes W) has size (mp*pn, rank).
                # Let X = np.kron(V, W)? No, the ordering matters.
                # V is (mp, rank), W is (pn, rank).
                # We need a matrix M of size (mp*pn, rank) such that
                # M[:, r] corresponds to the vectorized V[:,r] \otimes W[:,r].
                # If we vectorize V[:,r] and W[:,r], the Kronecker product is V[:,r] \otimes W[:,r].
                # Let's construct M.
                M = np.zeros((mp * pn, target_rank))
                for r in range(target_rank):
                    M[:, r] = np.kron(V[:, r], W[:, r])
                    
                # Now T_mat = U @ M.T
                # Solve U = T_mat @ pinv(M.T)
                T_mat = T.reshape(nm, mp * pn)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M.T, T_mat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                # Fix U, W.
                # T_mat_v of size (mp, nm*pn)
                # T_mat_v[b, a+c] = T[a, b, c]
                # T_mat_v = V @ (U \otimes W)^T ?
                # M_v = np.zeros((nm * pn, target_rank))
                # for r: M_v[:, r] = np.kron(U[:, r], W[:, r])
                # T_mat_v = V @ M_v.T
                
                M_v = np.zeros((nm * pn, target_rank))
                for r in range(target_rank):
                    M_v[:, r] = np.kron(U[:, r], W[:, r])
                T_mat_v = np.transpose(T, (1, 0, 2)).reshape(mp, nm * pn)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_v.T, T_mat_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                # Fix U, V.
                # T_mat_w of size (pn, nm*mp)
                # T_mat_w[c, a+b] = T[a, b, c]
                # T_mat_w = W @ (U \otimes V)^T ?
                M_w = np.zeros((nm * mp, target_rank))
                for r in range(target_rank):
                    M_w[:, r] = np.kron(U[:, r], V[:, r])
                T_mat_w = np.transpose(T, (2, 0, 1)).reshape(pn, nm * mp)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_w.T, T_mat_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Check error
                if it % 10 == 0:
                    # Reconstruct
                    T_rec = np.zeros_like(T)
                    for r in range(target_rank):
                        T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                    err = np.linalg.norm(T - T_rec)
                    if err < 1e-8:
                        # Try to snap to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        # Verify exactness
                        T_rec_snap = np.zeros_like(T)
                        for r in range(target_rank):
                            T_rec_snap += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                        
                        if np.allclose(T, T_rec_snap, atol=1e-9):
                            # Convert to lists of Fractions or ints
                            U_out = [[int(x) if x == int(x) else float(x) for x in row] for row in U_snap]
                            V_out = [[int(x) if x == int(x) else float(x) for x in row] for row in V_snap]
                            W_out = [[int(x) if x == int(x) else float(x) for x in row] for row in W_snap]
                            
                            best_rank = target_rank
                            best_factors = {'U': U_out, 'V': V_out, 'W': W_out}
                            found = True
                            break
            if found:
                break
                
    return best_factors
