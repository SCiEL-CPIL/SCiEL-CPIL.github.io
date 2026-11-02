import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[I, J, K] = 1 if I=im+j, J=jp+k, K=kn+i for some i,j,k
    # Otherwise 0.
    # We want U, V, W such that sum_r U[I,r] V[J,r] W[K,r] == T[I,J,K]
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
                
    def try_decomposition(rank, max_iters=200, tol=1e-9):
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        best_err = float('inf')
        
        for _ in range(max_iters):
            # Update U
            # T_unfolded_1 has shape (nm, mp*pn)
            # T_unfolded_1[I, J*pn + K] = T[I, J, K]
            T1 = T.reshape(nm, mp * pn)
            # Factor matrix for V, W: shape (mp*pn, rank)
            # (V \odot W)[J*pn + K, r] = V[J, r] * W[K, r]
            VW = (V.reshape(mp, 1, rank) * W.reshape(1, pn, rank)).reshape(mp * pn, rank)
            
            # Solve U * VW.T = T1
            # U = T1 @ VW @ pinv(VW.T @ VW)
            # Use least squares
            try:
                U, residuals, rank_u, s = np.linalg.lstsq(VW, T1.T, rcond=None)
                U = U.T
            except:
                pass
            
            # Update V
            T2 = T.reshape(mp, nm * pn)
            # T2[J, I*pn + K] = T[I, J, K]
            # We need to permute T2 to align with (U \odot W)
            # (U \odot W)[I*pn + K, r] = U[I, r] * W[K, r]
            UW = (U.reshape(nm, 1, rank) * W.reshape(1, pn, rank)).reshape(nm * pn, rank)
            # T2 is currently indexed by J, then I, then K.
            # We need T2_perm such that T2_perm[J, I*pn+K] matches the contraction.
            # T[I, J, K]. T2[J, I*pn+K] is correct if we view T as (J, I, K).
            # T shape (nm, mp, pn). T2 = T.transpose(1, 0, 2).reshape(mp, nm*pn)
            T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V, residuals, rank_v, s = np.linalg.lstsq(UW, T2.T, rcond=None)
                V = V.T
            except:
                pass
            
            # Update W
            T3 = T.reshape(pn, nm * mp)
            # T3[K, I*mp + J] = T[I, J, K]
            # (U \odot V)[I*mp + J, r] = U[I, r] * V[J, r]
            UV = (U.reshape(nm, 1, rank) * V.reshape(1, mp, rank)).reshape(nm * mp, rank)
            T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W, residuals, rank_w, s = np.linalg.lstsq(UV, T3.T, rcond=None)
                W = W.T
            except:
                pass
            
            # Calculate error
            # Reconstruct
            # Rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            # err = np.linalg.norm(Rec - T)
            # Faster check:
            # err = np.linalg.norm(T1 - U @ VW.T)
            # But U, V, W changed.
            # Let's just check periodically or at the end
            if _ % 10 == 0:
                Rec = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.linalg.norm(Rec - T)
                if err < best_err:
                    best_err = err
                    U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
                if err < tol:
                    break
                    
        if best_err < tol:
            # Try to snap to half-integers
            U_snap = np.round(U_best * 2) / 2
            V_snap = np.round(V_best * 2) / 2
            W_snap = np.round(W_best * 2) / 2
            
            Rec = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            if np.allclose(Rec, T, atol=1e-9):
                return U_snap, V_snap, W_snap
        return None

    # Try ranks from 120 downwards
    # 120 is trivial, so start 119
    for r in range(119, 90, -1):
        start_time = time.time()
        res = try_decomposition(r)
        elapsed = time.time() - start_time
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        if elapsed > 100:
            break
            
    # Fallback to trivial
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
