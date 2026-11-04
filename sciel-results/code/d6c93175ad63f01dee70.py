import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Flatten T for ALS
    T_flat = T.reshape(nm * mp, pn)
    # Actually, for ALS on tensor T = sum U_t V_t W_t, we view it as:
    # T[a, b, c] = sum_t U[a, t] V[b, t] W[c, t]
    # We can reshape T to (nm, mp*pn) for fixing U, etc.
    # Let's use the standard matricizations.
    # T_U: (nm, mp*pn), where row is a, col is (b, c)
    # T_U[a, b*pn + c] = T[a, b, c]
    T_U = T.reshape(nm, mp * pn)
    # T_V: (mp, nm*pn)
    T_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    # T_W: (pn, nm*mp)
    T_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor T."""
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        # Efficient reconstruction using einsum
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(T_recon - T))

    def schoolbook():
        """Schoolbook decomposition with rank n*m*p = 120."""
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def try_als_decomposition(rank, iterations=50, seed=0):
        """
        Attempt to find a rank-`rank` decomposition using Alternating Least Squares.
        Returns (U, V, W) if successful (low error), else None.
        """
        if time.time() > deadline - 5:
            return None
            
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # For better convergence, we can initialize using the schoolbook solution 
        # if rank >= 120, but we are searching for lower rank.
        # A better initialization: use the top singular vectors of the matricizations?
        # Or just random.
        
        for it in range(iterations):
            if time.time() > deadline - 2:
                return None
            
            # Fix V, W. Solve for U.
            # T[a, b, c] = U[a, t] * (V[b, t] W[c, t])
            # Let M_VW[b, c] = sum_t V[b, t] W[c, t]  (mp x pn)
            # Then T[a, b, c] = sum_t U[a, t] V[b, t] W[c, t]
            # This is not a simple linear system for U because V and W are multiplied element-wise in the sum.
            # We need to reshape.
            # T_U[a, b*pn+c] = sum_t U[a, t] * (V[b, t] W[c, t])
            # Let K_VW[t, b*pn+c] = V[b, t] W[c, t]
            # Then T_U = U @ K_VW.T ? No.
            # T_U[a, idx] = sum_t U[a, t] K_VW[t, idx]
            # So T_U = U @ K_VW.T is wrong dimension.
            # T_U is (nm, mp*pn). K_VW is (rank, mp*pn).
            # T_U = U @ K_VW.T -> (nm, rank) @ (rank, mp*pn) = (nm, mp*pn). Correct.
            
            K_VW = np.zeros((rank, mp * pn))
            for t in range(rank):
                # V[:, t] is (mp,), W[:, t] is (pn,)
                # Outer product V[:, t] W[:, t] is (mp, pn)
                # Flatten to (mp*pn,)
                K_VW[t, :] = np.outer(V[:, t], W[:, t]).flatten()
            
            # Solve U @ K_VW.T = T_U
            # U = T_U @ K_VW.T @ (K_VW @ K_VW.T)^-1
            # Or use lstsq: K_VW.T @ U.T = T_U.T
            try:
                U, _, _, _ = np.linalg.lstsq(K_VW.T, T_U.T, rcond=None)
                U = U.T
            except np.linalg.LinAlgError:
                return None

            # Fix U, W. Solve for V.
            # T_V[b, a*pn+c] = sum_t V[b, t] U[a, t] W[c, t]
            # K_UW[t, a*pn+c] = U[a, t] W[c, t]
            K_UW = np.zeros((rank, nm * pn))
            for t in range(rank):
                K_UW[t, :] = np.outer(U[:, t], W[:, t]).flatten()
            
            try:
                V, _, _, _ = np.linalg.lstsq(K_UW.T, T_V.T, rcond=None)
                V = V.T
            except np.linalg.LinAlgError:
                return None

            # Fix U, V. Solve for W.
            # T_W[c, a*mp+b] = sum_t W[c, t] U[a, t] V[b, t]
            # K_UV[t, a*mp+b] = U[a, t] V[b, t]
            K_UV = np.zeros((rank, nm * mp))
            for t in range(rank):
                K_UV[t, :] = np.outer(U[:, t], V[:, t]).flatten()
            
            try:
                W, _, _, _ = np.linalg.lstsq(K_UV.T, T_W.T, rcond=None)
                W = W.T
            except np.linalg.LinAlgError:
                return None

        # Calculate error
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        
        if err < 1e-6:
            # Try to round to integers or half-integers
            # The problem states "integer/half-integer entries"
            # We can try rounding to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            err_rounded = verify_decomposition(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-9:
                return U_rounded, V_rounded, W_rounded
            
            # If half-integer rounding fails, try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            err_int = verify_decomposition(U_int, V_int, W_int)
            if err_int < 1e-9:
                return U_int, V_int, W_int

            # If neither works, maybe the solution is exact but not on the 0.5 grid?
            # The problem requires integer/half-integer.
            # If numerical solution is close but not exactly on grid, we might need to adjust.
            # For now, return None if rounding doesn't yield exactness.
            pass
            
        return None

    # Strategy: Try ranks from 100 down to 90.
    # We know schoolbook is 120. Strassen-based is ~108.
    # Let's try 108, 106, 104...
    
    best_result = None
    best_rank = 120
    U_sb, V_sb, W_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    
    # List of ranks to try, prioritizing known good ones and lower ones
    ranks_to_try = [108, 106, 104, 102, 100, 98, 96, 94, 92, 90]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try multiple seeds for this rank
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            res = try_als_decomposition(rank, iterations=30, seed=seed)
            if res is not None:
                U, V, W = res
                # Verify one last time
                if verify_decomposition(U, V, W) < 1e-9:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
                    # If we found a valid one, we can try to go lower, 
                    # but given time constraints, we might stop early if we find a very good one.
                    # However, we want the MINIMAL rank.
                    # So we continue to lower ranks.
                    break # Break seed loop, try next lower rank
        
    # If no better rank was found via ALS, return schoolbook (or the best found)
    U, V, W = best_result
    
    # Final verification
    err = verify_decomposition(U, V, W)
    if err > 1e-9:
        # Fallback to schoolbook if something went wrong
        U, V, W = schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
