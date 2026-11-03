"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    deadline = time.time() + 120
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    def try_decomposition(U, V, W, rank, max_iter=300):
        """Alternating minimization with efficient solves."""
        for it in range(max_iter):
            if time.time() > deadline:
                return U, V, W, False
            
            # Solve for U: VW_flat.T @ U.T = T_U.T
            VW = np.einsum('br,cr->bcr', V, W)  # (mp, pn, rank)
            VW_flat = VW.transpose(2, 0, 1).reshape(rank, mp * pn)  # (rank, mp*pn)
            T_U = T.reshape(nm, -1)  # (nm, mp*pn)
            
            # U.T = lstsq(VW_flat.T, T_U.T)
            try:
                U_new_T, _, _, _ = np.linalg.lstsq(VW_flat.T, T_U.T, rcond=None)
                U_new = U_new_T.T  # (nm, rank)
            except:
                return U, V, W, False
            U = U_new
            
            # Solve for V: UW_flat.T @ V.T = T_V.T
            UW = np.einsum('ar,cr->acr', U, W)  # (nm, pn, rank)
            UW_flat = UW.transpose(2, 0, 1).reshape(rank, nm * pn)  # (rank, nm*pn)
            T_V = T.transpose(1, 0, 2).reshape(mp, -1)  # (mp, nm*pn)
            
            try:
                V_new_T, _, _, _ = np.linalg.lstsq(UW_flat.T, T_V.T, rcond=None)
                V_new = V_new_T.T  # (mp, rank)
            except:
                return U, V, W, False
            V = V_new
            
            # Solve for W: UV_flat.T @ W.T = T_W.T
            UV = np.einsum('ar,br->abr', U, V)  # (nm, mp, rank)
            UV_flat = UV.transpose(2, 0, 1).reshape(rank, nm * mp)  # (rank, nm*mp)
            T_W = T.transpose(2, 0, 1).reshape(pn, -1)  # (pn, nm*mp)
            
            try:
                W_new_T, _, _, _ = np.linalg.lstsq(UV_flat.T, T_W.T, rcond=None)
                W_new = W_new_T.T  # (pn, rank)
            except:
                return U, V, W, False
            W = W_new
            
            # Check convergence every 20 iterations
            if it % 20 == 0:
                R = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(R - T))
                if err < 1e-12:
                    return U, V, W, True
        
        # Final check
        R = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(R - T))
        return U, V, W, err < 1e-10
    
    def snap_and_verify(U, V, W, rank):
        """Try to snap to half-integers and verify."""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        R_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
        if np.max(np.abs(R_check - T)) < 1e-10:
            return U_rounded, V_rounded, W_rounded, True
        
        # Try integer snap
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        R_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        if np.max(np.abs(R_int - T)) < 1e-10:
            return U_int, V_int, W_int, True
        
        return None, None, None, False
    
    # Try ranks from low to high
    for rank in range(12, 37):
        if time.time() > deadline:
            break
        
        # Multiple initialization strategies
        inits = []
        
        # Strategy 1: Random
        for seed in range(30):
            np.random.seed(seed * 1000 + rank * 7)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            inits.append((U, V, W))
        
        # Strategy 2: Schoolbook-based perturbation (if rank >= nm*mp*pn/rank makes sense)
        if rank >= 18:
            # Start from schoolbook and randomly zero out some columns
            U_school = np.zeros((nm, nm*mp*pn))
            V_school = np.zeros((mp, nm*mp*pn))
            W_school = np.zeros((pn, nm*mp*pn))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i*m+j, col] = 1
                        V_school[j*p+k, col] = 1
                        W_school[k*n+i, col] = 1
                        col += 1
            
            # Select random subset of columns
            for seed in range(20):
                np.random.seed(seed * 2000 + rank * 3)
                cols = np.random.choice(nm*mp*pn, size=rank, replace=False)
                U = U_school[:, cols]
                V = V_school[:, cols]
                W = W_school[:, cols]
                inits.append((U, V, W))
        
        # Strategy 3: Structured initialization based on Kronecker products
        if rank >= 12:
            for seed in range(15):
                np.random.seed(seed * 3000 + rank * 11)
                # Create structured matrices
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
                # Scale to encourage sparse solutions
                U[np.abs(U) < 0.3] = 0
                V[np.abs(V) < 0.3] = 0
                W[np.abs(W) < 0.3] = 0
                inits.append((U, V, W))
        
        found = False
        for U0, V0, W0 in inits:
            if time.time() > deadline:
                break
            
            U, V, W, converged = try_decomposition(U0.copy(), V0.copy(), W0.copy(), rank)
            
            if converged:
                U_r, V_r, W_r, verified = snap_and_verify(U, V, W, rank)
                if verified:
                    best_rank = rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
        
        if found:
            break
    
    if best_rank is not None:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
