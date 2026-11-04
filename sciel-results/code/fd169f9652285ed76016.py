import numpy as np
import random

def construct():
    n, m, p = 3, 4, 6
    
    # Target Tensor T
    # T[a, b, c] where a in [0, nm), b in [0, mp), c in [0, pn)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T[a, b, c] = 1 if indices match a valid (i,j,k) path, else 0.
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Function to check if U, V, W is a valid decomposition
    def check(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = U.shape[1]
        # Compute sum_r U[:,r] o V[:,r] o W[:,r]
        # We only need to check if it equals T.
        # Efficient check:
        # Reshape T to (nm, mp*pn)
        T_flat = T.reshape(n*m, m*p * p*n)
        # Compute approx
        # approx[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        # approx_flat[a, (b, c)] = sum_r U[a,r] (V[b,r] W[c,r])
        # Let M_r = V[:,r] o W[:,r] (length mp*pn)
        # approx_flat = sum_r U[:,r] o M_r
        # This is a matrix multiplication: U @ M
        # M is (R, mp*pn)
        M = np.zeros((R, m*p * p*n), dtype=float)
        for r in range(R):
            M[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        
        approx_flat = U @ M.T # (nm, mp*pn)
        
        # Check if approx_flat equals T_flat
        # Use integer tolerance
        if not np.allclose(approx_flat, T_flat, atol=1e-9):
            return False
        # Also check if entries are close to integers? 
        # The problem requires integer/half-integer entries.
        # If we project to integers, they are integers.
        return True

    # ALS Solver
    def solve_als(R, max_iters=100, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with small random integers
        U = np.random.randint(-1, 2, size=(n*m, R))
        V = np.random.randint(-1, 2, size=(m*p, R))
        W = np.random.randint(-1, 2, size=(p*n, R))
        
        for _ in range(max_iters):
            # Fix V, W, solve for U
            # T_flat = U @ M.T
            # M is computed from V, W
            M = np.zeros((R, m*p * p*n), dtype=float)
            for r in range(R):
                M[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Solve U @ M.T = T_flat
            # U = T_flat @ M @ (M.T @ M)^-1
            # Use lstsq
            U_new, _, _, _ = np.linalg.lstsq(M.T, T_flat, rcond=None)
            # Project to integers
            U = np.round(U_new)
            
            # Fix U, W, solve for V
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each a, T[a, :, :] = sum_r U[a,r] (V[:,r] o W[:,r])
            # This is coupled across a.
            # Better: Fix U, W.
            # T_flat[a, (b,c)] = sum_r U[a,r] V[b,r] W[c,r]
            # Let N_r = U[:,r] o W[:,r]? No.
            # We want to solve for V.
            # T[a, b, c] = sum_r V[b,r] (U[a,r] W[c,r])
            # Let K_r = U[:,r] o W[:,r] (length nm*pn)
            # T_flat2 = T.reshape(m*p, n*m * p*n) ? No, b is first index in V.
            # T is (nm, mp, pn).
            # We want to extract V.
            # T[a, b, c] = sum_r V[b,r] * (U[a,r] W[c,r])
            # Let's reshape T to (mp, nm*pn)
            T_V = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
            K = np.zeros((R, n*m * p*n), dtype=float)
            for r in range(R):
                K[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
            V_new, _, _, _ = np.linalg.lstsq(K.T, T_V, rcond=None)
            V = np.round(V_new)
            
            # Fix U, V, solve for W
            # T[a, b, c] = sum_r W[c,r] (U[a,r] V[b,r])
            # Reshape T to (pn, nm*mp)
            T_W = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
            L = np.zeros((R, n*m * m*p), dtype=float)
            for r in range(R):
                L[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
            W_new, _, _, _ = np.linalg.lstsq(L.T, T_W, rcond=None)
            W = np.round(W_new)
            
            # Check convergence
            if check(U, V, W):
                return U, V, W
                
        return None

    # Try to find rank < 72
    schoolbook_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try a few ranks below schoolbook
    # Given time constraints, try 71 and 70
    for R in [schoolbook_rank - 1, schoolbook_rank - 2]:
        for seed in range(5):
            result = solve_als(R, max_iters=50, seed=seed)
            if result is not None:
                best_U, best_V, best_W = result
                break
        if best_U is not None:
            break
            
    if best_U is None:
        # Fallback to schoolbook
        R = schoolbook_rank
        U = np.zeros((n*m, R), dtype=float)
        V = np.zeros((m*p, R), dtype=float)
        W = np.zeros((p*n, R), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
