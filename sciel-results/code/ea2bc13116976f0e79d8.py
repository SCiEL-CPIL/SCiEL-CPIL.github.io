import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)
    
    deadline = time.time() + 135
    
    # Try structured decompositions
    # For 2x3 x 3x3, known rank is 16. 
    # For 2x3 x 3x9, try rank 36 (3 blocks of 12 each, sharing some structure)
    # Or try rank 45 (3 blocks of 15)
    
    # Strategy: Use a block structure where we decompose the 9 columns into 3 groups of 3
    # For each group, use a rank-12 or rank-16 decomposition for 2x3 x 3x3
    # But share some rank vectors across groups
    
    # Let's try a direct approach: search for rank r with structured V and W
    # Key insight: V and W should have structure related to the tensor
    
    for r in range(27, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        found = False
        # Try structured initializations
        # Pattern 1: V and W with periodic structure
        for trial in range(500):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + r * 1000 + trial)
            
            # Create V and W with some structure
            # V has shape (27, r), W has shape (18, r)
            # Try entries from {-1, 0, 1} with some zeros
            density = 0.5
            V = np.random.randint(-1, 2, size=(mp, r)) * (np.random.rand(mp, r) < density)
            W = np.random.randint(-1, 2, size=(pn, r)) * (np.random.rand(pn, r) < density)
            
            # Build M matrix
            M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            
            if np.linalg.matrix_rank(M) < r:
                continue
                
            # Solve for U
            U = np.zeros((nm, r))
            valid = True
            for a in range(nm):
                target = T[a, :, :].flatten()
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, target, rcond=None)
                    if residuals[0] > 1e-8:
                        valid = False
                        break
                    U[a, :] = sol
                except:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            if is_half_integer(U):
                U_snapped = np.round(2.0 * U) / 2.0
                if verify_solution(U_snapped, V, W, T):
                    best_rank = r
                    best_U, best_V, best_W = U_snapped, V, W
                    found = True
                    break
                    
        if found:
            break
            
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
