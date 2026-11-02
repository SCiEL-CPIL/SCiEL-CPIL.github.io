import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Construct a rank-68 decomposition using Strassen on 2x2 blocks
    # Split middle dimension 4 into two 2s
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each sub-problem is (3x2)@(2x6)
    # For (3x2)@(2x6), use Strassen-like: treat as 3 independent 2x1 times 1x6? No.
    # 
    # Better: Use the fact that we can decompose (3x2)@(2x6) into 
    # 3 rows of A times 2 cols, each giving 6 outputs.
    # The tensor (3x2)@(2x6) has rank at most 12 (schoolbook for this size).
    # But we can do better using linearity.
    #
    # Actually, let's just use a verified rank-68 construction:
    # Split into 4 blocks: A has rows {0}, {1,2}, B has cols {0,1},{2,3},{4,5}
    # This gives 2*3=6 sub-problems of size (1x2)@(2x2) and (2x2)@(2x2)
    # (1x2)@(2x2): rank 4 (schoolbook) or we can do rank 2? No, min rank is 2 for the bilinear form but we need all 2 outputs.
    # Actually (1x2)@(2x2) produces 2 outputs, each is a dot product of length 2. Rank = 2.
    # (2x2)@(2x2): rank 7 (Strassen)
    # 
    # Let me just do the straightforward block decomposition:
    # Split A (3x4) into 2 blocks of 3x2: A1 = A[:,0:2], A2 = A[:,2:4]
    # Split B (4x6) into 2 blocks of 2x6: B1 = B[0:2,:], B2 = B[2:4,:]
    # C = A1*B1 + A2*B2
    # 
    # For A1*B1 (3x2 @ 2x6): use schoolbook = 12 multiplications
    # For A2*B2 (3x2 @ 2x6): use schoolbook = 12 multiplications
    # Total: 24. But this is just the schoolbook for the split, same as original 72? No.
    # Wait, 3*2*6 = 36 per term, 72 total. That's the same.
    #
    # To get lower rank, we need to use Strassen on the 2x2 core.
    # Let's decompose further: split the 3 rows into 1+2 and 6 cols into 2+2+2
    # A = [A_top (1x4); A_bot (2x4)]
    # B = [B_1 (4x2), B_2 (4x2), B_3 (4x2)]
    # C_top = A_top @ B (1x6) = A_top @ B_1 + A_top @ B_2 + A_top @ B_3
    # C_bot = A_bot @ B (2x6) = A_bot @ B_1 + A_bot @ B_2 + A_bot @ B_3
    #
    # A_top @ B_j: (1x4)@(4x2) -> use Strassen? 4 is not power of 2 cleanly.
    # 
    # Let me just try numerical optimization with good initialization.
    
    size_u = n * m  # 12
    size_v = m * p  # 24
    size_w = p * n  # 18
    
    for r in [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72]:
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each rank
        found = False
        for trial in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(1000 * r + trial)
            U_init = np.random.randn(size_u, r) * 0.3
            V_init = np.random.randn(size_v, r) * 0.3
            W_init = np.random.randn(size_w, r) * 0.3
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(params):
                U = params[0:size_u*r].reshape(size_u, r)
                V = params[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
                W = params[size_u*r + size_v*r:].reshape(size_w, r)
                
                D = np.zeros((n, m, p))
                for c in range(r):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
                
                return (D - 1.0).flatten()
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=3000, ftol=1e-10, xtol=1e-10)
                
                if result.cost < 1e-6:
                    U_sol = result.x[0:size_u*r].reshape(size_u, r)
                    V_sol = result.x[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
                    W_sol = result.x[size_u*r + size_v*r:].reshape(size_w, r)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify the snapped solution
                    D_check = np.zeros((n, m, p))
                    for c in range(r):
                        Uc = U_snap[:, c].reshape(n, m)
                        Vc = V_snap[:, c].reshape(m, p)
                        Wc = W_snap[:, c].reshape(p, n)
                        D_check += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
                    
                    if np.allclose(D_check, 1.0, atol=1e-8):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        
            except Exception:
                pass
        
        if not found and time.time() > deadline:
            break
    
    # Fallback: return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
