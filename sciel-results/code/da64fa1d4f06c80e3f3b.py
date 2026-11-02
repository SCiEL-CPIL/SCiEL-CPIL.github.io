import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook decomposition (Rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find a rank 23 decomposition for M_{3,3,3}
    # If found, we can construct a rank 44 decomposition for M_{3,4,4}
    # M_{3,4,4} splits into M_{3,3,3} (rank 23), M_{3,1,3} (rank 9), M_{3,4,1} (rank 12)
    # Total rank 44.
    
    def find_rank23_m333():
        # Tensor M_{3,3,3}
        n3, m3, p3 = 3, 3, 3
        T3 = np.zeros((n3 * m3, m3 * p3, p3 * n3))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    T3[i * m3 + j, j * p3 + k, k * n3 + i] = 1.0
        
        R = 23
        # Try a few random initializations
        for trial in range(20):
            if time.time() > start_time + 10: # 10s budget for this sub-search
                return None
            
            np.random.seed(42 + trial)
            U3 = np.random.randn(n3 * m3, R) * 0.1
            V3 = np.random.randn(m3 * p3, R) * 0.1
            W3 = np.random.randn(p3 * n3, R) * 0.1
            
            # ALS
            for it in range(50):
                # Optimize U3
                T_flat = T3.reshape(n3 * m3, -1)
                B = np.zeros((R, m3 * p3 * p3 * n3))
                for r in range(R):
                    B[r, :] = (np.outer(V3[:, r], W3[:, r])).reshape(-1)
                U3_new, _, _, _ = np.linalg.lstsq(B.T, T_flat, rcond=None)
                U3 = U3_new
                
                # Optimize V3
                T_flat_v = T3.reshape(m3 * p3, -1)
                B_v = np.zeros((R, n3 * m3 * p3 * n3))
                for r in range(R):
                    B_v[r, :] = (np.outer(U3[:, r], W3[:, r])).reshape(-1)
                V3_new, _, _, _ = np.linalg.lstsq(B_v.T, T_flat_v, rcond=None)
                V3 = V3_new
                
                # Optimize W3
                T_flat_w = T3.reshape(p3 * n3, -1)
                B_w = np.zeros((R, n3 * m3 * m3 * p3))
                for r in range(R):
                    B_w[r, :] = (np.outer(U3[:, r], V3[:, r])).reshape(-1)
                W3_new, _, _, _ = np.linalg.lstsq(B_w.T, T_flat_w, rcond=None)
                W3 = W3_new
                
            # Check error
            recon = np.zeros_like(T3)
            for r in range(R):
                recon += np.einsum('a,b,c->abc', U3[:, r], V3[:, r], W3[:, r])
            error = np.linalg.norm(T3 - recon)
            
            if error < 1e-6:
                # Round to integers/half-integers
                U3_r = np.round(U3 * 2) / 2
                V3_r = np.round(V3 * 2) / 2
                W3_r = np.round(W3 * 2) / 2
                recon_r = np.zeros_like(T3)
                for r in range(R):
                    recon_r += np.einsum('a,b,c->abc', U3_r[:, r], V3_r[:, r], W3_r[:, r])
                if np.linalg.norm(T3 - recon_r) < 1e-6:
                    return U3_r, V3_r, W3_r
        return None

    m333_factors = find_rank23_m333()
    
    if m333_factors is not None:
        U3, V3, W3 = m333_factors
        # Construct M_{3,4,4} decomposition of rank 44
        # Parts:
        # 1. M_{3,3,3} (U3, V3, W3) -> Rank 23
        #    Maps A_{3x3} (cols 0-2) and B_{3x3} (rows 0-2, cols 0-2) to C_{3x3} (rows 0-2, cols 0-2)
        # 2. M_{3,1,3} (Schoolbook) -> Rank 9
        #    Maps A_{3x1} (col 3) and B_{1x3} (row 3, cols 0-2) to C_{3x3} (rows 0-2, cols 0-2)
        # 3. M_{3,4,1} (Schoolbook) -> Rank 12
        #    Maps A_{3x4} (all cols) and B_{4x1} (col 3) to C_{3x1} (rows 0-2, col 3)
        
        R_total = 23 + 9 + 12
        U_total = np.zeros((n * m, R_total))
        V_total = np.zeros((m * p, R_total))
        W_total = np.zeros((p * n, R_total))
        
        # Part 1: M_{3,3,3}
        # U3 shape (9, 23). U_total indices for A_{ij} where i<3, j<3.
        # A is 3x4. j=0,1,2.
        for r in range(23):
            for idx in range(9):
                i = idx // 3
                j = idx % 3
                U_total[i * 4 + j, r] = U3[idx, r]
            
            for idx in range(9):
                j = idx // 3
                k = idx % 3
                V_total[j * 4 + k, r] = V3[idx, r]
            
            for idx in range(9):
                k = idx // 3
                i = idx % 3
                W_total[k * 3 + i, r] = W3[idx, r]
                
        # Part 2: M_{3,1,3}
        # A_{3x1} (col 3 of A, so j=3). B_{1x3} (row 3 of B, so j=3). C_{3x3} (k=0,1,2).
        # Schoolbook for 3x1x3: 3*1*3 = 9 terms.
        start_idx = 23
        col = start_idx
        for i in range(3):
            for k in range(3):
                # Term for A_{i3} B_{3k} -> C_{ik}
                U_total[i * 4 + 3, col] = 1
                V_total[3 * 4 + k, col] = 1
                W_total[k * 3 + i, col] = 1
                col += 1
                
        # Part 3: M_{3,4,1}
        # A_{3x4} (all j). B_{4x1} (col 3 of B, so k=3). C_{3x1} (i=0,1,2).
        # Schoolbook for 3x4x1: 3*4*1 = 12 terms.
        for i in range(3):
            for j in range(4):
                # Term for A_{ij} B_{j3} -> C_{i3}
                U_total[i * 4 + j, col] = 1
                V_total[j * 4 + 3, col] = 1
                W_total[3 * 3 + i, col] = 1
                col += 1
                
        # Verify
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
                    
        recon = np.zeros_like(T)
        for r in range(R_total):
            recon += np.einsum('a,b,c->abc', U_total[:, r], V_total[:, r], W_total[:, r])
            
        if np.linalg.norm(T - recon) < 1e-6:
            best_U, best_V, best_W = U_total, V_total, W_total
            best_rank = R_total

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
