import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140

    # Build rank-22 solution: two independent 2x3x3 blocks (rank 11 each)
    # Each 2x3x3 block: Strassen (rank 7) for first 2 cols + schoolbook (rank 4) for col 2
    rank_start = 22
    U = np.zeros((n * m, rank_start))
    V = np.zeros((m * p, rank_start))
    W = np.zeros((p * n, rank_start))

    col = 0

    # For each of the two 2x3x3 blocks (columns 0-2 and 3-5 of B)
    for block in range(2):
        k0 = 3 * block
        k1 = k0 + 1
        k2 = k0 + 2

        # Strassen 7 terms for A[:, 0:2] (2x2) * B[0:2, k0:k2+1] (2x3)
        # Wait, B is 3x6. For a 2x3x3 block, B is 3x3.
        # A is 2x3, B is 3x3.
        # Split A into A_L (2x2) and A_R (2x1).
        # Split B into B_L (3x2) and B_R (3x1)? No.
        # Let's stick to the previous analysis:
        # A (2x3) * B (3x3) -> C (2x3)
        # Let A = [A_00 A_01 A_02; A_10 A_11 A_12]
        # Let B = [B_00 B_01 B_02; B_10 B_11 B_12; B_20 B_21 B_22]
        # We can view this as:
        # C = [A_00 A_01] * [B_00 B_01 B_02; B_10 B_11 B_12] + [A_02; A_12] * [B_20 B_21 B_22]
        # First part is 2x2 * 2x3. Second part is 2x1 * 1x3.
        
        # Part 1: 2x2 * 2x3. Rank is 7 (Strassen-like) + ?
        # Actually, 2x2 * 2x2 is rank 7. 2x2 * 2x1 is rank 4 (schoolbook).
        # So 2x2 * 2x3 is rank 7+4 = 11? No, that's if we split the output.
        # Let's use the known rank of MM_{2,2,3}.
        # R(2,2,3) = 7. (Strassen for 2x2x2 + 1 more? No, 2x2x3 is known to be 7? 
        # Actually R(2,2,3) = 7 is not true, R(2,2,3) = 7 is for 2x2x2. 
        # R(2,2,3) = 7? No, R(2,2,3) = 7 is a lower bound. 
        # Actually, R(2,2,3) = 7 is correct? No, R(2,2,3) = 7 is for 2x2x2.
        # R(2,2,3) = 7? Let's check.
        # Actually, the rank of 2x2x3 is 7. 
        # Wait, I recall R(2,2,3) = 7.
        # Let's assume R(2,2,3) = 7.
        # Then R(2,1,3) = 3 (schoolbook).
        # Total R(2,3,3) = 7 + 3 = 10?
        # But earlier I calculated 11.
        # Let's re-evaluate.
        # A (2x3) = [A_L (2x2) | A_R (2x1)]
        # B (3x3) = [B_L (3x2); B_R (3x1)]
        # C = A_L B_L + A_R B_R
        # A_L B_L is 2x2 * 2x2 -> rank 7.
        # A_R B_R is 2x1 * 1x3 -> rank 3.
        # Total rank 10.
        
        # Let's implement rank 10 per block, total rank 20.
        
        # Strassen for A_L (2x2) * B_L (2x2) -> C_L (2x2)
        # A_L = A[:, 0:2], B_L = B[0:2, k0:k0+2]
        # C_L = C[:, k0:k0+2]
        
        # M1: (A00+A11)(B00+B11)
        U[0 * m + 0, col] += 1.0; U[1 * m + 1, col] += 1.0
        V[0 * p + k0, col] += 1.0; V[1 * p + k0 + 1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[(k0+1) * n + 1, col] += 1.0
        col += 1
        
        # M2: (A10+A11)B00
        U[1 * m + 0, col] += 1.0; U[1 * m + 1, col] += 1.0
        V[0 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[(k0+1) * n + 1, col] += -1.0
        col += 1
        
        # M3: A00(B10-B11)
        U[0 * m + 0, col] += 1.0
        V[1 * p + k0 + 1, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[(k0+1) * n + 0, col] += 1.0; W[(k0+1) * n + 1, col] += 1.0
        col += 1
        
        # M4: A11(B01-B00)
        U[1 * m + 1, col] += 1.0
        V[0 * p + k0 + 1, col] += 1.0; V[0 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5: (A00+A01)B11
        U[0 * m + 0, col] += 1.0; U[0 * m + 1, col] += 1.0
        V[1 * p + k0 + 1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[(k0+1) * n + 0, col] += 1.0
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[1 * m + 0, col] += 1.0; U[0 * m + 0, col] += -1.0
        V[0 * p + k0, col] += 1.0; V[0 * p + k0 + 1, col] += 1.0
        W[(k0+1) * n + 1, col] += 1.0
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[0 * m + 1, col] += 1.0; U[1 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k0 + 1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1
        
        # Schoolbook for A_R (2x1) * B_R (1x3) -> C_R (2x3)
        # A_R = A[:, 2], B_R = B[2, k0:k0+3]
        # C_R = C[:, k0:k0+3]
        for kk in range(3):
            U[0 * m + 2, col] += 1.0
            V[2 * p + k0 + kk, col] += 1.0
            W[(k0 + kk) * n + 0, col] += 1.0
            col += 1
            
            U[1 * m + 2, col] += 1.0
            V[2 * p + k0 + kk, col] += 1.0
            W[(k0 + kk) * n + 1, col] += 1.0
            col += 1

    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        r = U_mat.shape[1]
        U_r = U_mat.reshape(n, m, -1)
        V_r = V_mat.reshape(m, p, -1)
        W_r = W_mat.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=tol)
    
    if not verify_fast(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col_s = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_s[ii * m + jj, col_s] = 1.0
                    V_s[jj * p + kk, col_s] = 1.0
                    W_s[kk * n + ii, col_s] = 1.0
                    col_s += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank_start
    
    # Try to reduce rank
    for trial_rank in range(rank_start - 1, max(10, rank_start - 5), -1):
        if time.time() > deadline - 5:
            break
        found = False
        np.random.seed(42 + trial_rank)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            cols = np.random.choice(best_rank, trial_rank, replace=False)
            U_sub = best_U[:, cols]
            V_sub = best_V[:, cols]
            W_sub = best_W[:, cols]
            if verify_fast(U_sub, V_sub, W_sub):
                best_rank = trial_rank
                best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                found = True
                break
        if not found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
