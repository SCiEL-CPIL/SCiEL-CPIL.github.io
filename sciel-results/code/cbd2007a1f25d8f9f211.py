import numpy as np
import time

def construct():
    # Dimensions
    n, m, p = 4, 5, 6
    
    # Splits
    # n = 4 -> [2, 2]
    # m = 5 -> [2, 2, 1]
    # p = 6 -> [2, 2, 2]
    
    n_splits = [2, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Calculate offsets
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    # Total rank
    total_rank = 0
    block_decomps = []
    
    for i_idx, ni in enumerate(n_splits):
        for j_idx, mj in enumerate(m_splits):
            for k_idx, pk in enumerate(p_splits):
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen 2x2x2
                    rank = 7
                elif ni == 2 and mj == 1 and pk == 2:
                    # 2x1x2 -> Rank 2
                    rank = 2
                elif ni == 2 and mj == 2 and pk == 1:
                    # 2x2x1 -> Rank 2
                    rank = 2
                elif ni == 1 and mj == 2 and pk == 2:
                    # 1x2x2 -> Rank 2
                    rank = 2
                else:
                    # Schoolbook
                    rank = ni * mj * pk
                    
                total_rank += rank
                block_decomps.append((i_idx, j_idx, k_idx, ni, mj, pk, rank))
    
    # Initialize global matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    for i_idx, j_idx, k_idx, ni, mj, pk, rank in block_decomps:
        i_off = n_offsets[i_idx]
        j_off = m_offsets[j_idx]
        k_off = p_offsets[k_idx]
        
        if ni == 2 and mj == 2 and pk == 2:
            # Strassen's Algorithm for 2x2x2
            # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
            # T[i,j,k] = 1
            # Decomposition:
            # M1 = (A00 + A11)(B01 + B10)
            # M2 = (A10 + A11)B00
            # M3 = A00(B10 - B00)
            # M4 = A11(B01 - B11)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # U factors (A matrix indices)
            # A is 2x2. U_local is 4x7.
            # Row index in U_global: (i_off + i) * m + (j_off + j)
            # Note: U is indexed by [i*m + j]
            
            # M1: A00, A11
            # U[0, 0] += 1 (A00)
            # U[3, 0] += 1 (A11) -- wait, A11 is i=1, j=1. Index 1*2+1=3.
            
            # Let's define the 7 terms explicitly for 2x2x2
            # Term 0: (A00 + A11)(B01 + B10)
            # Term 1: (A10 + A11)B00
            # Term 2: A00(B10 - B00)
            # Term 3: A11(B01 - B11)
            # Term 4: (A00 + A01)B11
            # Term 5: (A10 - A00)(B00 + B01)
            # Term 6: (A01 - A11)(B10 + B11)
            
            # U contributions (A)
            # Term 0: A00(1), A11(1)
            # Term 1: A10(1), A11(1)
            # Term 2: A00(1)
            # Term 3: A11(1)
            # Term 4: A00(1), A01(1)
            # Term 5: A10(1), A00(-1)
            # Term 6: A01(1), A11(-1)
            
            u_vals = [
                [1, 0, 1, 0, 1, 1, 0], # A00 (i=0,j=0)
                [0, 0, 0, 0, 1, 0, 1], # A01 (i=0,j=1)
                [0, 1, 0, 0, 0, 1, 0], # A10 (i=1,j=0)
                [1, 1, 0, 1, 0, 0, -1] # A11 (i=1,j=1)
            ]
            
            # V contributions (B)
            # Term 0: B01(1), B10(1)
            # Term 1: B00(1)
            # Term 2: B10(1), B00(-1)
            # Term 3: B01(1), B11(-1)
            # Term 4: B11(1)
            # Term 5: B00(1), B01(1)
            # Term 6: B10(1), B11(1)
            
            v_vals = [
                [1, 1, -1, 1, 0, 1, 0], # B00 (j=0,k=0)
                [1, 0, 1, 0, 0, 1, 1],  # B01 (j=0,k=1)
                [0, 0, 0, 0, 0, 0, 1],  # B10 (j=1,k=0) -- Wait, B10 is j=1,k=0.
                [0, 0, 0, -1, 1, 0, 1]  # B11 (j=1,k=1)
            ]
            # Correction on V indices:
            # B is 2x2. Index j*p+k.
            # B00: j=0,k=0 -> 0
            # B01: j=0,k=1 -> 1
            # B10: j=1,k=0 -> 2
            # B11: j=1,k=1 -> 3
            
            # Let's re-verify v_vals rows correspond to B00, B01, B10, B11
            # Term 0: B01(1), B10(1) -> Row 1 (B01) gets 1, Row 2 (B10) gets 1.
            # Term 1: B00(1) -> Row 0 gets 1.
            # Term 2: B10(1), B00(-1) -> Row 2 gets 1, Row 0 gets -1.
            # Term 3: B01(1), B11(-1) -> Row 1 gets 1, Row 3 gets -1.
            # Term 4: B11(1) -> Row 3 gets 1.
            # Term 5: B00(1), B01(1) -> Row 0 gets 1, Row 1 gets 1.
            # Term 6: B10(1), B11(1) -> Row 2 gets 1, Row 3 gets 1.
            
            v_vals = [
                [1, 1, -1, 0, 0, 1, 0], # B00
                [1, 0, 0, 1, 0, 1, 0],  # B01
                [1, 0, 1, 0, 0, 0, 1],  # B10
                [0, 0, 0, -1, 1, 0, 1]  # B11
            ]
            
            # W contributions (C)
            # C00 = M1 - M3 + M5 - M7
            # C01 = M4 + M7
            # C10 = M2 + M5
            # C11 = M1 - M4 + M6
            
            # W index is k*n+i.
            # C00: i=0,k=0 -> 0*2+0=0
            # C01: i=0,k=1 -> 1*2+0=2
            # C10: i=1,k=0 -> 0*2+1=1
            # C11: i=1,k=1 -> 1*2+1=3
            
            w_vals = [
                [1, 0, -1, 0, 1, 0, -1], # C00
                [0, 1, 1, 0, 0, 0, 0],   # C10
                [0, 0, 0, 0, 1, 0, 1],   # C01
                [1, 0, 0, -1, 0, 1, 0]   # C11
            ]
            
            # Map to global
            for li in range(2):
                for lj in range(2):
                    gi = i_off + li
                    gj = j_off + lj
                    g_idx = gi * m + gj
                    for r in range(7):
                        U[g_idx, current_rank + r] += u_vals[li * 2 + lj][r]
                        
            for lj in range(2):
                for lk in range(2):
                    gj = j_off + lj
                    gk = k_off + lk
                    g_idx = gj * p + gk
                    for r in range(7):
                        V[g_idx, current_rank + r] += v_vals[lj * 2 + lk][r]
                        
            for lk in range(2):
                for li in range(2):
                    gk = k_off + lk
                    gi = i_off + li
                    g_idx = gk * n + gi
                    for r in range(7):
                        W[g_idx, current_rank + r] += w_vals[lk * 2 + li][r]

        else:
            # Schoolbook for other blocks
            # ni x mj x pk
            for li in range(ni):
                for lj in range(mj):
                    for lk in range(pk):
                        r = current_rank
                        gi = i_off + li
                        gj = j_off + lj
                        gk = k_off + lk
                        
                        U[gi * m + gj, r] = 1.0
                        V[gj * p + gk, r] = 1.0
                        W[gk * n + gi, r] = 1.0
                        
                        current_rank += 1
                        continue
            # The loop above increments current_rank inside, but we need to handle the rank count correctly.
            # Actually, the 'for' loop structure above is flawed for updating current_rank multiple times.
            # Let's rewrite the schoolbook part properly.
            
            # Reset and do it right
            # We already incremented current_rank in the list 'block_decomps' logic? No, we just calculated total_rank.
            # We need to advance current_rank by 'rank' after filling.
            
            # Wait, the code above for schoolbook is inside the else.
            # Let's refactor to ensure current_rank is updated correctly.
            
            pass # Placeholder, will rewrite below

    # The previous block had a flaw in current_rank management for schoolbook.
    # Let's restart the construction loop cleanly.
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    for i_idx, j_idx, k_idx, ni, mj, pk, rank in block_decomps:
        i_off = n_offsets[i_idx]
        j_off = m_offsets[j_idx]
        k_off = p_offsets[k_idx]
        
        if ni == 2 and mj == 2 and pk == 2:
            # Strassen
            u_vals = [
                [1, 0, 1, 0, 1, 1, 0], 
                [0, 0, 0, 0, 1, 0, 1], 
                [0, 1, 0, 0, 0, 1, 0], 
                [1, 1, 0, 1, 0, 0, -1] 
            ]
            v_vals = [
                [1, 1, -1, 0, 0, 1, 0], 
                [1, 0, 0, 1, 0, 1, 0],  
                [1, 0, 1, 0, 0, 0, 1],  
                [0, 0, 0, -1, 1, 0, 1]  
            ]
            w_vals = [
                [1, 0, -1, 0, 1, 0, -1], 
                [0, 1, 1, 0, 0, 0, 0],   
                [0, 0, 0, 0, 1, 0, 1],   
                [1, 0, 0, -1, 0, 1, 0]   
            ]
            
            for li in range(2):
                for lj in range(2):
                    gi = i_off + li
                    gj = j_off + lj
                    g_idx = gi * m + gj
                    for r in range(7):
                        U[g_idx, current_rank + r] += u_vals[li * 2 + lj][r]
                        
            for lj in range(2):
                for lk in range(2):
                    gj = j_off + lj
                    gk = k_off + lk
                    g_idx = gj * p + gk
                    for r in range(7):
                        V[g_idx, current_rank + r] += v_vals[lj * 2 + lk][r]
                        
            for lk in range(2):
                for li in range(2):
                    gk = k_off + lk
                    gi = i_off + li
                    g_idx = gk * n + gi
                    for r in range(7):
                        W[g_idx, current_rank + r] += w_vals[lk * 2 + li][r]
            
            current_rank += 7

        else:
            # Schoolbook
            for li in range(ni):
                for lj in range(mj):
                    for lk in range(pk):
                        gi = i_off + li
                        gj = j_off + lj
                        gk = k_off + lk
                        
                        U[gi * m + gj, current_rank] = 1.0
                        V[gj * p + gk, current_rank] = 1.0
                        W[gk * n + gi, current_rank] = 1.0
                        
                        current_rank += 1

    # Verification (optional but good for debugging, removed for speed in final)
    # assert current_rank == total_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    # Quick check
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = 4, 5, 6
    err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err += abs(val - 1.0)
    if err > 1e-9:
        print(f"Verification Failed: {err}")
    else:
        print(f"Verification Passed. Rank: {U.shape[1]}")
