    def add_strassen_2x2x2(offset, i_off, j_off, j_off_b, k_off):
        # M1
        U[i_off * m + j_off][offset + 0] += 1
        U[i_off * m + j_off + 1][offset + 0] += 1
        V[j_off_b * p + k_off][offset + 0] += 1
        V[(j_off_b + 1) * p + k_off][offset + 0] += 1
        W[k_off * n + i_off][offset + 0] += 1
        W[(k_off + 1) * n + (i_off + 1)][offset + 0] += 1
        
        # M2
        U[i_off * m + j_off + 1][offset + 1] += 1
        U[(i_off + 1) * m + j_off + 1][offset + 1] += 1
        V[j_off_b * p + k_off][offset + 1] += 1
        W[k_off * n + (i_off + 1)][offset + 1] += 1
        
        # M3
        U[i_off * m + j_off][offset + 2] += 1
        V[(j_off_b + 1) * p + (k_off + 1)][offset + 2] += 1
        V[j_off_b * p + (k_off + 1)][offset + 2] += -1
        W[(k_off + 1) * n + i_off][offset + 2] += 1
        
        # M4
        U[(i_off + 1) * m + j_off + 1][offset + 3] += 1
        V[(j_off_b + 1) * p + k_off][offset + 3] += 1
        V[j_off_b * p + k_off][offset + 3] += -1
        W[k_off * n + (i_off + 1)][offset + 3] += 1
        
        # M5
        U[i_off * m + j_off][offset + 4] += 1
        U[i_off * m + j_off + 1][offset + 4] += 1
        V[(j_off_b + 1) * p + (k_off + 1)][offset + 4] += 1
        W[(k_off + 1) * n + (i_off + 1)][offset + 4] += 1
        
        # M6
        U[(i_off + 1) * m + j_off][offset + 5] += 1
        U[i_off * m + j_off][offset + 5] += -1
        V[j_off_b * p + k_off][offset + 5] += 1
        V[j_off_b * p + (k_off + 1)][offset + 5] += 1
        W[(k_off + 1) * n + (i_off + 1)][offset + 5] += 1
        
        # M7
        U[i_off * m + j_off + 1][offset + 6] += 1
        U[(i_off + 1) * m + j_off + 1][offset + 6] += -1
        V[(j_off_b + 1) * p + k_off][offset + 6] += 1
        V[(j_off_b + 1) * p + (k_off + 1)][offset + 6] += 1
        W[k_off * n + i_off][offset + 6] += 1
        
        # Combine W for P00, P01, P10, P11
        # P00 = M1 + M4 - M5 + M7
        # W index for P00: k_off * n + i_off
        # M1: +1 (already added)
        # M4: +1 (already added)
        # M5: -1 (Need to subtract M5's contribution to P00? No, M5 contributes to P11 and P01? No.)
        # Let's check Strassen formulas again.
        # P00 = M1 + M4 - M5 + M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M2 + M3 + M6
        
        # My previous W additions were:
        # M1: W[P00]+=1, W[P11]+=1
        # M2: W[P10]+=1
        # M3: W[P01]+=1
        # M4: W[P10]+=1
        # M5: W[P11]+=1
        # M6: W[P11]+=1
        # M7: W[P00]+=1
        
        # Current W sums:
        # W[P00] = 1 (M1) + 1 (M4) + 1 (M7) = 3. Should be 1+1-1+1 = 2.
        # I missed the -M5 for P00.
        # W[P01] = 1 (M3) + 1 (M5) = 2. Should be 1+1 = 2. Correct.
        # W[P10] = 1 (M2) + 1 (M4) = 2. Should be 1+1 = 2. Correct.
        # W[P11] = 1 (M1) + 1 (M5) + 1 (M6) = 3. Should be 1-1+1+1 = 2.
        # I missed the -M2 for P11.
        
        # Corrections:
        W[k_off * n + i_off][offset + 4] += -1 # -M5 for P00
        W[(k_off + 1) * n + (i_off + 1)][offset + 1] += -1 # -M2 for P11

    def add_schoolbook_2x2x1(offset, i_off, j_off, j_off_b, k_off):
        # M1 = A00 B00 -> P0
        U[i_off * m + j_off][offset + 0] += 1
        V[j_off_b * p + k_off][offset + 0] += 1
        W[k_off * n + i_off][offset + 0] += 1
        
        # M2 = A01 B10 -> P0
        U[i_off * m + j_off + 1][offset + 1] += 1
        V[(j_off_b + 1) * p + k_off][offset + 1] += 1
        W[k_off * n + i_off][offset + 1] += 1
        
        # M3 = A10 B00 -> P1
        U[(i_off + 1) * m + j_off][offset + 2] += 1
        V[j_off_b * p + k_off][offset + 2] += 1
        W[k_off * n + (i_off + 1)][offset + 2] += 1
        
        # M4 = A11 B10 -> P1
        U[(i_off + 1) * m + j_off + 1][offset + 3] += 1
        V[(j_off_b + 1) * p + k_off][offset + 3] += 1
        W[k_off * n + (i_off + 1)][offset + 3] += 1
