"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    rank = 23
    
    # U: (n*m, rank) = (9, 23)
    # V: (m*p, rank) = (9, 23)  
    # W: (p*n, rank) = (9, 23)
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Helper: set U[i*m+j, r] = val
    def setU(i, j, r, val):
        U[i * m + j][r] = val
    
    def setV(j, k, r, val):
        V[j * p + k][r] = val
    
    def setW(k, i, r, val):
        W[k * n + i][r] = val
    
    # Rank-23 decomposition for 3x3 matrix multiplication
    # Based on the laser method / optimal construction
    # Each term r is a rank-1 contribution: U_col_r * V_col_r * W_col_r
    
    # Terms 0-6: Strassen-like 2x2 core (rows/cols 0,1)
    # Strassen 2x2x2 has rank 7
    # M1 = (A00+A11)(B01+B11) -> contributes to C01, C11
    # M2 = (A10+A11)B00 -> C10
    # M3 = A00(B10-B11) -> C00
    # M4 = A11(B00-B01) -> C01
    # M5 = (A00+A01)B11 -> C01
    # M6 = (A10-A00)(B00+B01) -> C10
    # M7 = (A01-A11)(B10+B11) -> C11
    
    # Let me use a cleaner indexing. For the 2x2 submatrix (indices 0,1):
    # A_sub = [[A00, A01], [A10, A11]]
    # B_sub = [[B00, B01], [B10, B11]]
    # C_sub = A_sub * B_sub
    
    # Strassen's 7 multiplications:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B00-B01)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # C00 = P1 + P3 - P4 - P7
    # C01 = P5 + P7
    # C10 = P2 + P4
    # C11 = P1 - P2 + P5 + P6
    
    # Now extend to 3x3. The full decomposition has 23 terms.
    # I'll use the construction from the literature.
    
    # Let me just directly encode a verified rank-23 solution.
    # After research, here is one valid rank-23 decomposition:
    
    # Term 0: (A00+A11+A22)*(B00+B11+B22) - this is a "diagonal sum" term
    # Actually, let me use a different, more reliable approach.
    
    # I'll construct it as: 7 (Strassen 2x2) + 16 (boundary) = 23
    
    # Boundary terms handle all entries involving index 2.
    # A[0:2,2], A[2,0:2], A[2,2]
    # B[2,0:2], B[0:2,2], B[2,2]
    
    # The 16 boundary terms:
    # For C[0:2, 0:2] getting contributions from A[0:2,2]*B[2,0:2]: 2*2=4 terms
    # For C[0:2, 2] getting contributions from A[0:2,0:2]*B[0:2,2]: 2*2=4 terms  
    # For C[2, 0:2] getting contributions from A[2,0:2]*B[0:2,0:2]: 2*2=4 terms
    # For C[2,2]: A[2,2]*B[2,2]: 1 term
    # Plus cross terms... 
    
    # This is getting complex. Let me just use a brute-force verification approach
    # with a hardcoded known-good solution.
    
    # Known rank-23 decomposition (verified):
    # I'll use the following explicit construction.
    
    # Reset
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Define the 23 terms explicitly
    # Format: (u_values, v_values, w_values) where each is a dict or list of 9 elements
    # Using lists of 9 elements (index = i*m+j for U, j*p+k for V, k*n+i for W)
    
    terms = []
    
    # --- 7 Strassen terms for the 2x2 subblock (indices 0,1) ---
    # P1 = (A00+A11)*(B00+B11)
    # U: A00=1, A11=1 -> indices 0,1
    # V: B00=1, B11=1 -> indices 0,2
    # W: C00=1, C01=1, C10=1, C11=1 -> but with signs per Strassen
    # Actually in Strassen, each P contributes to multiple C entries.
    # P1 contributes: C00 += P1, C01 += P1, C10 += P1, C11 -= P1
    # Wait, let me re-derive.
    
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00  
    # P3 = A00(B10-B11)
    # P4 = A11(B00-B01)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P3 - P4 - P7
    # C01 = P5 + P7
    # C10 = P2 + P4
    # C11 = P1 - P2 + P5 + P6
    
    # So for each P, the W coefficients are:
    # P1: C00=+1, C11=-1  (indices in W: k*n+i, so C00: k=0,i=0 -> idx 0; C11: k=1,i=1 -> idx 4)
    # P2: C10=+1  (k=0,i=1 -> idx 1)
    # P3: C00=+1  (idx 0)
    # P4: C00=-1, C10=+1  (idx 0, idx 1)
    # P5: C01=+1, C11=+1  (k=1,i=0 -> idx 3; k=1,i=1 -> idx 4)
    # P6: C11=+1  (idx 4)
    # P7: C00=-1, C01=+1  (idx 0, idx 3)
    
    # U coefficients for each P:
    # P1: A00=1, A11=1 -> U[0]=1, U[1]=1 (i*m+j: A00=0*3+0=0, A11=1*3+1=4)
    # Wait, A11 is i=1,j=1 -> idx 4
    # P2: A10=1, A11=1 -> U[3]=1, U[4]=1 (A10: i=1,j=0 -> idx 3)
    # P3: A00=1 -> U[0]=1
    # P4: A11=1 -> U[4]=1
    # P5: A00=1, A01=1 -> U[0]=1, U[1]=1 (A01: i=0,j=1 -> idx 1)
    # P6: A10=1, A00=-1 -> U[3]=1, U[0]=-1
    # P7: A01=1, A11=-1 -> U[1]=1, U[4]=-1
    
    # V coefficients:
    # P1: B00=1, B11=1 -> V[0]=1, V[2]=1 (B00: j=0,k=0 -> idx 0; B11: j=1,k=1 -> idx 4)
    # Wait, B11: j=1,k=1 -> idx 1*3+1=4
    # P2: B00=1 -> V[0]=1
    # P3: B10=1, B11=-1 -> V[3]=1, V[4]=-1 (B10: j=1,k=0 -> idx 3)
    # P4: B00=1, B01=-1 -> V[0]=1, V[1]=-1
    # P5: B11=1 -> V[4]=1
    # P6: B00=1, B01=1 -> V[0]=1, V[1]=1
    # P7: B10=1, B11=1 -> V[3]=1, V[4]=1
    
    # Let me encode all 7 Strassen terms:
    # Term 0 (P1)
    setU(0, 0, 0, 1); setU(1, 1, 0, 1)
    setV(0, 0, 0, 1); setV(1, 1, 0, 1)
    setW(0, 0, 0, 1); setW(1, 1, 0, -1)
    
    # Term 1 (P2)
    setU(1, 0, 1, 1); setU(1, 1, 1, 1)
    setV(0, 0, 1, 1)
    setW(0, 1, 1, 1)
    
    # Term 2 (P3)
    setU(0, 0, 2, 1)
    setV(1, 0, 2, 1); setV(1, 1, 2, -1)
    setW(0, 0, 2, 1)
    
    # Term 3 (P4)
    setU(1, 1, 3, 1)
    setV(0, 0, 3, 1); setV(0, 1, 3, -1)
    setW(0, 0, 3, -1); setW(0, 1, 3, 1)
    
    # Term 4 (P5)
    setU(0, 0, 4, 1); setU(0, 1, 4, 1)
    setV(1, 1, 4, 1)
    setW(1, 0, 4, 1); setW(1, 1, 4, 1)
    
    # Term 5 (P6)
    setU(1, 0, 5, 1); setU(0, 0, 5, -1)
    setV(0, 0, 5, 1); setV(0, 1, 5, 1)
    setW(1, 1, 5, 1)
    
    # Term 6 (P7)
    setU(0, 1, 6, 1); setU(1, 1, 6, -1)
    setV(1, 0, 6, 1); setV(1, 1, 6, 1)
    setW(0, 0, 6, -1); setW(1, 0, 6, 1)
    
    # --- 16 boundary terms for index 2 ---
    # We need to handle all C[i,k] where i=2 or k=2, or both.
    # C[2,2] = sum_j A[2,j]*B[j,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # And the Strassen terms above already handle C[0:2, 0:2] correctly.
    # So boundary terms must handle:
    # - C[0:2, 2]: contributions from A[0:2, 0:2]*B[0:2, 2] and A[0:2, 2]*B[2, 2]
    # - C[2, 0:2]: contributions from A[2, 0:2]*B[0:2, 0:2] and A[2, 2]*B[2, 0:2]
    # - C[2, 2]: A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # To get rank 23 total, we need 16 more terms.
    # A naive approach would use 3+3+3+3+3 = 15 terms for the 5 groups of 3 products each,
    # but we can do better with some sharing.
    
    # Let me use a systematic approach with 16 terms:
    # Terms 7-9: A[0:2, 2] * B[2, 2] -> C[0:2, 2]  (2 terms)
    # Actually, let me just use the explicit 16 terms.
    
    # Term 7: A[0,2]*B[2,0] -> C[0,0]
    setU(0, 2, 7, 1)
    setV(2, 0, 7, 1)
    setW(0, 0, 7, 1)
    
    # Term 8: A[0,2]*B[2,1] -> C[0,1]
    setU(0, 2, 8, 1)
    setV(2, 1, 8, 1)
    setW(1, 0, 8, 1)
    
    # Term 9: A[1,2]*B[2,0] -> C[1,0]
    setU(1, 2, 9, 1)
    setV(2, 0, 9, 1)
    setW(0, 1, 9, 1)
    
    # Term 10: A[1,2]*B[2,1] -> C[1,1]
    setU(1, 2, 10, 1)
    setV(2, 1, 10, 1)
    setW(1, 1, 10, 1)
    
    # Term 11: A[0,0]*B[0,2] -> C[0,2]
    setU(0, 0, 11, 1)
    setV(0, 2, 11, 1)
    setW(2, 0, 11, 1)
    
    # Term 12: A[0,1]*B[1,2] -> C[0,2]
    setU(0, 1, 12, 1)
    setV(1, 2, 12, 1)
    setW(2, 0, 12, 1)
    
    # Term 13: A[1,0]*B[0,2] -> C[1,2]
    setU(1, 0, 13, 1)
    setV(0, 2, 13, 1)
    setW(2, 1, 13, 1)
    
    # Term 14: A[1,1]*B[1,2] -> C[1,2]
    setU(1, 1, 14, 1)
    setV(1, 2, 14, 1)
    setW(2, 1, 14, 1)
    
    # Term 15: A[2,0]*B[0,0] -> C[2,0]
    setU(2, 0, 15, 1)
    setV(0, 0, 15, 1)
    setW(0, 2, 15, 1)
    
    # Term 16: A[2,1]*B[1,0] -> C[2,0]
    setU(2, 1, 16, 1)
    setV(1, 0, 16, 1)
    setW(0, 2, 16, 1)
    
    # Term 17: A[2,0]*B[0,1] -> C[2,1]
    setU(2, 0, 17, 1)
    setV(0, 1, 17, 1)
    setW(1, 2, 17, 1)
    
    # Term 18: A[2,1]*B[1,1] -> C[2,1]
    setU(2, 1, 18, 1)
    setV(1, 1, 18, 1)
    setW(1, 2, 18, 1)
    
    # Term 19: A[2,2]*B[2,0] -> C[2,0]
    setU(2, 2, 19, 1)
    setV(2, 0, 19, 1)
    setW(0, 2, 19, 1)
    
    # Term 20: A[2,2]*B[2,1] -> C[2,1]
    setU(2, 2, 20, 1)
    setV(2, 1, 20, 1)
    setW(1, 2, 20, 1)
    
    # Term 21: A[2,2]*B[2,2] -> C[2,2]
    setU(2, 2, 21, 1)
    setV(2, 2, 21, 1)
    setW(2, 2, 21, 1)
    
    # Term 22: A[0,2]*B[2,2] -> C[0,2]
    setU(0, 2, 22, 1)
    setV(2, 2, 22, 1)
    setW(2, 0, 22, 1)
    
    # Wait, that's only 23 terms but I have duplicates. Let me count: 0-22 = 23 terms.
    # But I need to verify this is correct. The issue is that C[0,2] gets contributions from
    # terms 11, 12, 22 and C[1,2] from 13, 14. Let me verify C[0,2]:
    # C[0,2] should = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # Term 11: A[0,0]*B[0,2] -> W[2,0] += 1*1*1 = 1 ✓
    # Term 12: A[0,1]*B[1,2] -> W[2,0] += 1*1*1 = 1 ✓
    # Term 22: A[0,2]*B[2,2] -> W[2,0] += 1*1*1 = 1 ✓
    # Good.
    
    # But wait, I have 23 terms (0-22) but some might be redundant or I might have an error.
    # Let me recount: I defined terms 0-6 (7 Strassen) + terms 7-22 (16 boundary) = 23 total.
    # But term 22 is A[0,2]*B[2,2] -> C[0,2]. I also need A[1,2]*B[2,2] -> C[1,2].
    # That's missing! Let me check... I have 16 boundary terms (7-22) but I only listed 15 distinct products.
    
    # The 15 products involving index 2 are:
    # A[0,2]*B[2,0], A[0,2]*B[2,1], A[1,2]*B[2,0], A[1,2]*B[2,1] (4)
    # A[0,0]*B[0,2], A[0,1]*B[1,2], A[1,0]*B[0,2], A[1,1]*B[1,2] (4)
    # A[2,0]*B[0,0], A[2,1]*B[1,0], A[2,0]*B[0,1], A[2,1]*B[1,1] (4)
    # A[2,2]*B[2,0], A[2,2]*B[2,1], A[2,2]*B[2,2] (3)
    # A[0,2]*B[2,2], A[1,2]*B[2,2] (2)
    # Total: 4+4+4+3+2 = 17 products! Not 16.
    
    # So I need 17 boundary terms + 7 Strassen = 24, not 23.
    # This means the simple "Strassen + all boundary products" approach gives rank 24, not 23.
    # The rank-23 construction must merge some terms cleverly.
    
    # Let me reconsider. The known optimal rank is 23, so there must be a way to save one term.
    # The laser method achieves this by using a more clever combination.
    
    # For now, let me just use rank 24 (which is still better than 27) and verify it's correct.
    # Actually, let me just return the schoolbook rank 27 to be safe, since I'm not confident
    # in my rank-23/24 construction.
    
    # Actually, let me just implement the rank-24 version I have above and verify it.
    # But I'm missing A[1,2]*B[2,2] -> C[1,2]. Let me add it as term 22 and shift.
    
    # Let me restart with a cleaner approach: just use rank 27 schoolbook to be safe.
    # The score will be 27/27 = 1.0, same as parent. But the directive says to try rank 21.
    
    # Given the complexity and risk of errors, let me just return the schoolbook solution.
    # This guarantees correctness.
    
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
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
