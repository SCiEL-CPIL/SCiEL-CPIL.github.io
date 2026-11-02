"""Matrix multiplication 4x4x6 tensor decomposition."""
import time
import math

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook as fallback
    rank_school = n * m * p  # 96
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify(U, V, W, rank, n, m, p):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Block Strassen - 12 independent 2x2x2 Strassen products
    # A: 4x4 -> 2x2 blocks indexed by (a,c) where a=row_block, c=col_block
    # B: 4x6 -> 2x2 blocks indexed by (c,b) where c=row_block, b=col_block
    # C: 4x6 -> 2x2 blocks indexed by (a,b) where a=row_block, b=col_block
    # C[a,b] = sum_c A[a,c] * B[c,b]
    # Total: 2*2*3 = 12 products, each rank 7 -> rank 84
    
    # Strassen 2x2x2 decomposition terms
    # Each term: (A_coefs, B_coefs, C_coefs)
    strassen_terms = [
        # P1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # P3 = A00(B10-B11)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4 = A11(B01-B00)
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    rank1 = 12 * 7  # 84
    U1 = [[0] * rank1 for _ in range(n * m)]  # 16 x 84
    V1 = [[0] * rank1 for _ in range(m * p)]  # 24 x 84
    W1 = [[0] * rank1 for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for a in range(2):       # row block of A/C
        for c in range(2):   # col block of A / row block of B
            for b in range(3):  # col block of B/C
                for (a_coefs, b_coefs, c_coefs) in strassen_terms:
                    # A-side: A_block(a,c) = A[2a:2a+2, 2c:2c+2]
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar  # global row in A
                        gj = 2*c + ac  # global col in A
                        U1[gi * m + gj][col] += coef
                    
                    # B-side: B_block(c,b) = B[2c:2c+2, 2b:2b+2]
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*c + br  # global row in B
                        gj = 2*b + bc  # global col in B
                        V1[gi * p + gj][col] += coef
                    
                    # C-side: C_block(a,b) = C[2a:2a+2, 2b:2b+2]
                    # W[k*n+i] where k is row of C (=col of B), i is col of C (=row of A)
                    for (cr, cc), coef in c_coefs.items():
                        gi = 2*b + cc  # k (row of C)
                        gj = 2*a + cr  # i (col of C)
                        W1[gi * n + gj][col] += coef
                    
                    col += 1
    
    assert col == rank1, f"col={col}, rank1={rank1}"
    
    if verify(U1, V1, W1, rank1, n, m, p):
        best_rank = rank1
        best_U = U1
        best_V = V1
        best_W = W1
    
    # DFT-based approach for 4x4x6
    # Use DFT to diagonalize the cyclic structure
    # For 4x4x4, this gives rank 16 (4 DFT modes * 4 pointwise products)
    # For 4x4x6, we can use a similar approach with rectangular DFT
    
    # The key insight: if we use DFT on the row/column indices,
    # the multiplication becomes pointwise in the frequency domain.
    # For an n x n x n problem, DFT gives rank n^2 (not n^3).
    # For 4x4x4: rank 16 (vs schoolbook 64)
    # For 4x4x6: we need to handle the rectangular case.
    
    # Let's try a DFT-based decomposition for 4x4x6.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want to write T as a sum of rank-1 tensors.
    
    # Using DFT: Let omega = exp(2*pi*i/4) = i
    # For the 4x4x4 case, we can use:
    # T[i,j,k] = (1/4) * sum_{a,b,c=0}^{3} omega^{-(a*i + b*j + c*k)} * 
    #            (sum_{i',j'} omega^{a*i' + b*j'}) * 
    #            (sum_{k'} omega^{c*k'}) * 
    #            (sum_{i''} omega^{a*i'' + c*i''})
    # This is getting complicated. Let me try a simpler approach.
    
    # Actually, for matrix multiplication A(4x4) * B(4x6) = C(4x6),
    # the tensor is T[i,j,k] where i is row of A/C, j is col of A/row of B, k is col of B/C.
    # T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,6).
    
    # DFT approach: Use DFT on the j-index (shared between A and B).
    # Let's define:
    # U[i,j] = sum_{j'} omega^{i*j'} * A[i,j']  (DFT of A along rows)
    # V[j,k] = sum_{j'} omega^{-j*j'} * B[j',k]  (DFT of B along cols)
    # Then C[i,k] = sum_j A[i,j] * B[j,k]
    # In frequency domain: C[i,k] = (1/4) * sum_{j'} U[i,j'] * V[j',k]
    
    # But this doesn't directly give a tensor decomposition.
    
    # Let me try a different approach: use the known rank-49 for 4x4x4
    # and extend it to 4x4x6.
    
    # Actually, let's try a simpler DFT-based approach.
    # For 4x4x4, the DFT gives a rank-16 decomposition.
    # For 4x4x6, we can use DFT on the 4x4 part and handle the 6 separately.
    
    # Split B into B1 (4x4) and B2 (4x2).
    # C = A * B1 + A * B2
    # For A * B1 (4x4x4): use DFT rank-16
    # For A * B2 (4x4x2): use schoolbook rank 32
    # Total: 16 + 32 = 48. But this is not valid because A is shared.
    
    # Let me think about this differently.
    # The tensor T[i,j,k] = 1 for i in [0,4), j in [0,4), k in [0,6).
    # We want to find rank-1 tensors that sum to T.
    
    # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i].
    # So we need: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k.
    
    # Using DFT: Let's try to construct such a decomposition.
    # For the 4x4x4 case, one known decomposition uses:
    # U_r[i,j] = omega^{r*i + r*j} for r = 0,1,2,3
    # V_r[j,k] = omega^{-r*j + r*k} for r = 0,1,2,3
    # W_r[k,i] = omega^{-r*k + r*i} for r = 0,1,2,3
    # But this gives 4 terms, not 16. Let me check.
    
    # sum_r omega^{r*i + r*j} * omega^{-r*j + r*k} * omega^{-r*k + r*i}
    # = sum_r omega^{2*r*i}
    # = sum_r (omega^{2*i})^r
    # = 4 if i = 0 or 2, 0 if i = 1 or 3.
    # So this doesn't work.
    
    # Let me try a different approach.
    # For 4x4x4, the rank is known to be 49 (or close).
    # The DFT approach might not be the best.
    
    # Let's try to use the block Strassen approach but with a better block size.
    # Instead of 2x2 blocks, use 4x4 blocks (trivial) or 1x1 blocks (schoolbook).
    
    # Actually, let's try a 4x4x6 decomposition using a different strategy.
    # Split the problem into:
    # 1) 4x4x4 part: use a known rank-49 decomposition
    # 2) 4x4x2 part: use schoolbook rank 32
    # Total: 49 + 32 = 81. This is better than 84!
    
    # But implementing rank-49 for 4x4x4 is complex.
    # Let's try to implement a simpler rank-49 decomposition.
    
    # Actually, the known rank-49 decomposition for 4x4x4 is due to Strassen
    # and is quite complex. Let me try a different approach.
    
    # Let's try to use a 4x4x4 decomposition with rank 49 and see if I can
    # implement it. If not, I'll stick with rank 84.
    
    # For now, let's try to improve the block Strassen approach.
    # Instead of 2x2 blocks, use 4x4 blocks with a different algorithm.
    
    # Actually, let's try a 4x4x6 decomposition using a 4x4x4 rank-49
    # decomposition for the first 4 columns and a 4x4x2 schoolbook for the last 2.
    
    # But I don't have a rank-49 decomposition for 4x4x4.
    
    # Let me try a different approach: use a 4x4x6 decomposition based on
    # the fact that 4 = 2*2 and 6 = 2*3.
    
    # Split A into 4 2x2 blocks, B into 6 2x2 blocks, C into 6 2x2 blocks.
    # C[a,b] = sum_c A[a,c] * B[c,b] where a in [0,2), c in [0,2), b in [0,3).
    # This is the same as the block Strassen approach with rank 84.
    
    # Let me try to use a 4x4x6 decomposition based on a 4x4x4 rank-49
    # decomposition. I'll try to implement a simplified version.
    
    # Actually, let's try to use a 4x4x6 decomposition based on the
    # following idea:
    # 1) Use a 4x4x4 rank-49 decomposition for the first 4 columns.
    # 2) Use a 4x4x2 schoolbook decomposition for the last 2 columns.
    # Total: 49 + 32 = 81.
    
    # But I need a rank-49 decomposition for 4x4x4.
    
    # Let me try to implement a rank-49 decomposition for 4x4x4.
    # The known decomposition uses 49 multiplications.
    
    # For now, let's stick with rank 84 and try to verify it.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Let's try one more approach: use a 4x4x6 decomposition based on
    # the following idea:
    # 1) Use a 4x4x4 rank-49 decomposition for the first 4 columns.
    # 2) Use a 4x4x2 schoolbook decomposition for the last 2 columns.
    
    # I'll try to implement a rank-49 decomposition for 4x4x4.
    # The decomposition is based on the following:
    # A = [A11 A12; A21 A22] where each Ai is 2x2.
    # B = [B11 B12 B13; B21 B22 B23] where each Bj is 2x2.
    # C = [C11 C12 C13; C21 C22 C23] where each Cj is 2x2.
    
    # For the 4x4x4 case (B is 4x4):
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen for each 2x2x2:
    # C11: 7 multiplications
    # C12: 7 multiplications
    # C21: 7 multiplications
    # C22: 7 multiplications
    # Total: 28 multiplications. But this is not optimal.
    
    # The optimal rank-49 decomposition uses shared multiplications.
    # This is complex to implement.
    
    # For now, let's stick with rank 84.
    
    return {"U": best_U, "V": best_V, "W": best_W}
