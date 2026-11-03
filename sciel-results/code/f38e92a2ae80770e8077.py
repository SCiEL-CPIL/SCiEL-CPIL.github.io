"""Matrix multiplication 5x5x5 - tensor decomposition with minimal rank."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a hybrid approach:
    - Recursive Strassen for the 4x4 core (rank 49)
    - Schoolbook for the border involving index 4 (rank 61)
    - Total rank: 110
    
    The matrix multiplication tensor T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    For schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for term (i,j,k)
    """
    start = time.time()
    deadline = start + 110
    
    n, m, p = 5, 5, 5
    
    # Build recursive Strassen decomposition for 4x4 (rank 49)
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4, we apply Strassen at block level (2x2 blocks of 2x2)
    # This gives rank 7*7 = 49
    
    # Let me build this more carefully.
    # 4x4 matrix multiplication can be seen as 2x2 matrix multiplication
    # where each entry is a 2x2 matrix.
    
    # Actually, let me just use the recursive structure directly.
    # For 4x4, we partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bi is 2x2
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # Using Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each Mi is a 2x2 product, which requires rank 7 (Strassen)
    # So total rank = 7 * 7 = 49
    
    # Let me define the 2x2 Strassen terms for a single 2x2 multiplication:
    # Input: A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # Output: C = A*B
    
    # The 7 bilinear forms:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b10-b11)
    # m4 = a11*(b01-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # Now for 4x4, we have 2x2 blocks of 2x2 matrices.
    # Let me index the 4x4 matrices as:
    # A[i,j] for i,j in 0..3
    # Block A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # The 4x4 result C[i,k] for i,k in 0..3:
    # C[0:2, 0:2] = C00 = M1 + M4 - M5 + M7 (where each M is 2x2)
    # C[0:2, 2:4] = C01 = M3 + M5
    # C[2:4, 0:2] = C10 = M2 + M4
    # C[2:4, 2:4] = C11 = M1 - M2 + M3 + M6
    
    # Each Mx is computed via Strassen on 2x2 blocks, requiring 7 terms.
    # So total 49 terms for the 4x4 part.
    
    # Let me construct the U, V, W matrices.
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)  
    # W has shape (p*n, rank) = (25, rank)
    
    # For the 4x4 Strassen part, indices i,j,k are in 0..3
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Let me define the 2x2 Strassen as a function that returns the 7 terms'
    # (U, V, W) coefficients.
    
    # For a 2x2 multiplication with input indices (a,b,c,d) for A and (e,f,g,h) for B:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # The 7 bilinear forms have specific U,V,W coefficients.
    
    # Actually, let me think about this differently.
    # The tensor decomposition for 2x2x2 matmul has rank 7.
    # Each term c has U[i,j,c], V[j,k,c], W[k,i,c] values.
    
    # For 2x2:
    # Term 0: U[(0,0),0]=1, U[(1,1),0]=1, V[(0,0),0]=1, V[(0,1),0]=1, V[(1,0),0]=1, V[(1,1),0]=1, W[(0,0),0]=1, W[(1,0),0]=-1, W[(0,1),0]=1, W[(1,1),0]=1
    # Wait, let me be more careful.
    
    # m1 = (a00+a11)*(b00+b11)
    # U contribution: U[0,0,0]=1 (a00), U[1,1,0]=1 (a11)
    # V contribution: V[0,0,0]=1 (b00), V[0,1,0]=1 (b01), V[1,0,0]=1 (b10), V[1,1,0]=1 (b11)
    # Wait, that's not right. V[j,k] for b_jk.
    # b00 is V[0,0], b01 is V[0,1], b10 is V[1,0], b11 is V[1,1]
    # So V[(0,0),0]=1, V[(0,1),0]=1, V[(1,0),0]=1, V[(1,1),0]=1
    # W: c00 gets m1, c01 gets nothing from m1 directly... 
    # Actually c00 = m1 + m4 - m5 + m7, so W[0,0,0] += 1 (from m1)
    # c01 = m3 + m5, so no m1 in c01
    # c10 = m2 + m4, so no m1 in c10
    # c11 = m1 - m2 + m3 + m6, so W[1,1,0] += 1 (from m1)
    
    # This is getting complicated. Let me just build it programmatically.
    
    # Define the 2x2 Strassen decomposition:
    # Returns (U_2x2, V_2x2, W_2x2) each of shape (4, 7) where index is (i,j) -> i*2+j
    # U[i*2+j, c] for A[i,j]
    # V[j*2+k, c] for B[j,k]
    # W[k*2+i, c] for C[k,i] -> wait, W[k*n+i] with n=2, so W[k*2+i]
    
    def strassen_2x2():
        """Return U, V, W for 2x2 Strassen, shapes (4,7) each."""
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # m1 = (a00+a11)*(b00+b11)
        # U: a00=U[0], a11=U[3]
        U[0, 0] = 1  # a00
        U[3, 0] = 1  # a11
        # V: b00=V[0], b01=V[1], b10=V[2], b11=V[3]
        V[0, 0] = 1  # b00
        V[1, 0] = 1  # b01
        V[2, 0] = 1  # b10
        V[3, 0] = 1  # b11
        # W: c00=W[0], c10=W[1], c01=W[2], c11=W[3]
        # c00 = m1 + ... => W[0,0] += 1
        # c11 = m1 + ... => W[3,0] += 1
        W[0, 0] = 1   # c00
        W[3, 0] = 1   # c11
        
        # m2 = (a10+a11)*b00
        U[2, 1] = 1  # a10
        U[3, 1] = 1  # a11
        V[0, 1] = 1  # b00
        # c10 = m2 + ... => W[1,1] += 1
        # c11 = m1 - m2 + ... => W[3,1] -= 1
        W[1, 1] = 1   # c10
        W[3, 1] = -1  # c11
        
        # m3 = a00*(b10-b11)
        U[0, 2] = 1  # a00
        V[2, 2] = 1  # b10
        V[3, 2] = -1 # b11
        # c01 = m3 + ... => W[2,2] += 1
        # c11 = m1 - m2 + m3 + ... => W[3,2] += 1
        W[2, 2] = 1   # c01
        W[3, 2] = 1   # c11
        
        # m4 = a11*(b01-b00)
        U[3, 3] = 1  # a11
        V[1, 3] = 1  # b01
        V[0, 3] = -1 # b00
        # c00 = m1 + m4 - ... => W[0,3] += 1
        # c10 = m2 + m4 => W[1,3] += 1
        W[0, 3] = 1   # c00
        W[1, 3] = 1   # c10
        
        # m5 = (a00+a01)*b11
        U[0, 4] = 1  # a00
        U[1, 4] = 1  # a01
        V[3, 4] = 1  # b11
        # c00 = m1 + m4 - m5 + ... => W[0,4] -= 1
        # c01 = m3 + m5 => W[2,4] += 1
        W[0, 4] = -1  # c00
        W[2, 4] = 1   # c01
        
        # m6 = (a10-a00)*(b00+b01)
        U[2, 5] = 1  # a10
        U[0, 5] = -1 # a00
        V[0, 5] = 1  # b00
        V[1, 5] = 1  # b01
        # c11 = m1 - m2 + m3 + m6 => W[3,5] += 1
        W[3, 5] = 1   # c11
        
        # m7 = (a01-a11)*(b10+b11)
        U[1, 6] = 1  # a01
        U[3, 6] = -1 # a11
        V[2, 6] = 1  # b10
        V[3, 6] = 1  # b11
        # c00 = m1 + m4 - m5 + m7 => W[0,6] += 1
        W[0, 6] = 1   # c00
        
        return U, V, W
    
    # Now build 4x4 Strassen using the 2x2 blocks.
    # 4x4 indices: i,j in 0..3
    # Blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # Similarly for B and C.
    
    # The 4x4 Strassen has 7 "top-level" terms M1..M7, each computed by 2x2 Strassen (7 sub-terms).
    # Total 49 terms.
    
    # Top-level Strassen for 4x4 (treating 2x2 blocks):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mx is a 2x2 matrix product, so Mx = A_block * B_block where A_block and B_block are 2x2.
    # The sub-Strassen for Mx gives 7 terms.
    
    # Let me define the block indices:
    # A00: i in 0,1; j in 0,1  -> A[i,j] for i,j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # For top-level term t (0..6) corresponding to M_{t+1}:
    # The A-block and B-block involved, and the coefficients.
    
    # Let me define:
    # top_A_coeffs[t] = list of (block, coeff) for A
    # top_B_coeffs[t] = list of (block, coeff) for B
    # block is 0=A00, 1=A01, 2=A10, 3=A11
    
    top_A = [
        [(0, 1), (3, 1)],  # M1: A00 + A11
        [(2, 1), (3, 1)],  # M2: A10 + A11
        [(0, 1)],          # M3: A00
        [(3, 1)],          # M4: A11
        [(0, 1), (1, 1)],  # M5: A00 + A01
        [(2, 1), (0, -1)], # M6: A10 - A00
        [(1, 1), (3, -1)], # M7: A01 - A11
    ]
    
    top_B = [
        [(0, 1), (3, 1)],  # M1: B00 + B11
        [(0, 1)],          # M2: B00
        [(2, 1), (3, -1)], # M3: B10 - B11
        [(1, 1), (0, -1)], # M4: B01 - B00
        [(3, 1)],          # M5: B11
        [(0, 1), (1, 1)],  # M6: B00 + B01
        [(2, 1), (3, 1)],  # M7: B10 + B11
    ]
    
    # C-block coefficients for each top-level term:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C-blocks: C00 (block 0), C01 (block 1), C10 (block 2), C11 (block 3)
    top_W = [
        [(0, 1), (3, 1)],  # M1: C00 + C11
        [(2, 1), (3, -1)], # M2: C10 - C11
        [(1, 1), (3, 1)],  # M3: C01 + C11
        [(0, 1), (2, 1)],  # M4: C00 + C10
        [(0, -1), (1, 1)], # M5: -C00 + C01
        [(3, 1)],          # M6: C11
        [(0, 1)],          # M7: C00
    ]
    
    # Now build the full 4x4 U, V, W.
    # U4 shape: (16, 49) where row index = i*4+j for i,j in 0..3
    # V4 shape: (16, 49) where row index = j*4+k for j,k in 0..3
    # W4 shape: (16, 49) where row index = k*4+i for k,i in 0..3
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    U2, V2, W2 = strassen_2x2()
    
    for t in range(7):  # top-level term
        for s in range(7):  # sub-term within each 2x2 Strassen
            col = t * 7 + s
            
            # For this column, U4 gets contributions from A-blocks
            for (ablock, acoeff) in top_A[t]:
                # A-block ablock: 0=A00 (i in 0,1, j in 0,1), 1=A01 (i in 0,1, j in 2,3),
                #                  2=A10 (i in 2,3, j in 0,1), 3=A11 (i in 2,3, j in 2,3)
                # Within the block, local indices (li, lj) in 0,1
                # Global i = block_row_offset + li, global j = block_col_offset + lj
                # block_row_offset = 2*(ablock // 2), block_col_offset = 2*(ablock % 2)
                broff = 2 * (ablock // 2)
                boff = 2 * (ablock % 2)
                for li in range(2):
                    for lj in range(2):
                        uidx2 = li * 2 + lj  # local U index
                        uidx4 = (broff + li) * 4 + (boff + lj)  # global U index
                        U4[uidx4, col] += acoeff * U2[uidx2, s]
            
            # Similarly for V
            for (bblock, bcoeff) in top_B[t]:
                broff = 2 * (bblock // 2)
                boff = 2 * (bblock % 2)
                for lj in range(2):  # local row (j within block)
                    for lk in range(2):  # local col (k within block)
                        vidx2 = lj * 2 + lk
                        vidx4 = (broff + lj) * 4 + (boff + lk)
                        V4[vidx4, col] += bcoeff * V2[vidx2, s]
            
            # Similarly for W
            for (cblock, ccoeff) in top_W[t]:
                broff = 2 * (cblock // 2)
                boff = 2 * (cblock % 2)
                for lk in range(2):  # local row (k within block)
                    for li in range(2):  # local col (i within block)
                        widx2 = lk * 2 + li
                        widx4 = (broff + lk) * 4 + (boff + li)
                        W4[widx4, col] += ccoeff * W2[widx2, s]
    
    # Verify 4x4 Strassen
    # Check: sum_c U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c] == 1 if j is "connected"
    # Actually for matmul tensor: T[i,j,k] = 1 for all i,j,k (each term in the sum)
    # The decomposition should satisfy:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,k} ... no wait.
    
    # Actually the matrix multiplication tensor is:
    # T[i,j,k] = 1 for all i,j,k (it's a 3-tensor where every entry is 1)
    # And C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
    
    # So we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # Let me verify:
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                total = 0.0
                for c in range(49):
                    total += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                if abs(total - 1.0) > 1e-10:
                    valid = False
                    print(f"4x4 verification failed at i={i}, j={j}, k={k}, total={total}")
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        print("4x4 Strassen verification failed!")
    
    # Now build the full 5x5x5 decomposition.
    # Total rank = 49 (Strassen 4x4) + 61 (border schoolbook)
    
    # Border terms involve index 4.
    # For the border, we use schoolbook for terms where at least one of i,j,k is 4.
    
    # Actually, let's think about which terms are covered by the 4x4 Strassen.
    # The 4x4 Strassen covers C[i,k] for i,k in 0..3 using A[i,j] and B[j,k] for j in 0..3.
    # It does NOT cover terms involving j=4.
    
    # C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k]
    # For i,k in 0..3:
    #   C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    # The first part is handled by 4x4 Strassen (rank 49).
    # The second part A[i,4]*B[4,k] needs rank 4*4 = 16 terms (one for each i,k pair).
    
    # For i=4 or k=4 (but not both):
    # If i=4, k in 0..3: C[4,k] = sum_{j=0}^{4} A[4,j]*B[j,k] -> 5 terms
    # If k=4, i in 0..3: C[i,4] = sum_{j=0}^{4} A[i,j]*B[j,4] -> 5 terms
    # If i=4, k=4: C[4,4] = sum_{j=0}^{4} A[4,j]*B[j,4] -> 5 terms
    
    # Wait, but the 4x4 Strassen only uses j in 0..3. So for i,k in 0..3, we're missing the j=4 term.
    # And for i=4 or k=4, we need all j from 0 to 4.
    
    # Let me count the border terms:
    # Case 1: i in 0..3, k in 0..3, j=4: 4*4 = 16 terms (A[i,4]*B[4,k])
    # Case 2: i=4, k in 0..3, j in 0..4: 4*5 = 20 terms
    # Case 3: i in 0..3, k=4, j in 0..4: 4*5 = 20 terms
    # Case 4: i=4, k=4, j in 0..4: 5 terms
    # Total: 16 + 20 + 20 + 5 = 61 terms
    
    # So total rank = 49 + 61 = 110.
    
    total_rank = 110
    U = np.zeros((25, total_rank), dtype=float)
    V = np.zeros((25, total_rank), dtype=float)
    W = np.zeros((25, total_rank), dtype=float)
    
    # Copy 4x4 Strassen into the first 49 columns
    for c in range(49):
        for i in range(4):
            for j in range(4):
                U[i*5+j, c] = U4[i*4+j, c]
        for j in range(4):
            for k in range(4):
                V[j*5+k, c] = V4[j*4+k, c]
        for k in range(4):
            for i in range(4):
                W[k*5+i, c] = W4[k*4+i, c]
    
    # Now add border schoolbook terms starting at column 49
    col = 49
    
    # Case 1: i in 0..3, k in 0..3, j=4
    for i in range(4):
        for k in range(4):
            U[i*5+4, col] = 1  # A[i,4]
            V[4*5+k, col] = 1  # B[4,k]
            W[k*5+i, col] = 1  # C[k,i] -> wait, W[k*n+i] = W[k*5+i]
            col += 1
    
    # Case 2: i=4, k in 0..3, j in 0..4
    for k in range(4):
        for j in range(5):
            U[4*5+j, col] = 1  # A[4,j]
            V[j*5+k, col] = 1  # B[j,k]
            W[k*5+4, col] = 1  # W[k*5+4]
            col += 1
    
    # Case 3: i in 0..3, k=4, j in 0..4
    for i in range(4):
        for j in range(5):
            U[i*5+j, col] = 1  # A[i,j]
            V[j*5+4, col] = 1  # B[j,4]
            W[4*5+i, col] = 1  # W[4*5+i]
            col += 1
    
    # Case 4: i=4, k=4, j in 0..4
    for j in range(5):
        U[4*5+j, col] = 1  # A[4,j]
        V[j*5+4, col] = 1  # B[j,4]
        W[4*5+4, col] = 1  # W[4*5+4]
        col += 1
    
    # Verify col == 110
    assert col == 110, f"Expected 110 columns, got {col}"
    
    # Final verification
    if time.time() < deadline:
        valid = True
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    total = 0.0
                    for c in range(total_rank):
                        total += U[i*5+j, c] * V[j*5+k, c] * W[k*5+i, c]
                    if abs(total - 1.0) > 1e-10:
                        valid = False
                        print(f"5x5 verification failed at i={i}, j={j}, k={k}, total={total}")
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            print("5x5 verification failed!")
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
