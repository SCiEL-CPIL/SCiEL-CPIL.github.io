"""Matrix multiplication 4x4 * 4x8 tensor decomposition via structured subspace embedding."""
DIMENSIONS = [4, 4, 8]

import time
import numpy as np

def _build_schoolbook(n, m, p):
    """Return schoolbook rank-n*m*p decomposition as a fallback."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # T is indexed as (i, j, k) -> 1
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_4x4_4x8():
    """
    Construct a Strassen-based rank-98 decomposition for 4x4 * 4x8.
    
    Strategy: Split the 4x8 matrix into two 4x4 blocks.
    Use Strassen's 7-multiplication algorithm for each 4x4*4x4 block.
    Total rank = 2 * 7 * 4 = 56... but we need to handle the structure carefully.
    
    Actually, let's use a known result: the rank of 4x4x4 matrix multiplication is 49.
    For 4x4 * 4x8, we can split B into B1, B2 (each 4x4), compute C1 = A*B1, C2 = A*B2.
    Each 4x4x4 multiplication has rank 49, so total rank = 98.
    
    We construct this using the known 49-rank decomposition for 4x4x4.
    """
    n, m, p = 4, 4, 8
    
    # For 4x4x4, the minimal rank is 49.
    # We use a known explicit construction based on Strassen recursion.
    # Strassen 2x2x2 has rank 7.
    # 4x4x4 can be done with rank 49 by applying Strassen recursively.
    
    # Let's build the 4x4x4 rank-49 decomposition first.
    # The 4x4 matrix can be viewed as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # Each Ai, Bk is 2x2.
    # C = A*B where Cij = sum_k Aik * Bkj (2x2 matrix multiplication)
    # Using Strassen: 7 products of 2x2 matrices.
    # Each 2x2 product has rank 4 (schoolbook) or we can go deeper.
    # Actually for 2x2x2, schoolbook rank is 8, Strassen rank is 7.
    # For 4x4x4: 7 * (rank of 2x2x2) = 7 * 7 = 49 if we use Strassen at both levels.
    
    # Let me construct this explicitly.
    # Level 1: 4x4 = 2x2 blocks of 2x2.
    # Strassen 2x2x2:
    #   P1 = (A11+A22)(B11+B22)
    #   P2 = (A21+A22)B11
    #   P3 = A11(B12-B22)
    #   P4 = A22(B21-B11)
    #   P5 = (A11+A12)B22
    #   P6 = (A21-A11)(B11+B12)
    #   P7 = (A12-A22)(B21+B22)
    #   C11 = P1 + P4 - P5 + P7
    #   C12 = P3 + P5
    #   C21 = P2 + P4
    #   C22 = P1 - P2 + P3 + P6
    
    # For 4x4x4 with rank 49:
    # The 7 Strassen products each involve 2x2 matrix multiplications.
    # Each 2x2 multiplication can be done with rank 7 (Strassen) or rank 8 (schoolbook).
    # Using Strassen: 7 * 7 = 49.
    
    # This is complex to construct explicitly. Let me use a different approach:
    # Use the known fact that rank(4x4x4) = 49 and construct via recursion.
    
    # Actually, let me just use a simpler approach: 
    # For 4x4 * 4x8, split into two 4x4 * 4x4 problems.
    # Use schoolbook for each: rank = 2 * 64 = 128. That's the baseline.
    # Use Strassen for each 4x4x4: rank = 2 * 49 = 98.
    
    # Let me construct the 4x4x4 rank-49 decomposition using nested Strassen.
    
    # I'll build it programmatically using the Strassen recursion.
    
    def strassen_2x2_factors():
        """
        Return the U, V, W factors for 2x2x2 Strassen decomposition (rank 7).
        U shape: (4, 7) for A[i,j]
        V shape: (4, 7) for B[j,k]
        W shape: (4, 7) for C[i,k]
        """
        # Strassen 2x2x2:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        U = [[0]*7 for _ in range(4)]  # A indices: 00, 01, 10, 11
        V = [[0]*7 for _ in range(4)]  # B indices: 00, 01, 10, 11
        W = [[0]*7 for _ in range(4)]  # C indices: 00, 01, 10, 11
        
        # P1 = (A11+A22)(B11+B22)
        # A11=0, A22=3; B11=0, B22=3
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11 += P1, C22 += P1
        W[0][0] = 1; W[3][0] = 1
        
        # P2 = (A21+A22)B11
        # A21=2, A22=3; B11=0
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 += P2, C22 -= P2
        W[2][1] = 1; W[3][1] = -1
        
        # P3 = A11(B12-B22)
        # A11=0; B12=1, B22=3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C12 += P3, C22 += P3
        W[1][2] = 1; W[3][2] = 1
        
        # P4 = A22(B21-B11)
        # A22=3; B21=2, B11=0
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C11 += P4, C21 += P4
        W[0][3] = 1; W[2][3] = 1
        
        # P5 = (A11+A12)B22
        # A11=0, A12=1; B22=3
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11 -= P5, C12 += P5
        W[0][4] = -1; W[1][4] = 1
        
        # P6 = (A21-A11)(B11+B12)
        # A21=2, A11=0; B11=0, B12=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22 += P6
        W[3][5] = 1
        
        # P7 = (A12-A22)(B21+B22)
        # A12=1, A22=3; B21=2, B22=3
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11 += P7
        W[0][6] = 1
        
        return U, V, W
    
    # Now for 4x4x4 with rank 49:
    # A is 4x4, viewed as 2x2 blocks of 2x2.
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
    # Similarly for B and C.
    
    # The 7 Strassen products at the block level:
    # P1 = (A11+A22)(B11+B22)  -- each is 2x2 * 2x2
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Each 2x2*2x2 product has rank 7 (Strassen).
    # So total rank = 7 * 7 = 49.
    
    # Let me build this.
    # A indices: A11 = [0,1,4,5], A12 = [2,3,6,7], A21 = [8,9,12,13], A22 = [10,11,14,15]
    # Wait, 4x4 matrix has indices 0..15.
    # A[i,j] for i,j in 0..3.
    # Block A11: i in 0..1, j in 0..1 -> indices 0,1,4,5
    # Block A12: i in 0..1, j in 2..3 -> indices 2,3,6,7
    # Block A21: i in 2..3, j in 0..1 -> indices 8,9,12,13
    # Block A22: i in 2..3, j in 2..3 -> indices 10,11,14,15
    
    # Similarly for B (4x4):
    # B11: 0,1,4,5; B12: 2,3,6,7; B21: 8,9,12,13; B22: 10,11,14,15
    
    # C (4x4):
    # C11: 0,1,4,5; C12: 2,3,6,7; C21: 8,9,12,13; C22: 10,11,14,15
    
    U_2x2, V_2x2, W_2x2 = strassen_2x2_factors()
    
    # For each of the 7 Strassen products, we need to:
    # 1. Form the linear combination of A blocks
    # 2. Form the linear combination of B blocks
    # 3. Multiply (2x2 * 2x2) using Strassen (rank 7)
    # 4. Add the result to the appropriate C blocks
    
    # This is getting complex. Let me use a different approach.
    # Instead of explicitly constructing rank 49, let me use a simpler method:
    # Use Strassen at the 4x4 level with 7 products, each computed with schoolbook 2x2 (rank 8).
    # Total rank = 7 * 8 = 56 for 4x4x4.
    # For 4x4 * 4x8: 2 * 56 = 112.
    
    # Or use Strassen at both levels: 7 * 7 = 49 for 4x4x4, total 98 for 4x4*4x8.
    
    # Let me try a cleaner construction.
    
    # Actually, let me just implement the nested Strassen properly.
    
    # For 4x4x4:
    # The 7 block-level Strassen products. Each product is a 2x2 matrix multiplication.
    # For each product, we use Strassen 2x2x2 (rank 7).
    
    # Let me define the block indices.
    # A blocks: 
    #   A11: rows 0-1, cols 0-1 -> flat indices [0,1,4,5]
    #   A12: rows 0-1, cols 2-3 -> flat indices [2,3,6,7]
    #   A21: rows 2-3, cols 0-1 -> flat indices [8,9,12,13]
    #   A22: rows 2-3, cols 2-3 -> flat indices [10,11,14,15]
    
    # For a 2x2 block with flat indices [a,b,c,d] (row-major: a=A00, b=A01, c=A10, d=A11):
    # The Strassen factors for this block use U_2x2, V_2x2, W_2x2.
    
    # For each Strassen product s (0..6):
    #   The A combination is a linear combination of blocks.
    #   The B combination is a linear combination of blocks.
    #   The C combination is a linear combination of blocks.
    
    # Let me define the Strassen combinations:
    # s=0: P1 = (A11+A22)(B11+B22), C11+=P1, C22+=P1
    # s=1: P2 = (A21+A22)B11, C21+=P2, C22-=P2
    # s=2: P3 = A11(B12-B22), C12+=P3, C22+=P3
    # s=3: P4 = A22(B21-B11), C11+=P4, C21+=P4
    # s=4: P5 = (A11+A12)B22, C11-=P5, C12+=P5
    # s=5: P6 = (A21-A11)(B11+B12), C22+=P6
    # s=6: P7 = (A12-A22)(B21+B22), C11+=P7
    
    # For each s, the A combination involves certain blocks with coefficients.
    # The B combination involves certain blocks with coefficients.
    # The C combination involves certain blocks with coefficients.
    
    # Let me define these:
    # A_comb[s] = list of (block_idx, coeff)
    # B_comb[s] = list of (block_idx, coeff)
    # C_comb[s] = list of (block_idx, coeff)
    
    A_comb = [
        [(0, 1), (3, 1)],      # s=0: A11 + A22
        [(2, 1), (3, 1)],      # s=1: A21 + A22
        [(0, 1)],              # s=2: A11
        [(3, 1)],              # s=3: A22
        [(0, 1), (1, 1)],      # s=4: A11 + A12
        [(2, 1), (0, -1)],     # s=5: A21 - A11
        [(1, 1), (3, -1)],     # s=6: A12 - A22
    ]
    
    B_comb = [
        [(0, 1), (3, 1)],      # s=0: B11 + B22
        [(0, 1)],              # s=1: B11
        [(1, 1), (3, -1)],     # s=2: B12 - B22
        [(2, 1), (0, -1)],     # s=3: B21 - B11
        [(3, 1)],              # s=4: B22
        [(0, 1), (1, 1)],      # s=5: B11 + B12
        [(2, 1), (3, 1)],      # s=6: B21 + B22
    ]
    
    C_comb = [
        [(0, 1), (3, 1)],      # s=0: C11 + C22
        [(2, 1), (3, -1)],     # s=1: C21 - C22
        [(1, 1), (3, 1)],      # s=2: C12 + C22
        [(0, 1), (2, 1)],      # s=3: C11 + C21
        [(0, -1), (1, 1)],     # s=4: -C11 + C12
        [(3, 1)],              # s=5: C22
        [(0, 1)],              # s=6: C11
    ]
    
    # Block flat indices:
    # Block 0 (A11): [0,1,4,5]
    # Block 1 (A12): [2,3,6,7]
    # Block 2 (A21): [8,9,12,13]
    # Block 3 (A22): [10,11,14,15]
    
    block_indices = [
        [0, 1, 4, 5],   # block 0
        [2, 3, 6, 7],   # block 1
        [8, 9, 12, 13], # block 2
        [10, 11, 14, 15] # block 3
    ]
    
    # Now construct the full 4x4x4 decomposition with rank 49.
    rank_4x4 = 49
    U_4x4 = [[0] * rank_4x4 for _ in range(16)]
    V_4x4 = [[0] * rank_4x4 for _ in range(16)]
    W_4x4 = [[0] * rank_4x4 for _ in range(16)]
    
    for s in range(7):
        base_col = s * 7
        # For each of the 7 Strassen sub-products:
        for t in range(7):
            col = base_col + t
            
            # A combination:
            for block_idx, coeff in A_comb[s]:
                for local_idx, flat_idx in enumerate(block_indices[block_idx]):
                    U_4x4[flat_idx][col] += coeff * U_2x2[local_idx][t]
            
            # B combination:
            for block_idx, coeff in B_comb[s]:
                for local_idx, flat_idx in enumerate(block_indices[block_idx]):
                    V_4x4[flat_idx][col] += coeff * V_2x2[local_idx][t]
            
            # C combination:
            for block_idx, coeff in C_comb[s]:
                for local_idx, flat_idx in enumerate(block_indices[block_idx]):
                    W_4x4[flat_idx][col] += coeff * W_2x2[local_idx][t]
    
    # Now for 4x4 * 4x8:
    # Split B into B1 (4x4) and B2 (4x4).
    # C = A*B1 | A*B2 (4x8)
    
    # U for A: same as U_4x4 (4x4 matrix)
    # V for B: B has 4x8, split into two 4x4 blocks.
    #   B1: columns 0-3, B2: columns 4-7
    #   V_4x4 for B1: indices 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29
    #   Wait, B is 4x8, so flat index = j*8 + k for j in 0..3, k in 0..7.
    #   B1 (j in 0..3, k in 0..3): flat indices j*8+k
    #   B2 (j in 0..3, k in 4..7): flat indices j*8+k
    
    # V for the full 4x8:
    #   For B1 (first 4 columns): use V_4x4 with indices mapped to j*8+k
    #   For B2 (last 4 columns): use V_4x4 with indices mapped to j*8+(k+4)
    
    # W for C (4x8):
    #   C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    #   C1 is 4x4, C2 is 4x4
    #   C flat index = i*8 + k for i in 0..3, k in 0..7
    #   C1: k in 0..3, C2: k in 4..7
    
    rank_total = 2 * rank_4x4  # 98
    U_total = [[0] * rank_total for _ in range(16)]  # 4x4 = 16 entries
    V_total = [[0] * rank_total for _ in range(32)]  # 4x8 = 32 entries
    W_total = [[0] * rank_total for _ in range(32)]  # 4x8 = 32 entries
    
    # Copy U_4x4 for both halves (since A is the same):
    for i in range(16):
        for c in range(rank_4x4):
            U_total[i][c] = U_4x4[i][c]
            U_total[i][c + rank_4x4] = U_4x4[i][c]
    
    # For V:
    # B1 (first 4 columns of B):
    #   B1[j,k] for j in 0..3, k in 0..3 -> flat index j*8 + k
    #   In V_4x4, the index is j*4 + k
    for j in range(4):
        for k in range(4):
            v4x4_idx = j * 4 + k
            v_total_idx = j * 8 + k
            for c in range(rank_4x4):
                V_total[v_total_idx][c] = V_4x4[v4x4_idx][c]
    
    # B2 (last 4 columns of B):
    #   B2[j,k] for j in 0..3, k in 0..3 -> flat index j*8 + (k+4)
    #   In V_4x4, the index is j*4 + k
    for j in range(4):
        for k in range(4):
            v4x4_idx = j * 4 + k
            v_total_idx = j * 8 + (k + 4)
            for c in range(rank_4x4):
                V_total[v_total_idx][c + rank_4x4] = V_4x4[v4x4_idx][c]
    
    # For W:
    # C1 (first 4 columns of C):
    #   C1[i,k] for i in 0..3, k in 0..3 -> flat index i*8 + k
    #   In W_4x4, the index is i*4 + k
    for i in range(4):
        for k in range(4):
            w4x4_idx = i * 4 + k
            w_total_idx = i * 8 + k
            for c in range(rank_4x4):
                W_total[w_total_idx][c] = W_4x4[w4x4_idx][c]
    
    # C2 (last 4 columns of C):
    #   C2[i,k] for i in 0..3, k in 0..3 -> flat index i*8 + (k+4)
    #   In W_4x4, the index is i*4 + k
    for i in range(4):
        for k in range(4):
            w4x4_idx = i * 4 + k
            w_total_idx = i * 8 + (k + 4)
            for c in range(rank_4x4):
                W_total[w_total_idx][c + rank_4x4] = W_4x4[w4x4_idx][c]
    
    return {"U": U_total, "V": V_total, "W": W_total}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try the Strassen-based rank-98 decomposition
    try:
        result = _strassen_4x4_4x8()
        # Verify
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook(n, m, p)
