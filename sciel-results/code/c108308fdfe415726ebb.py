"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Target: Rank 50 decomposition for 2x4 * 4x7 -> 2x7
    # Strategy: Constructive algebraic decomposition.
    # We view the multiplication C = A * B where A is 2x4, B is 4x7, C is 2x7.
    # This is equivalent to two independent 4x7 multiplications (one for each row of A).
    # Let A = [A0; A1] where A0, A1 are 1x4.
    # C0 = A0 * B (1x7)
    # C1 = A1 * B (1x7)
    
    # We use a specific rank-50 construction derived from splitting the 4x7 product
    # into overlapping 2x2 Strassen blocks and handling boundaries with rank-1 terms.
    
    # The 4x7 product can be seen as:
    # C = [A0 A1 A2 A3] * B
    # where A0..A3 are 1x1 (scalars from A row) and B is 4x7.
    # Actually, let's look at the indices.
    # A is 2x4. Rows i=0,1. Cols j=0..3.
    # B is 4x7. Rows j=0..3. Cols k=0..6.
    # C is 2x7. Rows i=0,1. Cols k=0..6.
    
    # Decomposition of 4x7 multiplication (for one row of A) into rank 25?
    # If we can do 4x7 in rank 25, then 2x4x7 is rank 50.
    # Known lower bound for 4x7: max(4*7, ...) ? 
    # Strassen for 2x2 is rank 7.
    # 4x7 can be viewed as 2x(2x7).
    # 2x7 product: A(2x2) * B(2x7).
    # If we use Strassen for the 2x2 part, we need to handle the 7 columns.
    
    # Let's use a specific known decomposition for 4x7 of rank 25?
    # Or simply construct a valid rank 50 decomposition explicitly.
    
    # A robust way to get rank 50:
    # Use Strassen for the 2x2 blocks of the 4x4 part? No, B is 4x7.
    
    # Let's try a direct construction based on "Strassen-like" splits.
    # Split A (2x4) into columns: A = [A_L (2x2) | A_R (2x2)]
    # Split B (4x7) into rows: B = [B_T (2x7); B_B (2x7)]
    # C = A_L B_T + A_R B_B
    
    # Each term is (2x2) * (2x7).
    # We can decompose (2x2) * (2x7) using Strassen on the 2x2 part.
    # Let M = 2x2, N = 2x7.
    # Strassen for M * N?
    # Standard Strassen is for square matrices.
    # For rectangular, we can pad N to 2x8? Or use specific linear combos.
    
    # Let's use the "rank 7" property for 2x2x2.
    # For 2x2x7, we can treat it as 3.5 blocks of 2x2x2? No.
    
    # Alternative: Use the known result that 4x7 can be done in rank 25?
    # Actually, the best known bound for 4x7 is 25?
    # If 4x7 rank is 25, then 2x4x7 rank is 50.
    
    # Let's construct the rank 25 decomposition for 4x7.
    # A (1x4) * B (4x7) -> C (1x7).
    # This is equivalent to 4x7x1 tensor.
    
    # We will use a constructive approach with integer/half-integers.
    # We define 25 rank-1 terms.
    # U: 4x25, V: 7x25, W: 1x25 (for one row of A).
    # For 2 rows of A, we duplicate this structure, so rank 50.
    # U_total: 8x50, V_total: 28x50, W_total: 14x50.
    
    # How to get rank 25 for 4x7?
    # Split 4x7 into 2x7 + 2x7?
    # 2x7 product rank?
    # 2x7 = 2x(2x2 + 2x2 + 1x2)?
    # 
    # Let's use a simpler, verified construction:
    # Use Strassen for 2x2x2 blocks where possible.
    # 4x7 can be decomposed as:
    # [A0 A1] [B0; B1] (2x2 * 2x7) + [A2 A3] [B2; B3] (2x2 * 2x7)
    # Wait, A is 1x4, so A0,A1,A2,A3 are scalars.
    # B is 4x7. B0..B3 are 1x7 rows.
    # C = A0*B0 + A1*B1 + A2*B2 + A3*B3.
    # This is just rank 4 if we treat B0..B3 as vectors? No, A0..A3 are scalars from A.
    # The tensor is T_{i,j,k} = delta_{i,0} ... no.
    # The multiplication tensor for 1x4 * 4x7 has rank 4?
    # No, the inputs are vectors.
    # C_k = sum_j A_j B_{j,k}.
    # This is a bilinear map R^4 x R^{28} -> R^7.
    # The rank is the minimum number of rank-1 tensors.
    # A rank-1 tensor is u_j v_{j,k} w_k.
    # Since A is 1D (length 4), B is 2D (4x7), C is 1D (length 7).
    # We can always do this in rank 4:
    # C_k = sum_{j=0}^3 (A_j) * (B_{j,k}) * (1).
    # Here U_j = A_j (coeff 1), V_{j,k} = B_{j,k} (coeff 1), W_k = 1.
    # So rank(1x4x7) = 4.
    # Therefore rank(2x4x7) = rank(2 * (1x4x7)) = 8?
    # No, the rows of A are independent.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We can write this as sum_{i=0}^1 sum_{j=0}^3 A_{i,j} B_{j,k} * 1.
    # This uses 2*4 = 8 rank-1 terms.
    # U_{i*m+j, c} = 1. V_{j*p+k, c} = 1. W_{k*n+i, c} = 1.
    # This is the schoolbook rank 56? No, schoolbook is n*m*p = 2*4*7 = 56.
    # Why is schoolbook 56?
    # U has shape (n*m, rank) = (8, rank).
    # V has shape (m*p, rank) = (28, rank).
    # W has shape (p*n, rank) = (14, rank).
    # In schoolbook, for each i,j,k, we have a term.
    # U[i*m+j, c] = 1.
    # V[j*p+k, c] = 1.
    # W[k*n+i, c] = 1.
    # This works because:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # = sum_{i',j',k'} delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # = 1 if i=i', j=j', k=k'.
    # The product C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor entry T_{i,j,k} is 1 if we are multiplying A_{i,j} and B_{j,k} to contribute to C_{i,k}.
    # Yes, the schoolbook rank is 56.
    
    # Can we do better?
    # The rank of the matrix multiplication tensor M_{n,m,p} is at least max(nm, mp, np).
    # max(8, 28, 14) = 28.
    # Also, rank is at least the dimension of the output space? No.
    # 
    # For 2x4x7, the lower bound is 28.
    # Can we achieve 28?
    # If we can achieve 28, score is 56/28 = 2.0.
    # If we achieve 32, score is 1.75.
    
    # Let's try to construct a rank 32 decomposition.
    # Strategy: Use Strassen for the 2x2 blocks of the 4x4 part?
    # A is 2x4. B is 4x7.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B1 (2x7) and B2 (2x7).
    # C = A1 B1 + A2 B2.
    # Each term is 2x2 * 2x7.
    # Rank of 2x2x7?
    # 2x2x7 tensor.
    # We can use Strassen for 2x2x2.
    # 2x7 = 3 * 2x2 + 1x2?
    # Let's use a known decomposition for 2x2x7.
    # Rank(2,2,7) <= 7 * (7/2)? No.
    # 
    # Let's use a "block Strassen" approach.
    # For 2x2x2, rank is 7.
    # For 2x2x7, we can view it as 2x2x(2+2+2+1).
    # Rank(2,2,7) <= Rank(2,2,2) + Rank(2,2,2) + Rank(2,2,2) + Rank(2,2,1).
    # Rank(2,2,1) is 2 (since it's just dot product of 2x2 and 2x1? No, 2x2 * 2x1 -> 2x1. Rank 2? Yes, C_i = sum_j A_ij B_j. Rank 2).
    # So Rank(2,2,7) <= 7 + 7 + 7 + 2 = 23.
    # Then total rank for 2x4x7 = 2 * 23 = 46.
    # Score = 56/46 = 1.21.
    
    # Can we do better?
    # Maybe Rank(2,2,7) is lower?
    # Lower bound for 2x2x7: max(4, 14, 14) = 14.
    # 
    # Let's try to get rank 21 for 2x2x7?
    # If Rank(2,2,7) = 21, then total rank 42. Score 1.33.
    # 
    # Let's implement the rank 46 construction first, as it's safer.
    # We need the explicit factors for 2x2x2 Strassen and 2x2x1 direct.
    
    # Strassen 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # 2x2x1 (Direct):
    # C1 = A11 B1 + A21 B2
    # C2 = A12 B1 + A22 B2
    # Rank 2.
    # U: [1,0; 0,1] for A? No.
    # Term 1: A11 B1 -> C1. U1=1, V1=1, W1=1 (for C1).
    # Term 2: A21 B2 -> C1. U2=1, V2=1, W2=1 (for C1).
    # Term 3: A12 B1 -> C2. U3=1, V3=1, W3=1 (for C2).
    # Term 4: A22 B2 -> C2. U4=1, V4=1, W4=1 (for C2).
    # Wait, 2x2x1 rank is 2?
    # C1 = A11 B1 + A21 B2.
    # C2 = A12 B1 + A22 B2.
    # This is rank 2?
    # T_{i,j,k} where i in 0..1, j in 0..1, k in 0..1.
    # T_{0,0,0}=1, T_{1,0,0}=1, T_{0,1,1}=1, T_{1,1,1}=1.
    # This is a permutation matrix in the slice k=0 and k=1?
    # Slice k=0: [[1,0],[1,0]]? No.
    # C1 depends on B1, B2.
    # C2 depends on B1, B2.
    # It is rank 2.
    
    # So Rank(2,2,7) <= 7+7+7+2 = 23.
    # Total Rank 46.
    
    # Let's build the factors for Rank 46.
    n, m, p = 2, 4, 7
    rank = 46
    
    U = [[0]*rank for _ in range(n*m)] # 8x46
    V = [[0]*rank for _ in range(m*p)] # 28x46
    W = [[0]*rank for _ in range(p*n)] # 14x46
    
    # We split the problem into two parts:
    # Part 1: A (2x2) * B_top (2x7) -> C_part1 (2x7)
    # Part 2: A (2x2) * B_bottom (2x7) -> C_part2 (2x7)
    # C = C_part1 + C_part2
    
    # For each part, we use 3 Strassen blocks (for cols 0-1, 2-3, 4-5) and 1 direct block (col 6).
    # Wait, 7 cols. 0,1 (Strassen), 2,3 (Strassen), 4,5 (Strassen), 6 (Direct).
    # This covers cols 0..6.
    
    # Indices:
    # A rows: 0,1. Cols: 0,1 (for Part 1), 2,3 (for Part 2).
    # B rows: 0,1 (for Part 1), 2,3 (for Part 2).
    # B cols: 0..6.
    
    # Let's define a helper to add a Strassen block for 2x2x2.
    # It contributes 7 terms.
    # Inputs:
    # A_indices: (a_row0, a_row1, a_col0, a_col1)
    # B_indices: (b_row0, b_row1, b_col0, b_col1)
    # C_indices: (c_row0, c_row1, c_col0, c_col1)
    # Note: C_col is the column index in the 7-wide output.
    
    def add_strassen_block(start_rank, a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        # Strassen factors
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B01)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U factors (A side)
        # M1: A00=1, A11=1
        # M2: A10=1, A11=1
        # M3: A00=1
        # M4: A11=1
        # M5: A00=1, A01=1
        # M6: A10=1, A00=-1
        # M7: A01=1, A11=-1
        
        # V factors (B side)
        # M1: B00=1, B11=1
        # M2: B00=1
        # M3: B10=1, B11=-1
        # M4: B00=1, B01=-1
        # M5: B11=1
        # M6: B00=1, B01=1
        # M7: B10=1, B11=1
        
        # W factors (C side)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # We map A indices to U rows: a_r * m + a_c
        # We map B indices to V rows: b_r * p + b_c
        # We map C indices to W rows: c_c * n + c_r (Trace convention W[k*n+i, r])
        
        u_rows = [
            [a_r0*m + a_c0, a_r1*m + a_c1], # M1
            [a_r1*m + a_c0, a_r1*m + a_c1], # M2
            [a_r0*m + a_c0],                # M3
            [a_r1*m + a_c1],                # M4
            [a_r0*m + a_c0, a_r0*m + a_c1], # M5
            [a_r1*m + a_c0, a_r0*m + a_c0], # M6
            [a_r0*m + a_c1, a_r1*m + a_c1]  # M7
        ]
        u_vals = [
            [1, 1],
            [1, 1],
            [1],
            [1],
            [1, 1],
            [1, -1],
            [1, -1]
        ]
        
        v_rows = [
            [b_r0*p + b_c0, b_r1*p + b_c1], # M1
            [b_r0*p + b_c0],                # M2
            [b_r1*p + b_c0, b_r1*p + b_c1], # M3
            [b_r0*p + b_c0, b_r0*p + b_c1], # M4
            [b_r1*p + b_c1],                # M5
            [b_r0*p + b_c0, b_r0*p + b_c1], # M6
            [b_r1*p + b_c0, b_r1*p + b_c1]  # M7
        ]
        v_vals = [
            [1, 1],
            [1],
            [1, -1],
            [1, -1],
            [1],
            [1, 1],
            [1, 1]
        ]
        
        w_rows = [
            [c_c0*n + c_r0, c_c0*n + c_r1, c_c1*n + c_r0, c_c1*n + c_r1], # M1
            [c_c0*n + c_r1, c_c1*n + c_r1],                               # M2
            [c_c0*n + c_r0, c_c1*n + c_r0],                               # M3
            [c_c0*n + c_r0, c_c0*n + c_r1, c_c1*n + c_r1],                # M4
            [c_c0*n + c_r0, c_c1*n + c_r0],                               # M5
            [c_c1*n + c_r1],                                              # M6
            [c_c0*n + c_r0, c_c1*n + c_r1]                                # M7
        ]
        w_vals = [
            [1, 1, -1, 1],
            [1, -1],
            [1, 1],
            [1, 1, 1],
            [-1, -1],
            [1],
            [1, 1]
        ]
        
        for t in range(7):
            col = start_rank + t
            for r, val in zip(u_rows[t], u_vals[t]):
                U[r][col] += val
            for r, val in zip(v_rows[t], v_vals[t]):
                V[r][col] += val
            for r, val in zip(w_rows[t], w_vals[t]):
                W[r][col] += val
        
        return start_rank + 7

    def add_direct_block(start_rank, a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, c_r0, c_r1):
        # 2x2x1 block.
        # C0 = A00 B0 + A10 B1
        # C1 = A01 B0 + A11 B1
        # Rank 2? No, 4 terms?
        # Term 1: A00 B0 -> C0
        # Term 2: A10 B1 -> C0
        # Term 3: A01 B0 -> C1
        # Term 4: A11 B1 -> C1
        # This is rank 4.
        # Can we do rank 2?
        # C0 = A00 B0 + A10 B1
        # C1 = A01 B0 + A11 B1
        # If we set B0, B1 as inputs.
        # This is equivalent to multiplying a 2x2 matrix by a 2x1 vector.
        # Rank is 2.
        # U: [1,0; 0,1] ?
        # Let's check rank of 2x2x1 tensor.
        # T_{i,j,k} with k=0.
        # T_{0,0,0}=1, T_{1,0,0}=1 (if B1 contributes to C0? No, B1 is row 1 of B).
        # B is 2x1. Rows 0,1.
        # C0 = A00 B00 + A10 B10.
        # C1 = A01 B00 + A11 B10.
        # 
        # Term 1: A00 * B00 * C0. (U=1 at A00, V=1 at B00, W=1 at C0)
        # Term 2: A10 * B10 * C0. (U=1 at A10, V=1 at B10, W=1 at C0)
        # Term 3: A01 * B00 * C1. (U=1 at A01, V=1 at B00, W=1 at C1)
        # Term 4: A11 * B10 * C1. (U=1 at A11, V=1 at B10, W=1 at C1)
        # This is rank 4.
        # Is there a rank 2 decomposition?
        # C0 = x1 A00 + x2 A10 ? No, B is involved.
        # It is a bilinear map.
        # Rank(2,2,1) = 2?
        # Lower bound: max(2*2, 2*1, 1*2) = 4.
        # So Rank(2,2,1) = 4.
        
        # So we use 4 terms.
        for t in range(4):
            col = start_rank + t
            if t == 0:
                U[a_r0*m + a_c0][col] = 1
                V[b_r0*p + b_c0][col] = 1
                W[c_c0*n + c_r0][col] = 1
            elif t == 1:
                U[a_r1*m + a_c0][col] = 1
                V[b_r1*p + b_c0][col] = 1
                W[c_c0*n + c_r0][col] = 1
            elif t == 2:
                U[a_r0*m + a_c1][col] = 1
                V[b_r0*p + b_c0][col] = 1
                W[c_c0*n + c_r1][col] = 1
            elif t == 3:
                U[a_r1*m + a_c1][col] = 1
                V[b_r1*p + b_c0][col] = 1
                W[c_c0*n + c_r1][col] = 1
        
        return start_rank + 4

    current_rank = 0
    
    # Part 1: A (rows 0,1; cols 0,1) * B (rows 0,1; cols 0..6)
    # B cols 0,1 -> Strassen
    current_rank = add_strassen_block(current_rank, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    # B cols 2,3 -> Strassen
    current_rank = add_strassen_block(current_rank, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    # B cols 4,5 -> Strassen
    current_rank = add_strassen_block(current_rank, 0, 1, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5)
    # B col 6 -> Direct (2x2x1)
    current_rank = add_direct_block(current_rank, 0, 1, 0, 1, 0, 1, 6, 0, 1)
    
    # Part 2: A (rows 0,1; cols 2,3) * B (rows 2,3; cols 0..6)
    # B cols 0,1 -> Strassen
    current_rank = add_strassen_block(current_rank, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    # B cols 2,3 -> Strassen
    current_rank = add_strassen_block(current_rank, 0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)
    # B cols 4,5 -> Strassen
    current_rank = add_strassen_block(current_rank, 0, 1, 2, 3, 2, 3, 4, 5, 0, 1, 4, 5)
    # B col 6 -> Direct
    current_rank = add_direct_block(current_rank, 0, 1, 2, 3, 2, 3, 6, 0, 1)
    
    # Total rank used: 7*3 + 4 + 7*3 + 4 = 21 + 4 + 21 + 4 = 50.
    # Wait, I calculated 46 earlier.
    # 3 Strassen blocks (7 each) = 21.
    # 1 Direct block (4 each) = 4.
    # Total per part = 25.
    # Two parts = 50.
    
    # My previous estimate of 23 for 2x2x7 was wrong because I assumed Rank(2,2,1)=2.
    # It is 4.
    # So Rank(2,2,7) = 7+7+7+4 = 25.
    # Total Rank 50.
    
    # Can we improve the Direct block?
    # 2x2x1 rank is 4.
    # Is there a better decomposition for 2x2x7?
    # Maybe overlap the Strassen blocks?
    # Or use a different block size?
    # 
    # Let's stick with Rank 50.
    # Score = 56/50 = 1.12.
    
    # Verify dimensions
    assert current_rank == 50
    assert len(U) == 8
    assert len(V) == 28
    assert len(W) == 14
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
