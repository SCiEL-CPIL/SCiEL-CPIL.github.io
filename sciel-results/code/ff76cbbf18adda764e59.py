"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2x2():
    """Returns U, V, W for 2x2 * 2x2 -> 2x2 using Strassen (rank 7)."""
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p1+p4-p5+p6, p2+p4], [p3+p5, p1-p3+p4+p7]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-h)
    # p5 = (a+b)h
    # p6 = (c-d)(e+f)
    # p7 = (b-d)(g+h)
    
    # U (4x7), V (4x7), W (4x7)
    # Indices: U[i*2+j], V[i*2+j], W[i*2+j]
    # i,j in 0..1
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1: (a+d)(e+h)
    # a=U[0,0], d=U[1,1] -> U[0,0]=1, U[1,1]=1
    # e=V[0,0], h=V[1,1] -> V[0,0]=1, V[1,1]=1
    # Contributes to C[0,0] (+), C[1,1] (+)
    W[0,0] += 1; W[1,1] += 1
    U[0,0] = 1; U[1,1] = 1
    V[0,0] = 1; V[1,1] = 1
    
    # p2: (c+d)e
    # c=U[1,0], d=U[1,1] -> U[1,0]=1, U[1,1]=1
    # e=V[0,0] -> V[0,0]=1
    # Contributes to C[0,1] (+)
    W[0,1] += 1
    U[1,0] = 1; U[1,1] = 1
    V[0,0] = 1
    
    # p3: a(f-h)
    # a=U[0,0] -> U[0,0]=1
    # f=V[0,1], h=V[1,1] -> V[0,1]=1, V[1,1]=-1
    # Contributes to C[1,0] (+)
    W[1,0] += 1
    U[0,0] = 1
    V[0,1] = 1; V[1,1] = -1
    
    # p4: d(g-h)
    # d=U[1,1] -> U[1,1]=1
    # g=V[1,0], h=V[1,1] -> V[1,0]=1, V[1,1]=-1
    # Contributes to C[0,0] (+), C[1,1] (+)
    W[0,0] += 1; W[1,1] += 1
    U[1,1] = 1
    V[1,0] = 1; V[1,1] = -1
    
    # p5: (a+b)h
    # a=U[0,0], b=U[0,1] -> U[0,0]=1, U[0,1]=1
    # h=V[1,1] -> V[1,1]=1
    # Contributes to C[0,0] (-), C[1,0] (+)
    W[0,0] -= 1; W[1,0] += 1
    U[0,0] = 1; U[0,1] = 1
    V[1,1] = 1
    
    # p6: (c-d)(e+f)
    # c=U[1,0], d=U[1,1] -> U[1,0]=1, U[1,1]=-1
    # e=V[0,0], f=V[0,1] -> V[0,0]=1, V[0,1]=1
    # Contributes to C[0,0] (+)
    W[0,0] += 1
    U[1,0] = 1; U[1,1] = -1
    V[0,0] = 1; V[0,1] = 1
    
    # p7: (b-d)(g+h)
    # b=U[0,1], d=U[1,1] -> U[0,1]=1, U[1,1]=-1
    # g=V[1,0], h=V[1,1] -> V[1,0]=1, V[1,1]=1
    # Contributes to C[1,1] (+)
    W[1,1] += 1
    U[0,1] = 1; U[1,1] = -1
    V[1,0] = 1; V[1,1] = 1
    
    return U, V, W

def _outer_2x1x3():
    """Returns U, V, W for 2x1 * 1x3 -> 2x3 (rank 6)."""
    # A (2x1), B (1x3), C (2x3)
    # C[i,k] = A[i,0] * B[0,k]
    # Rank 6: for each i in 0..1, k in 0..2
    U = np.zeros((2, 6))
    V = np.zeros((1, 6))
    W = np.zeros((6, 2)) # W[k*2+i]
    
    col = 0
    for i in range(2):
        for k in range(3):
            U[i, col] = 1
            V[0, col] = 1
            W[k*2 + i, col] = 1
            col += 1
    return U, V, W

def _schoolbook_2x2x3():
    """Returns U, V, W for 2x2 * 2x3 -> 2x3 (rank 12)."""
    # A (2x2), B (2x3), C (2x3)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Rank 12: for each i in 0..1, j in 0..1, k in 0..2
    U = np.zeros((4, 12))
    V = np.zeros((6, 12)) # B is 2x3, flattened 6
    W = np.zeros((6, 12)) # C is 2x3, flattened 6. W[k*2+i]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2 + j, col] = 1
                V[j*3 + k, col] = 1
                W[k*2 + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the global U, V, W by concatenating blocks.
    # Total rank = 42 (Part 1) + 72 (Part 2) = 114.
    total_rank = 114
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    # --- Part 1: A1 (4x2) * B1 (2x6) -> C1 (6x4) ---
    # A1 is rows 0:4, cols 0:2 of A.
    # B1 is rows 0:2, cols 0:6 of B.
    # C1 is rows 0:6, cols 0:4 of C.
    
    # Split A1 into A1a (2x2) and A1b (2x2)
    # A1a: A[0:2, 0:2]
    # A1b: A[2:4, 0:2]
    
    # For each, split B1 into B11, B12, B13 (2x2 each)
    # B11: B[0:2, 0:2]
    # B12: B[0:2, 2:4]
    # B13: B[0:2, 4:6]
    
    # C1a = A1a * B1 (2x6). Split into C1a1, C1a2, C1a3 (2x2 each)
    # C1a1: C[0:2, 0:2]
    # C1a2: C[0:2, 2:4]
    # C1a3: C[0:2, 4:6]
    
    # C1b = A1b * B1 (2x6). Split into C1b1, C1b2, C1b3 (2x2 each)
    # C1b1: C[0:2, 2:4] ? No, C1b is rows 2:4 of C?
    # C is 6x4. C1 is top 4 rows? No, C1 is result of A1(4x2)*B1(2x6) -> 4x6?
    # Wait, C is p x n = 6 x 4.
    # A1 is 4x2. B1 is 2x6.
    # A1 * B1 is 4x6.
    # But C is 6x4.
    # My block decomposition logic was:
    # C = A B. A is 4x5, B is 5x6. C is 4x6?
    # No, C is usually n x p. But the task says W[k*n+i].
    # W index is k*n+i. k is row of C (0..p-1), i is col of C (0..n-1).
    # So C is p x n = 6 x 4.
    # Standard matrix multiplication: C = B^T A^T?
    # Let's check the trace convention: W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # Sum over r, i, j.
    # Term: U[i*m+j] V[j*p+k] W[k*n+i]
    # This corresponds to C[k,i] = sum_{j} A[i,j] B[j,k].
    # So C is p x n.
    # A is n x m. B is m x p.
    
    # So in Part 1:
    # A1 is 4x2 (n=4, m_part=2).
    # B1 is 2x6 (m_part=2, p=6).
    # C1 is 6x4 (p=6, n=4).
    # C1[k,i] = sum_{j=0}^{1} A1[i,j] B1[j,k].
    
    # Split n=4 into 2+2.
    # A1a: rows 0:2 of A1. (2x2)
    # A1b: rows 2:4 of A1. (2x2)
    
    # C1a: rows 0:6, cols 0:2 of C1. (6x2)
    # C1b: rows 0:6, cols 2:4 of C1. (6x2)
    
    # C1a[k,i] = sum_{j=0}^{1} A1a[i,j] B1[j,k].
    # C1b[k,i] = sum_{j=0}^{1} A1b[i,j] B1[j,k].
    
    # Now split p=6 into 2+2+2.
    # B11: B1[0:2, 0:2] (2x2)
    # B12: B1[0:2, 2:4] (2x2)
    # B13: B1[0:2, 4:6] (2x2)
    
    # C1a1: C1a[0:2, :] (2x2)
    # C1a2: C1a[2:4, :] (2x2)
    # C1a3: C1a[4:6, :] (2x2)
    
    # C1a1[k,i] = sum_{j=0}^{1} A1a[i,j] B11[j,k].
    # This is a 2x2 * 2x2 -> 2x2 multiplication.
    
    # So we need 3 Strassen decompositions for C1a.
    # And 3 for C1b.
    # Total 6 Strassen decompositions.
    # Rank 6 * 7 = 42.
    
    # Indices:
    # A1a[i,j] is at A[i, j] (global i 0:2, j 0:2)
    # A1b[i,j] is at A[i+2, j] (global i 2:4, j 0:2)
    
    # B11[j,k] is at B[j, k] (global j 0:2, k 0:2)
    # B12[j,k] is at B[j, k+2] (global j 0:2, k 2:4)
    # B13[j,k] is at B[j, k+4] (global j 0:2, k 4:6)
    
    # C1a1[k,i] is at C[k, i] (global k 0:2, i 0:2)
    # C1a2[k,i] is at C[k+2, i] (global k 2:4, i 0:2)
    # C1a3[k,i] is at C[k+4, i] (global k 4:6, i 0:2)
    
    # C1b1[k,i] is at C[k, i+2] (global k 0:2, i 2:4)
    # C1b2[k,i] is at C[k+2, i+2] (global k 2:4, i 2:4)
    # C1b3[k,i] is at C[k+4, i+2] (global k 4:6, i 2:4)
    
    for block_idx in range(6):
        U_s, V_s, W_s = _strassen_2x2x2()
        
        # Determine which A block, B block, C block
        if block_idx < 3:
            # C1a
            a_row_offset = 0
            c_col_offset = 0
            b_col_offset = block_idx * 2
            c_row_offset = block_idx * 2
        else:
            # C1b
            a_row_offset = 2
            c_col_offset = 2
            b_col_offset = (block_idx - 3) * 2
            c_row_offset = (block_idx - 3) * 2
            
        # Map U_s (4x7) to U (20x114)
        # U_s[i*2+j, r] -> U[(a_row_offset+i)*5 + j, col_offset+r]
        for i in range(2):
            for j in range(2):
                for r in range(7):
                    U[(a_row_offset + i) * 5 + j, col_offset + r] += U_s[i*2 + j, r]
                    
        # Map V_s (4x7) to V (30x114)
        # V_s[j*2+k, r] -> V[j*6 + (b_col_offset+k), col_offset+r]
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    V[j * 6 + (b_col_offset + k), col_offset + r] += V_s[j*2 + k, r]
                    
        # Map W_s (4x7) to W (24x114)
        # W_s[k*2+i, r] -> W[(c_row_offset+k)*4 + (c_col_offset+i), col_offset+r]
        for k in range(2):
            for i in range(2):
                for r in range(7):
                    W[(c_row_offset + k) * 4 + (c_col_offset + i), col_offset + r] += W_s[k*2 + i, r]
                    
        col_offset += 7
        
    # --- Part 2: A2 (4x3) * B2 (3x6) -> C2 (6x4) ---
    # A2 is rows 0:4, cols 2:5 of A.
    # B2 is rows 2:5, cols 0:6 of B.
    # C2 is rows 0:6, cols 0:4 of C.
    
    # Split n=4 into 2+2.
    # A2a: rows 0:2 of A2. (2x3)
    # A2b: rows 2:4 of A2. (2x3)
    
    # C2a: rows 0:6, cols 0:2 of C2. (6x2)
    # C2b: rows 0:6, cols 2:4 of C2. (6x2)
    
    # Split p=6 into 3+3.
    # B21: B2[0:3, 0:3] (3x3)
    # B22: B2[0:3, 3:6] (3x3)
    
    # C2a1: C2a[0:3, :] (3x2)
    # C2a2: C2a[3:6, :] (3x2)
    
    # C2b1: C2b[0:3, :] (3x2)
    # C2b2: C2b[3:6, :] (3x2)
    
    # Each is 2x3 * 3x3 -> 3x2?
    # No, C2a1[k,i] = sum_{j=0}^{2} A2a[i,j] B21[j,k].
    # A2a is 2x3. B21 is 3x3. Result is 2x3?
    # Wait, C is 6x4.
    # C2a1 is k in 0:3, i in 0:2. So 3x2.
    # A2a[i,j] (i 0:2, j 0:3). B21[j,k] (j 0:3, k 0:3).
    # Sum j. Result C[i,k] is 2x3.
    # But C is stored as W[k*n+i].
    # So C[k,i] corresponds to W[k*4+i].
    # My previous logic for C1 was C[k,i] with k row, i col.
    # Here C2a1 is 3x2 (k 0:3, i 0:2).
    # So it is a 2x3 * 3x3 -> 2x3 multiplication, transposed in storage?
    # No, the tensor is symmetric in the sense of the sum.
    # The sub-problem is $M(2,3,3)$ but the output is $3 \times 2$ in C's indexing?
    # $C[k,i] = \sum A[i,j] B[j,k]$.
    # If we fix i, k, it's a scalar.
    # The block is defined by $i \in \{0,1\}, k \in \{0,1,2\}$.
    # This is a $2 \times 3 \times 3$ tensor.
    # We can use the decomposition for $M(2,3,3)$.
    
    # We determined $M(2,3,3) \le 18$ using Outer(2,1,3) + Schoolbook(2,2,3).
    # Let's implement that combination.
    
    for block_idx in range(4):
        # 4 blocks: C2a1, C2a2, C2b1, C2b2
        
        # Determine offsets
        if block_idx < 2:
            a_row_offset = 0 # A2a
            c_col_offset = 0 # C2a
            b_col_offset = block_idx * 3
            c_row_offset = block_idx * 3
        else:
            a_row_offset = 2 # A2b
            c_col_offset = 2 # C2b
            b_col_offset = (block_idx - 2) * 3
            c_row_offset = (block_idx - 2) * 3
            
        # Sub-problem: 2x3 * 3x3 -> 2x3 (stored as 3x2 in C)
        # A_sub is 2x3. B_sub is 3x3.
        
        # Decomposition:
        # Split m=3 into 1+2.
        # A_sub = [A_sub1, A_sub2] (2x1, 2x2)
        # B_sub = [B_sub1; B_sub2] (1x3, 2x3)
        
        # Term 1: A_sub1 * B_sub1 (2x1 * 1x3 -> 2x3)
        # Use Outer 2x1x3 (rank 6)
        
        # Term 2: A_sub2 * B_sub2 (2x2 * 2x3 -> 2x3)
        # Use Schoolbook 2x2x3 (rank 12)
        
        # Total rank 18.
        
        U_o, V_o, W_o = _outer_2x1x3()
        U_s, V_s, W_s = _schoolbook_2x2x3()
        
        # Map Outer Part
        # A_sub1 is cols 0:1 of A_sub.
        # A_sub is cols 2:5 of A.
        # So A_sub1 is cols 2:3 of A.
        # A_sub1[i,0] is A[i + a_row_offset, 2].
        
        # B_sub1 is rows 0:1 of B_sub.
        # B_sub is rows 2:5 of B.
        # So B_sub1 is rows 2:3 of B.
        # B_sub1[0,k] is B[2, k + b_col_offset].
        
        # C_sub1 is the result.
        # C_sub1[k,i] is C[k + c_row_offset, i + c_col_offset].
        
        # Outer U (2x6): U[i, r] -> A[i, 0]
        # Map to U[(i + a_row_offset)*5 + 2, col_offset + r]
        for i in range(2):
            for r in range(6):
                U[(i + a_row_offset) * 5 + 2, col_offset + r] += U_o[i, r]
                
        # Outer V (1x6): V[0, r] -> B[0, k]
        # Map to V[2*6 + (0 + 0)*3 + (k + b_col_offset), col_offset + r]?
        # B is 5x6. B[j, k].
        # B_sub1 is row 0 of B_sub, which is row 2 of B.
        # So B[2, k + b_col_offset].
        # V index is j*6 + k.
        # So V[2*6 + (k + b_col_offset), col_offset + r].
        for k in range(3):
            for r in range(6):
                V[2 * 6 + (k + b_col_offset), col_offset + r] += V_o[0, r]
                
        # Outer W (6x2): W[k*2+i, r] -> C[k, i]
        # Map to W[(k + c_row_offset)*4 + (i + c_col_offset), col_offset + r]
        for k in range(3):
            for i in range(2):
                for r in range(6):
                    W[(k + c_row_offset) * 4 + (i + c_col_offset), col_offset + r] += W_o[k*2 + i, r]
                    
        col_offset += 6
        
        # Map Schoolbook Part
        # A_sub2 is cols 1:3 of A_sub.
        # So cols 3:5 of A.
        # A_sub2[i,j] is A[i + a_row_offset, 3 + j].
        
        # B_sub2 is rows 1:3 of B_sub.
        # So rows 3:5 of B.
        # B_sub2[j,k] is B[3 + j, k + b_col_offset].
        
        # C_sub2 is the result.
        # C_sub2[k,i] is C[k + c_row_offset, i + c_col_offset].
        
        # Schoolbook U (4x12): U[i*2+j, r] -> A[i, j]
        # Map to U[(i + a_row_offset)*5 + (3 + j), col_offset + r]
        for i in range(2):
            for j in range(2):
                for r in range(12):
                    U[(i + a_row_offset) * 5 + (3 + j), col_offset + r] += U_s[i*2 + j, r]
                    
        # Schoolbook V (6x12): V[j*3+k, r] -> B[j, k]
        # Map to V[(3 + j)*6 + (k + b_col_offset), col_offset + r]
        for j in range(2):
            for k in range(3):
                for r in range(12):
                    V[(3 + j) * 6 + (k + b_col_offset), col_offset + r] += V_s[j*3 + k, r]
                    
        # Schoolbook W (6x12): W[k*2+i, r] -> C[k, i]
        # Map to W[(k + c_row_offset)*4 + (i + c_col_offset), col_offset + r]
        for k in range(3):
            for i in range(2):
                for r in range(12):
                    W[(k + c_row_offset) * 4 + (i + c_col_offset), col_offset + r] += W_s[k*2 + i, r]
                    
        col_offset += 12
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
