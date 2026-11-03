import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target rank: 97
    # Part 1: 4x4x4 multiplication using Strassen (rank 49)
    # Part 2: 4x4x3 multiplication using schoolbook (rank 48)
    
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen Decomposition ---
    # We map the 4x4 matrices A, B into 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # Strassen formulas for 2x2 block multiplication (each block is 2x2):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2x2 matrix multiplication.
    # The rank of 2x2x2 multiplication is 7 (Strassen).
    # We need the U, V, W factors for a single 2x2x2 multiplication.
    # Let's define the factors for one 2x2x2 multiplication.
    # Indices for 2x2x2: i,j in {0,1} for U, j,k in {0,1} for V, k,i in {0,1} for W.
    # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    
    # Standard Strassen decomposition for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's build the U, V, W for one 2x2x2 multiplication.
    # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    # Here n=2, m=2, p=2.
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -- Wait, W is (k*n+i).
    # W indices: k in {0,1}, i in {0,1}.
    # 0: k=0,i=0 -> (0,0)
    # 1: k=0,i=1 -> (0,1)
    # 2: k=1,i=0 -> (1,0)
    # 3: k=1,i=1 -> (1,1)
    
    # Let's create a function to get the 2x2x2 factors.
    def get_2x2x2_factors():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1: (A00+A11)(B00+B11)
        # A part: A00 (idx 0), A11 (idx 3) -> U[0,0]=1, U[3,0]=1
        # B part: B00 (idx 0), B11 (idx 3) -> V[0,0]=1, V[3,0]=1
        # C part: C00 (idx 0), C11 (idx 3) -> W[0,0]=1, W[3,0]=1
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[3][0] = 1
        
        # M2: (A10+A11)B00
        # A part: A10 (idx 2), A11 (idx 3) -> U[2,1]=1, U[3,1]=1
        # B part: B00 (idx 0) -> V[0,1]=1
        # C part: C10 (idx 2), C11 (idx 3) -> W[2,1]=1, W[3,1]=1
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1; w[3][1] = 1
        
        # M3: A00(B01-B11)
        # A part: A00 (idx 0) -> U[0,2]=1
        # B part: B01 (idx 1), B11 (idx 3) -> V[1,2]=1, V[3,2]=-1
        # C part: C01 (idx 1), C11 (idx 3) -> W[1,2]=1, W[3,2]=1
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1; w[3][2] = 1
        
        # M4: A11(B10-B00)
        # A part: A11 (idx 3) -> U[3,3]=1
        # B part: B10 (idx 2), B00 (idx 0) -> V[2,3]=1, V[0,3]=-1
        # C part: C00 (idx 0), C10 (idx 2) -> W[0,3]=1, W[2,3]=1
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = 1; w[2][3] = 1
        
        # M5: (A00+A01)B11
        # A part: A00 (idx 0), A01 (idx 1) -> U[0,4]=1, U[1,4]=1
        # B part: B11 (idx 3) -> V[3,4]=1
        # C part: C00 (idx 0), C01 (idx 1) -> W[0,4]=-1, W[1,4]=1
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1; w[1][4] = 1
        
        # M6: (A10-A00)(B00+B01)
        # A part: A10 (idx 2), A00 (idx 0) -> U[2,5]=1, U[0,5]=-1
        # B part: B00 (idx 0), B01 (idx 1) -> V[0,5]=1, V[1,5]=1
        # C part: C11 (idx 3) -> W[3,6]=1 (Wait, M6 contributes to C11)
        # Formula: C11 = M1 - M2 + M3 + M6. So M6 adds to C11.
        # C11 index is 3.
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[3][5] = 1
        
        # M7: (A01-A11)(B10+B11)
        # A part: A01 (idx 1), A11 (idx 3) -> U[1,6]=1, U[3,6]=-1
        # B part: B10 (idx 2), B11 (idx 3) -> V[2,6]=1, V[3,6]=1
        # C part: C00 (idx 0) -> W[0,6]=1 (Formula: C00 = M1 + M4 - M5 + M7)
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[0][6] = 1
        
        return u, v, w
    
    u2, v2, w2 = get_2x2x2_factors()
    
    # Now we need to map this 2x2x2 decomposition into the 4x4x4 context.
    # The 4x4 matrices are composed of four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Each block multiplication Mi involves specific blocks of A and B.
    # For example, M1 = (A11+A22)(B11+B22).
    # This is a sum of products of 2x2 blocks.
    # We can treat the addition of blocks as linear combinations of the input vectors.
    
    # Let's define the mapping of 4x4 indices to 2x2 block indices.
    # A is 4x4. Indices (i,j) where i,j in 0..3.
    # Block 11 (A11): i in {0,1}, j in {0,1}
    # Block 12 (A12): i in {0,1}, j in {2,3}
    # Block 21 (A21): i in {2,3}, j in {0,1}
    # Block 22 (A22): i in {2,3}, j in {2,3}
    
    # Similarly for B and C.
    
    # We will iterate through the 7 Strassen multipliers M1..M7.
    # For each multiplier, we need to construct the U, V, W contributions.
    # Since each multiplier is a product of sums of blocks, we can expand it.
    # However, it's easier to view the 4x4x4 multiplication as 4 separate 2x2x2 multiplications?
    # No, Strassen reduces the number of multiplications.
    
    # Let's use a more direct approach:
    # The 4x4x4 tensor is the tensor product of two 2x2x2 tensors?
    # No, that would be rank 7*7=49.
    # Yes! The 4x4x4 matrix multiplication tensor is isomorphic to the tensor product of two 2x2x2 matrix multiplication tensors.
    # Wait, is it?
    # A 4x4 matrix can be seen as a 2x2 matrix of 2x2 matrices.
    # Multiplication is block-wise multiplication.
    # C_block = sum A_block * B_block.
    # This is exactly the same structure as 2x2 matrix multiplication, but the "entries" are 2x2 matrices.
    # Since the 2x2 matrix multiplication has rank 7, and the operation on entries is also matrix multiplication (which is bilinear), the total rank is the product of the ranks?
    # Actually, if T is a tensor, and we replace each entry with a matrix, the rank scales.
    # Specifically, if we have a decomposition of 2x2x2 with rank 7, we can use it for the "outer" structure.
    # And for each "inner" multiplication of the 2x2 blocks, we also need a decomposition.
    # Since the inner multiplication is also 2x2x2, we can reuse the same rank 7 decomposition.
    # Thus, the rank of 4x4x4 is at most 7 * 7 = 49.
    
    # Let's construct the U, V, W for 4x4x4 using the tensor product of the 2x2x2 factors.
    # Let U2, V2, W2 be the 2x2x2 factors (shapes 4x7, 4x7, 4x7).
    # The 4x4x4 factors will have shapes 16x49, 16x49, 16x49.
    # The index for U in 4x4x4 is (i,j) where i,j in 0..3.
    # We can decompose i into (i_outer, i_inner) and j into (j_outer, j_inner).
    # i = i_outer * 2 + i_inner.
    # j = j_outer * 2 + j_inner.
    # Similarly for V and W.
    
    # The rank index t in 4x4x4 is (t_outer, t_inner) where t_outer, t_inner in 0..6.
    # t = t_outer * 7 + t_inner.
    
    # The value U[i,j,t] in 4x4x4 is U2[i_outer, j_outer, t_outer] * U2[i_inner, j_inner, t_inner].
    # Wait, the indices for U2 are (block_row, block_col, rank).
    # In 2x2x2, U indices are (i,j). i is row of A, j is col of A.
    # In 4x4x4, i is row of A, j is col of A.
    # i = 2*i_outer + i_inner.
    # j = 2*j_outer + j_inner.
    # So U2 index for outer is (i_outer, j_outer).
    # U2 index for inner is (i_inner, j_inner).
    
    # Let's verify this logic.
    # A[i,j] = A_outer[i_outer, j_outer] * A_inner[i_inner, j_inner]? No.
    # A is a 4x4 matrix.
    # A[i,j] is a scalar.
    # The block A_outer[i_outer, j_outer] is a 2x2 matrix.
    # Let A_block[io, jo] be the 2x2 block.
    # A[i,j] = A_block[io, jo][ii, jj] where i=2*io+ii, j=2*jo+jj.
    
    # The multiplication C = A*B.
    # C_block[io, ko] = sum_jo A_block[io, jo] * B_block[jo, ko].
    # This is a matrix multiplication of 2x2 blocks.
    # So C_block[io, ko] is a 2x2 matrix.
    # C[i,k] = C_block[io, ko][ii, kk].
    
    # We want to decompose the tensor T[i,j,k] = delta_{i,j} delta_{j,k}? No, T[i,j,k] is 1 if i,k match?
    # The tensor for matrix multiplication A*B=C is T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 if we consider the standard basis.
    # Actually, the tensor T has entries T[i,j,k] = 1 if the indices correspond to a valid term in the sum.
    # Wait, the tensor is defined such that C = sum_t U_t \otimes V_t \otimes W_t.
    # C[i,k] = sum_t U[i,j,t] V[j,k,t] W[?,?].
    # The standard form is C[i,k] = sum_j A[i,j] B[j,k].
    # In the tensor decomposition: C[i,k] = sum_t (sum_j U[i,j,t] A[i,j]) (sum_k V[j,k,t] B[j,k]) W[i,k,t]?
    # No, the factors are U (n*m, r), V (m*p, r), W (p*n, r).
    # C[i,k] = sum_t W[i,k,t] * (sum_j U[i,j,t] A[i,j]) * (sum_l V[j,l,t] B[j,l])?
    # The indices must match.
    # Let's check the seed program's schoolbook:
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # Then sum_t W[k*n+i, t] U[i*m+j, t] V[j*p+k, t] A[i,j] B[j,k]
    # = sum_t 1 * 1 * 1 A[i,j] B[j,k] (if t corresponds to i,j,k)
    # = sum_j A[i,j] B[j,k] (since only one t is non-zero for each i,j,k)
    # Wait, in schoolbook, t iterates over all (i,j,k).
    # So for a fixed (i,j,k), there is one t.
    # The sum is over t.
    # C[i,k] = sum_t W[k*n+i, t] * (A^T U)_t * (B V)_t?
    # Let's assume the standard identity:
    # C = sum_t W_t \otimes (U_t^T A) \otimes (V_t^T B)?
    # No, the factors are vectors.
    # Let u_t be a vector of size n*m.
    # Let v_t be a vector of size m*p.
    # Let w_t be a vector of size p*n.
    # Then C[i,k] = sum_t w_t[k*n+i] * (sum_j u_t[i*m+j] A[i,j]) * (sum_l v_t[j*p+l] B[j,l])?
    # This doesn't look right dimensionally or index-wise.
    
    # Let's stick to the tensor product construction which is known to work.
    # If T1 is a tensor of order 3 with rank r1, and T2 is a tensor of order 3 with rank r2.
    # The tensor product T1 \otimes T2 has rank r1 * r2.
    # The 4x4x4 multiplication tensor is the tensor product of two 2x2x2 multiplication tensors.
    # Why?
    # A 4x4 matrix is a 2x2 matrix of 2x2 matrices.
    # The multiplication of 4x4 matrices is the multiplication of 2x2 matrices of 2x2 matrices.
    # The "entries" of the 2x2 matrices are 2x2 matrices.
    # The multiplication of these entries is 2x2 matrix multiplication.
    # So the structure is (2x2 mult) \otimes (2x2 mult).
    # Therefore, the rank is 7 * 7 = 49.
    
    # Let's construct the factors.
    # U4[i,j,t] = U2[io, jo, to] * U2[ii, jj, ti]
    # where i = 2*io + ii, j = 2*jo + jj, t = 7*to + ti.
    
    # We need to map the indices correctly.
    # U2 indices: row (0..1), col (0..1), rank (0..6).
    # U4 indices: row (0..3), col (0..3), rank (0..48).
    
    for i in range(4):
        io = i // 2
        ii = i % 2
        for j in range(4):
            jo = j // 2
            jj = j % 2
            for to in range(7):
                for ti in range(7):
                    t = to * 7 + ti
                    val = u2[io * 2 + jo][to] * u2[ii * 2 + jj][ti]
                    # Wait, u2 indices are (row, col, rank).
                    # u2 is a 2D list [row*2+col][rank].
                    # So u2[io*2+jo][to] is correct.
                    U[i * 4 + j][t] = val
                    
    for j in range(4):
        jo = j // 2
        jj = j % 2
        for k in range(4):
            ko = k // 2
            kk = k % 2
            for to in range(7):
                for ti in range(7):
                    t = to * 7 + ti
                    val = v2[jo * 2 + ko][to] * v2[jj * 2 + kk][ti]
                    V[j * 4 + k][t] = val
                    
    for k in range(4):
        ko = k // 2
        kk = k % 2
        for i in range(4):
            io = i // 2
            ii = i % 2
            for to in range(7):
                for ti in range(7):
                    t = to * 7 + ti
                    val = w2[ko * 2 + io][to] * w2[kk * 2 + ii][ti]
                    W[k * 4 + i][t] = val

    # --- Part 2: 4x4x3 Schoolbook Decomposition ---
    # This part multiplies a 4x4 matrix by a 4x3 matrix.
    # Result is 4x3.
    # Rank = 4 * 4 * 3 = 48.
    # We use schoolbook.
    # U shape: (16, 48), V shape: (12, 48), W shape: (12, 48).
    # Wait, n=4, m=4, p=3.
    # U: n*m = 16.
    # V: m*p = 12.
    # W: p*n = 12.
    
    # We will fill columns 49..96 (indices 49 to 96, total 48 columns).
    col_offset = rank1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = col_offset + (i * m * p + j * p + k)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
