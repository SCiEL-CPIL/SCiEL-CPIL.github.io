"""Program for matrix multiplication 4x4x6 using structured Kronecker-sum formulation."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    
    # We'll use a block decomposition approach.
    # Split the 4x4 matrix into 2x2 blocks, and 4x6 into 2x3 blocks.
    # This gives us 4 blocks of 2x2 * 2x3 multiplication.
    # Each 2x2 * 2x3 multiplication can be done with rank 6 (schoolbook for 2x2x3 is 12, but we can do better).
    # Actually, let's use known results: 2x2x3 has rank 6? No, schoolbook is 2*2*3=12.
    # The minimal rank for 2x2x3 is actually 6? Let me check: for 2x2x2 it's 7 (Strassen).
    # For 2x2x3, I believe the rank is 6 or 7. Let's use a known decomposition.
    
    # Alternative approach: Use Strassen-like ideas on blocks.
    # Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B (4x6) into 2x3 blocks: B = [[B11, B12], [B21, B22]] where each is 2x3
    # Then C = A*B has blocks:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each block product is 2x2 * 2x3 -> 2x3
    # We can use Strassen's algorithm adapted for 2x2 * 2x3.
    
    # For 2x2 * 2x3, let's define the Strassen-like decomposition.
    # Let A = [[a11, a12], [a21, a22]] (2x2)
    # Let B = [[b11, b12, b13], [b21, b22, b23]] (2x3)
    # C = A*B = [[c11, c12, c13], [c21, c22, c23]] (2x3)
    
    # Standard Strassen for 2x2*2x2 uses 7 multiplications.
    # For 2x2*2x3, we can use 7 multiplications as well:
    # M1 = (a11+a21) * (b11+b21)  -> but B has 3 columns, so this needs care.
    
    # Actually, for rectangular multiplication, the rank of 2x2x3 tensor:
    # The tensor T_{2,2,3} has rank 6? Let me think...
    # Schoolbook rank is 12. 
    # I recall that rank(2x2x3) = 6 is not correct; it's actually 7 or 8.
    # Let's just use a direct construction.
    
    # Let me use a simpler approach: use the fact that 4x4x6 can be decomposed
    # using the structure of 2x2x2 blocks.
    
    # Actually, let's try a different strategy: use the known rank of 4x4x4 which is 49 (Strassen's algorithm for 4x4x4).
    # For 4x4x6, we can pad B to 4x4x4 and then... no, that doesn't work directly.
    
    # Let me use a more systematic approach. I'll construct the decomposition using
    # the block structure and known small-matrix multiplication algorithms.
    
    # For 2x2 * 2x3 multiplication, here's a rank-7 decomposition:
    # A = [[a, b], [c, d]] (2x2), B = [[e, f, g], [h, i, j]] (2x3)
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    # Using Strassen-like:
    # M1 = (a+c)(e+h) = ae+ah+ce+ch
    # M2 = (a+c)i = ai+ci  -- wait, this doesn't quite work for 3 columns.
    
    # Let me use a different known decomposition for 2x2x3.
    # Actually, I'll just use the schoolbook for the sub-problems but combine them smartly.
    
    # Better approach: Use the fact that we can multiply 4x4 * 4x6 by viewing it as
    # two 4x4 * 4x3 multiplications, and each 4x4*4x3 can be done with rank 4*4*3=48 schoolbook,
    # or better using Strassen on 4x4*4x4 padded.
    
    # Let me just use Strassen's algorithm recursively for 4x4x4 and adapt for 4x4x6.
    
    # Actually, the simplest improvement: use Strassen's algorithm for the 4x4x4 part.
    # 4x4x4 has rank 49 with Strassen's algorithm (7^2 = 49).
    # For 4x4x6, we can split B into two 4x3 matrices, giving 2 * (4x4x3).
    # 4x4x3: can be done with Strassen by padding to 4x4x4, so rank 49 each, total 98.
    # That's worse than schoolbook 96.
    
    # Let me try yet another approach: use the bilinear complexity of matrix multiplication.
    # For 4x4x6, the schoolbook rank is 96.
    # Using Strassen's algorithm: 4x4x4 has rank 49. 
    # We can compute 4x4 * 4x6 as follows:
    # Write B = [B1 | B2] where B1, B2 are 4x3.
    # C = A * B = [A*B1 | A*B2]
    # Each A*B1 is 4x4 * 4x3. 
    # To compute 4x4 * 4x3, we can pad B1 to 4x4 by adding a zero column, compute 4x4*4x4 with rank 49,
    # and discard the last column of the result. But this uses 49 multiplications per 4x3 block.
    # Total: 2 * 49 = 98. Worse than 96.
    
    # So Strassen on 4x4x4 blocks doesn't help for 4x4x6 directly.
    
    # Let me try a different block size. What about using 2x2x2 Strassen blocks?
    # 4x4x6 can be seen as 2x2 blocks of 2x2x3 multiplications.
    # There are 2*2 = 4 such blocks (since 4/2=2 in both dimensions, and 6/3=2... wait, 6/2=3).
    # Actually, if we split into 2x2 blocks: A has 2x2 = 4 blocks of 2x2.
    # B (4x6) has 2x3 = 6 blocks of 2x2? No, 6/2=3, so 2 rows of blocks, 3 columns of blocks, each 2x2.
    # So B has 2*3 = 6 blocks of 2x2.
    # C (4x6) has 2*3 = 6 blocks of 2x2.
    
    # Each C_block = sum of products of A_blocks * B_blocks.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Each term is 2x2 * 2x2 -> 2x2, which can be done with rank 7 (Strassen).
    # There are 12 such terms (4 A-blocks * 3 B-blocks... no, it's more complex).
    
    # Actually, the total number of 2x2*2x2 multiplications needed is:
    # For each of the 6 C-blocks, we need 2 multiplications, so 12 multiplications.
    # Each uses rank 7, so total rank = 12 * 7 = 84.
    # That's better than 96!
    
    # Let me implement this.
    
    # Strassen's algorithm for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now I need to construct the U, V, W matrices for the full 4x4x6 problem.
    
    # The tensor for matrix multiplication is: T[i,j,k] = 1 if the (i,j) element of A times (j,k) element of B contributes to (i,k) element of C.
    # In our indexing: U has shape (n*m, rank) = (16, rank), V has shape (m*p, rank) = (24, rank), W has shape (p*n, rank) = (24, rank).
    # The convention is W[k*n+i, r] for the output.
    
    # Let me build the decomposition by combining 12 Strassen decompositions for 2x2*2x2.
    
    # First, let me define the Strassen decomposition for a single 2x2*2x2 multiplication.
    # For A (2x2) with elements at positions (i1, j1) where i1 in {0,1}, j1 in {0,1}
    # B (2x2) with elements at positions (j1, k1) where j1 in {0,1}, k1 in {0,1}
    # C (2x2) with elements at positions (k1, i1) in the W convention... wait, W[k*n+i, r].
    
    # Let me be careful with the indexing.
    # A is 4x4, indexed by (i, j) where i in 0..3, j in 0..3.
    # B is 4x6, indexed by (j, k) where j in 0..3, k in 0..5.
    # C is 4x6, indexed by (i, k) where i in 0..3, k in 0..5.
    # W has shape (p*n, rank) = (24, rank), indexed by W[k*n+i, r] = W[k*4+i, r].
    
    # For the block decomposition:
    # A[i, j] where i = 2*i1 + i2, j = 2*j1 + j2, with i1, j1 in {0,1}, i2, j2 in {0,1}
    # B[j, k] where j = 2*j1 + j2, k = 2*k1 + k2, with j1 in {0,1}, k1 in {0,1,2}, j2, k2 in {0,1}
    # Wait, 6/2 = 3, so k1 in {0,1,2}, k2 in {0,1}.
    
    # C[i, k] where i = 2*i1 + i2, k = 2*k1 + k2
    # C[i1, k1] block = sum over j1 of A[i1, j1] * B[j1, k1]
    # where A[i1, j1] is 2x2, B[j1, k1] is 2x2.
    
    # So for each (i1, k1) in {0,1} x {0,1,2}, we have:
    # C[i1, k1] = A[i1, 0] * B[0, k1] + A[i1, 1] * B[1, k1]
    
    # That's 2 * 2 * 3 = 12 multiplications of 2x2 * 2x2.
    
    # Now, for each such multiplication, I'll use Strassen's algorithm.
    
    # Let me define the Strassen factors for a 2x2*2x2 multiplication.
    # For A_block (2x2) and B_block (2x2), the 7 multiplications are:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # And the output:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Now I need to map these to the global U, V, W matrices.
    
    # For each of the 12 block multiplications, I'll assign 7 columns in U, V, W.
    # Total rank = 12 * 7 = 84.
    
    rank = 84
    U = [[0] * rank for _ in range(16)]  # 4*4 = 16
    V = [[0] * rank for _ in range(24)]  # 4*6 = 24
    W = [[0] * rank for _ in range(24)]  # 6*4 = 24
    
    col = 0
    for i1 in range(2):
        for k1 in range(3):
            for j1 in range(2):
                # This is the multiplication A[i1, j1] * B[j1, k1]
                # A[i1, j1] is a 2x2 block at rows 2*i1, 2*i1+1 and cols 2*j1, 2*j1+1
                # B[j1, k1] is a 2x2 block at rows 2*j1, 2*j1+1 and cols 2*k1, 2*k1+1
                
                # The 7 Strassen multiplications:
                # M1 = (A[2*i1, 2*j1] + A[2*i1+1, 2*j1+1]) * (B[2*j1, 2*k1] + B[2*j1+1, 2*k1+1])
                # etc.
                
                # For M1:
                # U factors: A[2*i1, 2*j1] and A[2*i1+1, 2*j1+1]
                # V factors: B[2*j1, 2*k1] and B[2*j1+1, 2*k1+1]
                # W factors: C[2*i1, 2*k1] and C[2*i1+1, 2*k1+1] with appropriate signs
                
                # Let me define a helper to add a term.
                
                def add_term(u_rows, v_rows, w_rows, signs_u, signs_v, signs_w, col_idx):
                    for r, s in zip(u_rows, signs_u):
                        U[r][col_idx] += s
                    for r, s in zip(v_rows, signs_v):
                        V[r][col_idx] += s
                    for r, s in zip(w_rows, signs_w):
                        W[r][col_idx] += s
                
                # M1 = (A[i1*2, j1*2] + A[i1*2+1, j1*2+1]) * (B[j1*2, k1*2] + B[j1*2+1, k1*2+1])
                # = A[i1*2, j1*2]*B[j1*2, k1*2] + A[i1*2, j1*2]*B[j1*2+1, k1*2+1]
                #   + A[i1*2+1, j1*2+1]*B[j1*2, k1*2] + A[i1*2+1, j1*2+1]*B[j1*2+1, k1*2+1]
                # This contributes to:
                # C[i1*2, k1*2] += 1 (from A[i1*2,j1*2]*B[j1*2,k1*2])
                # C[i1*2, k1*2+1] += 1 (from A[i1*2,j1*2]*B[j1*2+1,k1*2+1])
                # C[i1*2+1, k1*2] += 1 (from A[i1*2+1,j1*2+1]*B[j1*2,k1*2])
                # C[i1*2+1, k1*2+1] += 1 (from A[i1*2+1,j1*2+1]*B[j1*2+1,k1*2+1])
                
                # But in Strassen, M1 is a single multiplication, so U, V, W each have one entry per factor.
                # U row for A[i1*2, j1*2] is i1*2*4 + j1*2 = i1*8 + j1*2
                # U row for A[i1*2+1, j1*2+1] is (i1*2+1)*4 + (j1*2+1) = i1*8 + j1*2 + 5
                
                # V row for B[j1*2, k1*2] is j1*2*6 + k1*2 = j1*12 + k1*2
                # V row for B[j1*2+1, k1*2+1] is (j1*2+1)*6 + (k1*2+1) = j1*12 + k1*2 + 7
                
                # W row for C[i1*2, k1*2] is k1*2*4 + i1*2 = k1*8 + i1*2
                # W row for C[i1*2+1, k1*2+1] is (k1*2+1)*4 + (i1*2+1) = k1*8 + i1*2 + 5
                
                # For M1, the U factor is the sum, so we put 1 in both U rows.
                # Similarly for V and W.
                
                add_term(
                    [i1*8 + j1*2, i1*8 + j1*2 + 5],
                    [j1*12 + k1*2, j1*12 + k1*2 + 7],
                    [k1*8 + i1*2, k1*8 + i1*2 + 5],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    col
                )
                col += 1
                
                # M2 = (A[i1*2+1, j1*2] + A[i1*2+1, j1*2+1]) * B[j1*2, k1*2]
                # U: A[i1*2+1, j1*2] and A[i1*2+1, j1*2+1]
                # V: B[j1*2, k1*2]
                # W: C[i1*2+1, k1*2] and C[i1*2+1, k1*2+1]... 
                # Wait, M2 = (c+d)*e where c=A[1,0], d=A[1,1], e=B[0,0]
                # This contributes to C[1,0] and C[1,1] in the 2x2 block.
                # C[1,0] += M2, C[1,1] += M2 (with appropriate signs in the final combination)
                
                # Actually, in Strassen:
                # C[1,0] = M2 + M4
                # C[1,1] = M1 - M2 + M3 + M6
                # So M2 contributes +1 to C[1,0] and -1 to C[1,1].
                
                add_term(
                    [i1*8 + j1*2 + 4, i1*8 + j1*2 + 5],  # A[i1*2+1, j1*2], A[i1*2+1, j1*2+1]
                    [j1*12 + k1*2],  # B[j1*2, k1*2]
                    [k1*8 + i1*2 + 4, k1*8 + i1*2 + 5],  # C[i1*2+1, k1*2], C[i1*2+1, k1*2+1]
                    [1, 1],
                    [1],
                    [1, -1],
                    col
                )
                col += 1
                
                # M3 = A[i1*2, j1*2] * (B[j1*2, k1*2+1] - B[j1*2+1, k1*2+1])
                add_term(
                    [i1*8 + j1*2],  # A[i1*2, j1*2]
                    [j1*12 + k1*2 + 1, j1*12 + k1*2 + 7],  # B[j1*2, k1*2+1], B[j1*2+1, k1*2+1]
                    [k1*8 + i1*2 + 1, k1*8 + i1*2 + 5],  # C[i1*2, k1*2+1], C[i1*2+1, k1*2+1]
                    [1],
                    [1, -1],
                    [1, 1],
                    col
                )
                col += 1
                
                # M4 = A[i1*2+1, j1*2+1] * (B[j1*2+1, k1*2] - B[j1*2, k1*2])
                add_term(
                    [i1*8 + j1*2 + 5],  # A[i1*2+1, j1*2+1]
                    [j1*12 + k1*2 + 6, j1*12 + k1*2],  # B[j1*2+1, k1*2], B[j1*2, k1*2]
                    [k1*8 + i1*2, k1*8 + i1*2 + 4],  # C[i1*2, k1*2], C[i1*2+1, k1*2]
                    [1],
                    [1, -1],
                    [1, 1],
                    col
                )
                col += 1
                
                # M5 = (A[i1*2, j1*2] + A[i1*2, j1*2+1]) * B[j1*2+1, k1*2+1]
                add_term(
                    [i1*8 + j1*2, i1*8 + j1*2 + 1],  # A[i1*2, j1*2], A[i1*2, j1*2+1]
                    [j1*12 + k1*2 + 7],  # B[j1*2+1, k1*2+1]
                    [k1*8 + i1*2 + 1, k1*8 + i1*2 + 5],  # C[i1*2, k1*2+1], C[i1*2+1, k1*2+1]
                    [1, 1],
                    [1],
                    [-1, 1],
                    col
                )
                col += 1
                
                # M6 = (A[i1*2+1, j1*2] - A[i1*2, j1*2]) * (B[j1*2, k1*2] + B[j1*2, k1*2+1])
                add_term(
                    [i1*8 + j1*2 + 4, i1*8 + j1*2],  # A[i1*2+1, j1*2], A[i1*2, j1*2]
                    [j1*12 + k1*2, j1*12 + k1*2 + 1],  # B[j1*2, k1*2], B[j1*2, k1*2+1]
                    [k1*8 + i1*2 + 5],  # C[i1*2+1, k1*2+1]
                    [1, -1],
                    [1, 1],
                    [1],
                    col
                )
                col += 1
                
                # M7 = (A[i1*2, j1*2+1] - A[i1*2+1, j1*2+1]) * (B[j1*2+1, k1*2] + B[j1*2+1, k1*2+1])
                add_term(
                    [i1*8 + j1*2 + 1, i1*8 + j1*2 + 5],  # A[i1*2, j1*2+1], A[i1*2+1, j1*2+1]
                    [j1*12 + k1*2 + 6, j1*12 + k1*2 + 7],  # B[j1*2+1, k1*2], B[j1*2+1, k1*2+1]
                    [k1*8 + i1*2, k1*8 + i1*2 + 5],  # C[i1*2, k1*2], C[i1*2+1, k1*2+1]
                    [1, -1],
                    [1, 1],
                    [1, 1],
                    col
                )
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
