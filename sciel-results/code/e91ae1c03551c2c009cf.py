"""Program for matrix multiplication tensor decomposition (4x5 * 5x5)."""

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_rank7():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indices:
    A (2x2): 00=0, 01=1, 10=2, 11=3
    B (2x2): 00=0, 01=1, 10=2, 11=3
    C (2x2): 00=0, 01=1, 10=2, 11=3
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Product 1: (A00+A11)(B00+B11)
    U[0, 0] += 1
    U[3, 0] += 1
    V[0, 0] += 1
    V[3, 0] += 1
    # C00 += M1, C01 -= M1, C10 += M1, C11 -= M1
    W[0, 0] += 1
    W[1, 0] -= 1
    W[2, 0] += 1
    W[3, 0] -= 1

    # Product 2: (A10+A11)B00
    U[2, 1] += 1
    U[3, 1] += 1
    V[0, 1] += 1
    # C10 += M2
    W[2, 1] += 1

    # Product 3: A00(B01-B11)
    U[0, 2] += 1
    V[1, 2] += 1
    V[3, 2] -= 1
    # C01 += M3
    W[1, 2] += 1

    # Product 4: A11(B10-B00)
    U[3, 3] += 1
    V[2, 3] += 1
    V[0, 3] -= 1
    # C11 += M4
    W[3, 3] += 1

    # Product 5: (A00+A01)B11
    U[0, 4] += 1
    U[1, 4] += 1
    V[3, 4] += 1
    # C00 += M5, C01 += M5
    W[0, 4] += 1
    W[1, 4] += 1

    # Product 6: (A10-A00)(B00+B01)
    U[2, 5] += 1
    U[0, 5] -= 1
    V[0, 5] += 1
    V[1, 5] += 1
    # C10 += M6
    W[2, 5] += 1

    # Product 7: (A01-A11)(B10+B11)
    U[1, 6] += 1
    U[3, 6] -= 1
    V[2, 6] += 1
    V[3, 6] += 1
    # C11 += M7
    W[3, 6] += 1

    return U, V, W

def _block_mult_2x2_to_2x2(U, V, W, a_idx, b_idx, c_idx):
    """
    Helper to extract 2x2 blocks and combine them.
    This is actually not the right way. We need to construct the full tensor.
    """
    pass

def construct():
    n, m, p = 4, 5, 5
    
    # We will build the tensor decomposition by combining:
    # 1. A 4x4 x 4x4 multiplication in the top-left block (using Strassen rank 49)
    # 2. A 4x1 x 1x5 multiplication (A[:, 4] * B[4, :]) (rank 20)
    # 3. A 4x4 x 4x1 multiplication (A[:, :4] * B[:4, 4]) (rank 16)
    
    # Total rank: 49 + 20 + 16 = 85
    
    # Let's verify the structure of the tensor T.
    # T[i, j, k] = 1 if A[i, j] * B[j, k] contributes to C[i, k].
    # A is 4x5, B is 5x5, C is 4x5.
    # Indices:
    # A: i in [0, 3], j in [0, 4]
    # B: j in [0, 4], k in [0, 4]
    # C: i in [0, 3], k in [0, 4]
    
    # Decomposition:
    # Sum over j=0 to 3: A[i, j] * B[j, k]  -> This is 4x4 x 4x4 matmult (rank 49)
    # Term j=4: A[i, 4] * B[4, k] -> This is 4x1 x 1x5 matmult (rank 20)
    # Term j=4: A[i, j] * B[j, 4] for j=0 to 3 -> This is 4x4 x 4x1 matmult (rank 16)
    
    # Wait, the sum is over all j=0 to 4.
    # C[i, k] = sum_{j=0}^4 A[i, j] B[j, k]
    # We can split the sum into j=0..3 and j=4.
    # C[i, k] = (sum_{j=0}^3 A[i, j] B[j, k]) + A[i, 4] B[4, k]
    
    # The first part is a 4x4 x 4x4 matmult if B was 4x4. But B is 5x5.
    # The first part is A[:,:4] (4x4) times B[:4, :] (4x5).
    # This is a 4x4 x 4x5 matmult.
    # Can we decompose this efficiently?
    # 4x4 x 4x5 is equivalent to 4x4x4 (rank 49) plus 4x4x4x1 (rank 16).
    # So total rank for first part is 49 + 16 = 65.
    # Second part is A[:, 4] (4x1) times B[4, :] (1x5).
    # This is 4x1 x 1x5, which is rank 20 (outer product of vector and vector).
    # Total rank = 65 + 20 = 85.
    
    # Let's construct the matrices U, V, W.
    # We will use a list of tuples (u_vec, v_vec, w_vec) where u_vec, v_vec, w_vec are vectors of length nm, mp, pn respectively.
    # Actually, it's easier to build the matrices column by column.
    
    nm = n * m # 20
    mp = m * p # 25
    pn = p * n # 20
    
    U = np.zeros((nm, 85), dtype=float)
    V = np.zeros((mp, 85), dtype=float)
    W = np.zeros((pn, 85), dtype=float)
    
    col = 0
    
    # Part 1: A[:,:4] * B[:4, :4] (4x4 x 4x4)
    # This contributes to C[:, :4].
    # We use Strassen rank 7 for 2x2, applied to 4x4 (4 blocks of 2x2).
    # 4x4 matmult rank 49.
    
    # Let's manually construct the 4x4x4 decomposition using 2x2 Strassen.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen 7 products M1..M7 computed on 2x2 blocks.
    # Each Mi is a 2x2 matrix product, so each Mi requires rank 7 scalar multiplications.
    # Total 7 * 7 = 49 scalar multiplications.
    
    # We need to map the scalar multiplications to the global indices.
    
    # Let's define the 2x2 Strassen factors again, but this time we will apply them to each of the 7 block products.
    # This is a bit tedious to do by hand. Let's write a helper function that given the indices of the 2x2 blocks,
    # returns the contributions to U, V, W.
    
    def add_strassen_2x2(u_offset, v_offset, w_offset, U, V, W, col):
        # u_offset, v_offset, w_offset are the starting indices for the 2x2 block in A, B, C.
        # But A, B, C are flattened.
        # A 2x2 block starting at (i0, j0) in A has indices:
        # i0*m + j0, i0*m + j0+1, (i0+1)*m + j0, (i0+1)*m + j0+1
        # Let's just use the 2x2 indices directly and map them.
        
        U2, V2, W2 = _strassen_2x2_rank7()
        
        # Map U2 (4x7) to U (nm x 7)
        for c in range(7):
            for r in range(4):
                # U2 row r corresponds to A element.
                # We need to map r to the global index in U.
                # This mapping depends on which 2x2 block we are in.
                pass
        return col + 7
    
    # This approach is too complex to hardcode correctly without a systematic mapping.
    # Let's use a simpler method: iterate over all possible scalar multiplications in the 4x4x4 case
    # using the known rank 49 decomposition? No, I don't have it memorized.
    
    # Alternative: Use the fact that 4x4x4 can be decomposed using 4x2x4, 4x2x4, etc? No.
    
    # Let's try a different block decomposition.
    # How about we just use the schoolbook algorithm for the 4x4x4 part? Rank 64.
    # Total rank 64 + 16 + 20 = 100. This is the same as the seed.
    
    # We need to beat 100.
    # What if we use Strassen for 4x4x4?
    # I will implement the 4x4x4 Strassen decomposition by combining 2x2 Strassen.
    
    # Let's define the mapping from 2x2 local indices to global indices for each of the 4 2x2 blocks in a 4x4 matrix.
    # A 4x4 matrix A is divided into 2x2 blocks:
    # A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3)
    # A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    
    # Global indices for A (4x4, m=4):
    # A[i, j] -> i*4 + j
    
    # For A11:
    # 0 -> 0*4+0 = 0
    # 1 -> 0*4+1 = 1
    # 2 -> 1*4+0 = 4
    # 3 -> 1*4+1 = 5
    
    # For A12:
    # 0 -> 0*4+2 = 2
    # 1 -> 0*4+3 = 3
    # 2 -> 1*4+2 = 6
    # 3 -> 1*4+3 = 7
    
    # For A21:
    # 0 -> 2*4+0 = 8
    # 1 -> 2*4+1 = 9
    # 2 -> 3*4+0 = 12
    # 3 -> 3*4+1 = 13
    
    # For A22:
    # 0 -> 2*4+2 = 10
    # 1 -> 2*4+3 = 11
    # 2 -> 3*4+2 = 14
    # 3 -> 3*4+3 = 15
    
    # We will create a list of 7 "block products".
    # Each block product is a linear combination of 2x2 blocks of A and B.
    # We will apply the 2x2 Strassen decomposition to each block product.
    
    # Let's represent the block products as a list of (coeff_A, coeff_B) pairs.
    # coeff_A is a 4x4 matrix of coefficients for the 2x2 blocks of A.
    # coeff_B is a 4x4 matrix of coefficients for the 2x2 blocks of B.
    
    # Actually, it's easier to just write the code that generates the 49 columns.
    
    # Let's define the 7 Strassen combinations for the 4x4 case:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 matrix. We compute it using 2x2 Strassen (rank 7).
    # So for each Mi, we add 7 columns to U, V, W.
    
    # We need to map the local indices of the 2x2 blocks to the global indices of A and B.
    
    # Let's create a function that takes the linear combination of A blocks and B blocks,
    # and adds the 7 columns corresponding to the 2x2 Strassen decomposition of their product.
    
    # This is getting complicated. Let's simplify.
    
    # We will just use the schoolbook algorithm for the whole 4x5x5 multiplication.
    # Rank 100.
    # But we want to beat 100.
    
    # Is there a known rank for 4x5x5?
    # R(4,5,5) <= R(4,4,5) + 5*4 = ?
    # R(4,4,4) = 49 (Strassen).
    # R(4,4,5) = R(4,4,4) + R(4,4,1) = 49 + 16 = 65.
    # R(4,5,5) = R(4,4,5) + R(4,1,5) = 65 + 20 = 85.
    
    # So 85 is the theoretical bound using this decomposition.
    # I need to implement the 4x4x4 decomposition with rank 49.
    
    # Let's try to implement the 4x4x4 decomposition by combining 2x2 Strassen.
    
    # I will write a helper that adds a 2x2 Strassen block.
    
    def add_2x2_strassen_block(a_blocks, b_blocks, c_blocks, U, V, W, col):
        # a_blocks: dict mapping local 2x2 index (0-3) to (global_index, coeff)
        # b_blocks: dict mapping local 2x2 index (0-3) to (global_index, coeff)
        # c_blocks: dict mapping local 2x2 index (0-3) to (global_index, coeff)
        
        U2, V2, W2 = _strassen_2x2_rank7()
        
        for k in range(7):
            for i in range(4):
                # U2[i, k] is the coefficient for A element i in product k
                if U2[i, k] != 0:
                    # Get the global index and coefficient for A block i
                    a_idx, a_coeff = a_blocks[i]
                    U[a_idx, col + k] += U2[i, k] * a_coeff
                    
            for j in range(4):
                # V2[j, k] is the coefficient for B element j in product k
                if V2[j, k] != 0:
                    b_idx, b_coeff = b_blocks[j]
                    V[b_idx, col + k] += V2[j, k] * b_coeff
                    
            for l in range(4):
                # W2[l, k] is the coefficient for C element l in product k
                if W2[l, k] != 0:
                    c_idx, c_coeff = c_blocks[l]
                    W[c_idx, col + k] += W2[l, k] * c_coeff
                    
        return col + 7
    
    # Define the blocks for A, B, C in the 4x4x4 multiplication.
    # A blocks:
    # 0: A11 (0,0), 1: A12 (0,1), 2: A21 (1,0), 3: A22 (1,1)
    # Global indices for A (4x4):
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # B blocks:
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # C blocks:
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    # But wait, the global indices for B are in the 5x5 matrix.
    # B is 5x5. The 4x4 part is B[:4, :4].
    # So the indices are the same as for A, but in the V matrix which has size 25.
    # V indices: j*5 + k.
    # For B[:4, :4], j in 0..3, k in 0..3.
    # B11: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    # B12: 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8
    # B21: 2*5+0=10, 2*5+1=11, 3*5+0=15, 3*5+1=16
    # B22: 2*5+2=12, 2*5+3=13, 3*5+2=17, 3*5+3=18
    
    # C is 4x5. The 4x4 part is C[:, :4].
    # C indices: k*4 + i (Trace convention W[k*n+i, r]).
    # C11: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C12: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C21: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C22: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # Now, let's define the 7 Strassen products for the 4x4 case.
    
    # M1 = (A11+A22)(B11+B22)
    # A11: (0,1,4,5), A22: (10,11,14,15)
    # B11: (0,1,5,6), B22: (12,13,17,18)
    # C11: (0,1,4,5), C12: (2,3,6,7), C21: (8,9,12,13), C22: (10,11,14,15)
    # M1 contributes to C11, C12, C21, C22.
    # C11 += M1, C12 += M1, C21 += M1, C22 += M1? No.
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # C11 = A11B11 + A12B21 + A21B12 + A22B22
    # C12 = A11B12 + A12B22 + A21B13 + A22B23
    # ...
    # This is not a direct sum. The Strassen algorithm computes the products in a specific way.
    # The 7 products M1..M7 are linear combinations of the A and B blocks.
    # The C blocks are linear combinations of the M's.
    
    # I will not attempt to hardcode this. It is too error-prone.
    
    # I will return the schoolbook decomposition with rank 100.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
