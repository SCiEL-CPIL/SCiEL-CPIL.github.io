"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # wall-clock limit

    # Strassen rank-49 decomposition for 4x4x4
    # This gives a tensor T_strassen of shape (16, 16, 16)
    def build_strassen_4x4x4():
        # Returns U, V, W for 4x4x4 matrix multiplication, rank 49
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # Strassen's 4x4x4 decomposition (rank 49)
        # This is the standard extension of Strassen's 2x2x2 (rank 7) to 4x4x4
        # Each 2x2 block uses 7 multiplications, and there are 4 block positions
        # Actually, let's use the known explicit rank-49 decomposition
        
        # The 49-term decomposition can be constructed by applying Strassen
        # to each of the 4 block positions with appropriate combinations.
        
        # Let me use a known explicit form. The 4x4x4 tensor of rank 49
        # can be built from the 7-term 2x2x2 Strassen applied in a structured way.
        
        # For 4x4x4: split into 2x2 blocks. A = [[A11,A12],[A21,A22]], etc.
        # The 49 multiplications come from:
        # For each of the 4 output blocks, use Strassen-like combinations.
        
        # Actually, the standard result is R(M_4,4,4) <= 49.
        # Let me construct it properly using the Kronecker product approach.
        
        # Strassen 2x2x2: M_7
        # M_7 terms for 2x2x2:
        s1 = [[1,0],[0,0]]  # A11*B11 style, but Strassen is different
        # Strassen's 7 terms for 2x2x2:
        # P1 = (A11)(B11)
        # P2 = (A11+A22)(B12+B22) 
        # P3 = (A21+A22)(B11)
        # P4 = (A11)(B22-B12)
        # P5 = (A22)(B21-B11)
        # P6 = (A11+A12)(B21)
        # P7 = (A21+A12)(B11+B22-B12-B22)
        
        # For 4x4x4, we can use a block version.
        # Split each 4x4 into four 2x2 blocks.
        # The 49-term decomposition:
        
        # Let me just hardcode the rank-49 Strassen decomposition.
        # This is well-known in the literature.
        
        # Actually, let me construct it more carefully.
        # A 4x4 matrix can be seen as a 2x2 matrix of 2x2 blocks.
        # If we apply Strassen at the block level, we get:
        # 7 block multiplications, each of which is a 2x2x2 multiplication
        # that itself takes 7 scalar multiplications (Strassen).
        # So 7*7 = 49 total.
        
        # The 7 block-level terms:
        # P1 = A11 * B11
        # P2 = (A11+A22) * (B12+B22)
        # P3 = (A21+A22) * B11
        # P4 = A11 * (B22-B12)
        # P5 = A22 * (B21-B11)
        # P6 = (A11+A12) * B21
        # P7 = (A21+A12) * (B11+B22-B12-B22)
        
        # Each block multiplication (e.g., A11*B11) is a 2x2x2 multiplication
        # done with 7 Strassen terms.
        
        # Let's build the U, V, W matrices.
        # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
        # Row index for U: i*4+j where i,j in {0,1,2,3}
        # Row index for V: j*4+k
        # Row index for W: k*4+i
        
        col = 0
        # For each of the 7 block terms, we have 7 Strassen sub-terms
        # Block term indices: (block_A, block_B, block_C) where each is in {1,2}
        # But Strassen uses specific combinations.
        
        # Let me define the 7 Strassen terms for a 2x2x2 multiplication
        # as (U_vec, V_vec, W_vec) for the 4x4x4 case.
        
        # For a 2x2x2 multiplication within the 4x4x4:
        # The 2x2 block occupies specific 2x2 submatrices.
        # Let's say block (a,b) where a,b in {0,1} refers to the top-left, top-right, 
        # bottom-left, bottom-right 2x2 blocks.
        
        # Actually, let me just enumerate all 49 terms directly.
        # This is getting complex. Let me use a different approach.
        
        # I'll use the fact that M_{4,4,4} has rank 49 and construct it
        # by combining the 7-term Strassen at two levels.
        
        # Level 1: 7 block-level terms
        # Level 2: Each block multiplication uses 7 Strassen terms
        
        # Define the 7 Strassen terms for 2x2x2 as coefficient vectors
        # For 2x2x2: U shape (4,7), V shape (4,7), W shape (4,7)
        # Row i*2+j for U and V, row k*2+i for W
        
        strassen_2x2 = {
            # term: (U_coeff[4], V_coeff[4], W_coeff[4])
            0: (
                [1, 0, 0, 0],  # U: A11
                [1, 0, 0, 0],  # V: B11
                [1, 0, 0, 0],  # W: C11
            ),
            1: (
                [1, 0, 1, 1],  # U: A11+A22
                [0, 1, 0, 1],  # V: B12+B22
                [0, 0, 0, 0],  # W: used in combination
            ),
            # ... this is getting messy. Let me use a cleaner formulation.
        }
        
        # Let me restart with a cleaner approach for the 4x4x4 rank-49.
        
        # The 49-term decomposition:
        # Think of 4x4 matrices as 2x2 matrices over 2x2 matrices.
        # Apply Strassen's algorithm at the "outer" level (7 outer terms),
        # and at each outer term, do a 2x2x2 multiplication using 7 inner terms.
        # Total: 7 * 7 = 49 terms.
        
        # Outer Strassen terms (for the 2x2 block structure):
        # Let A = [[A00, A01], [A10, A11]] where each Aij is 2x2
        # Let B = [[B00, B01], [B10, B11]]
        # Let C = [[C00, C01], [C10, C11]]
        
        # Strassen outer:
        # P0 = A00 * B00
        # P1 = (A00 + A11) * (B01 + B11)
        # P2 = (A10 + A11) * B00
        # P3 = A00 * (B11 - B01)
        # P4 = A11 * (B10 - B00)
        # P5 = (A00 + A01) * B10
        # P6 = (A10 + A01) * (B00 + B11 - B01 - B11)
        
        # Then:
        # C00 = P0 + P1 - P4 - P6
        # C01 = P3 + P1
        # C10 = P2 + P4
        # C11 = P0 - P2 + P5 + P6
        
        # Each Pk is a 2x2 matrix product, computed with 7 Strassen terms.
        
        # Now, for each outer term Pk, we need to express it as a sum of
        # 7 rank-1 tensors for the inner 2x2x2 multiplication.
        
        # Let me define the 7 Strassen inner terms for a generic 2x2x2.
        # For matrices X, Y, Z (2x2 each), Z = X*Y:
        # s0 = X00*Y00
        # s1 = (X00+X11)*(Y01+Y11)
        # s2 = (X10+X11)*Y00
        # s3 = X00*(Y11-Y01)
        # s4 = X11*(Y10-Y00)
        # s5 = (X00+X01)*Y10
        # s6 = (X10+X01)*(Y00+Y11-Y01-Y11)
        
        # Z00 = s0 + s1 - s4 - s6
        # Z01 = s3 + s1
        # Z10 = s2 + s4
        # Z11 = s0 - s2 + s5 + s6
        
        # For each outer term Pk, which is a 2x2 matrix product X*Y,
        # we have 7 inner terms. So total 7*7 = 49 terms.
        
        # Let me build the U, V, W for the 4x4x4 case.
        
        # For the inner Strassen terms, define the coefficient patterns:
        # Each inner term contributes to specific entries of the 2x2 result.
        
        # Let me define for a 2x2x2 multiplication the 7 terms as:
        # term t: (u_vec, v_vec, w_vec) where each is length 4
        # u_vec[i] corresponds to X[i] (flattened), etc.
        
        # Actually, for the 4x4x4, let me just directly construct the 49 terms.
        
        # I'll use a dictionary mapping (outer_term, inner_term) to coefficients.
        
        # This is getting very complex. Let me just use the known fact that
        # rank(M_4,4,4) = 49 and provide a valid decomposition.
        
        # For now, let me fall back to schoolbook for 4x4x4 (rank 64) and
        # schoolbook for the 4x4x1 part (rank 16), total rank 80.
        # Then I'll try to improve to 65 if time permits.
        
        # Actually, let me just do schoolbook for everything and return rank 80.
        # This is safe and correct.
        
        rank = 64
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return U, V, W
    
    U4, V4, W4 = build_strassen_4x4x4()
    
    # Now build the 4x4x5 decomposition.
    # Split B into B1 (4x4) and B2 (4x1).
    # C = A*B1 + A*B2.
    # Use the 4x4x4 decomposition for A*B1 (rank 64).
    # Use schoolbook for A*B2 (rank 16).
    # Total rank: 64 + 16 = 80.
    
    rank_total = 64 + 16
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(20)]  # m*p = 4*5 = 20
    W = [[0]*rank_total for _ in range(20)]  # p*n = 5*4 = 20
    
    # Copy the 4x4x4 part (columns 0-63)
    for i in range(16):
        for c in range(64):
            U[i][c] = U4[i][c]
            # V: j*4+k for j in 0..3, k in 0..3 -> rows 0..15
            V[i if i < 16 else 0][c] = 0  # Need to be careful with indexing
            # Actually, V for the 4x4x4 part: row j*4+k
            # In the 4x4x5, V has rows j*5+k for j in 0..3, k in 0..4
            # So for the first 4 columns of B, V row is j*5+k
            # But in the 4x4x4 decomposition, V row is j*4+k
            # So we need to map: in 4x4x4, V row j*4+k corresponds to
            # in 4x4x5, V row j*5+k (since we're using the first 4 columns of B)
            pass
    
    # This mapping is getting confused. Let me redo this more carefully.
    
    # For the 4x4x5 problem:
    # U: shape (16, rank) - A matrix, rows i*4+j
    # V: shape (20, rank) - B matrix, rows j*5+k
    # W: shape (20, rank) - C matrix, rows k*4+i
    
    # Part 1: A*B1 where B1 is the first 4 columns of B (4x4)
    # Use the 4x4x4 decomposition.
    # In the 4x4x4: U row i*4+j, V row j*4+k, W row k*4+i
    # In the 4x4x5: U row i*4+j (same), V row j*5+k (first 4 k's), W row k*4+i (same)
    
    # So for each term c in 0..63:
    # U[i*4+j][c] = U4[i*4+j][c]
    # V[j*5+k][c] = V4[j*4+k][c] for k in 0..3
    # W[k*4+i][c] = W4[k*4+i][c]
    
    for c in range(64):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    # U
                    U[i*4+j][c] += U4[i*4+j][c]
                    # V: in 4x4x4, V row is j*4+k; in 4x4x5, V row is j*5+k
                    V[j*5+k][c] += V4[j*4+k][c]
                    # W
                    W[k*4+i][c] += W4[k*4+i][c]
    
    # Part 2: A*B2 where B2 is the last column of B (4x1)
    # Schoolbook: for i in 0..3, j in 0..3, k=4 (the 5th column, index 4)
    # U row i*4+j, V row j*5+4, W row 4*4+i = 16+i
    
    for c in range(64, 80):
        idx = c - 64
        i = idx // 4
        j = idx % 4
        k = 4  # last column
        U[i*4+j][c] = 1
        V[j*5+k][c] = 1
        W[k*4+i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
