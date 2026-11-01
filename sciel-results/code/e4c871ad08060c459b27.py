"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    We use a block decomposition into six 2x2x2 matrix multiplications.
    Each 2x2x2 multiplication uses the Strassen algorithm (rank 7).
    Total rank = 6 * 7 = 42.
    """
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # We decompose the problem into 6 sub-problems of size 2x2x2.
    # A is 2x4, split into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # B is 4x6, split into B1 (2x2, rows 0-1, cols 0-1), B2 (2x2, rows 0-1, cols 2-3), B3 (2x2, rows 0-1, cols 4-5),
    #                 B4 (2x2, rows 2-3, cols 0-1), B5 (2x2, rows 2-3, cols 2-3), B6 (2x2, rows 2-3, cols 4-5).
    # C is 2x6, split into C1 (2x2, cols 0-1), C2 (2x2, cols 2-3), C3 (2x2, cols 4-5).
    # C1 = A1*B1 + A2*B4
    # C2 = A1*B2 + A2*B5
    # C3 = A1*B3 + A2*B6
    
    # Each term like A1*B1 is a 2x2x2 multiplication.
    # We use Strassen for each.
    # Strassen computes C = A*B using 7 products:
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to construct the global U, V, W matrices.
    # Rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a Strassen decomposition for a specific 2x2x2 block
    # i_offset: row offset in A (0 or 1 for the 2 rows of the sub-block? No, A is 2x4, so rows are 0,1)
    # Actually, A1 is rows 0,1 cols 0,1. A2 is rows 0,1 cols 2,3.
    # So for A1, i is 0,1. j is 0,1.
    # For A2, i is 0,1. j is 2,3.
    
    # B1 is rows 0,1 cols 0,1.
    # B4 is rows 2,3 cols 0,1.
    
    # Let's define the blocks:
    # Block 1: A1 (i:0-1, j:0-1) * B1 (j:0-1, k:0-1) -> C1 (i:0-1, k:0-1)
    # Block 2: A2 (i:0-1, j:2-3) * B4 (j:2-3, k:0-1) -> C1 (i:0-1, k:0-1)
    # Block 3: A1 (i:0-1, j:0-1) * B2 (j:0-1, k:2-3) -> C2 (i:0-1, k:2-3)
    # Block 4: A2 (i:0-1, j:2-3) * B5 (j:2-3, k:2-3) -> C2 (i:0-1, k:2-3)
    # Block 5: A1 (i:0-1, j:0-1) * B3 (j:0-1, k:4-5) -> C3 (i:0-1, k:4-5)
    # Block 6: A2 (i:0-1, j:2-3) * B6 (j:2-3, k:4-5) -> C3 (i:0-1, k:4-5)
    
    # Strassen coefficients for 2x2x2:
    # U indices: i*2 + j_local
    # V indices: j_local*2 + k_local
    # W indices: k_local*2 + i_local
    
    # We map local indices to global indices.
    
    def add_strassen(i_off, j_off, k_off, col_start):
        # i_off, j_off, k_off are offsets for the 2x2 block
        # i is 0,1 relative to i_off
        # j is 0,1 relative to j_off
        # k is 0,1 relative to k_off
        
        # Strassen products P1..P7
        # P1 = (A11 + A22) * (B11 + B22)
        # A11 is (i_off, j_off), A22 is (i_off+1, j_off+1)
        # B11 is (j_off, k_off), B22 is (j_off+1, k_off+1)
        
        # We need to add terms to U, V, W.
        # For each product, we add a rank-1 term.
        
        # P1
        # U: A11 + A22. So U[i_off, col] += 1, U[i_off+1, col] += 1? No.
        # The term is (A11 + A22) * (B11 + B22) * (C11 + C22 - C12 - C21)? No.
        # The decomposition is sum of 7 rank-1 tensors.
        # Tensor r: U_r, V_r, W_r.
        # For P1:
        # U_r has 1s at (i_off, j_off) and (i_off+1, j_off+1).
        # V_r has 1s at (j_off, k_off) and (j_off+1, k_off+1).
        # W_r has 1s at (k_off, i_off) and (k_off+1, i_off+1).
        # Wait, W index is (k, i).
        # W[k*n + i].
        
        # Let's list the 7 terms for Strassen:
        # 1. (A11+A22)(B11+B22) -> C11 += 1, C12 -= 1, C21 -= 1, C22 += 1?
        #    Actually:
        #    C11 = P1 + P4 - P5 + P7
        #    C12 = P3 + P5
        #    C21 = P2 + P4
        #    C22 = P1 - P2 + P3 + P6
        
        # Term 1: P1
        # U: 1 at (i_off, j_off), 1 at (i_off+1, j_off+1)
        # V: 1 at (j_off, k_off), 1 at (j_off+1, k_off+1)
        # W: 1 at (k_off, i_off), -1 at (k_off+1, i_off), -1 at (k_off, i_off+1), 1 at (k_off+1, i_off+1)
        # Wait, W coefficients must match the C updates.
        # C11 gets +1. C11 is at (i_off, k_off). So W[k_off*n + i_off] += 1.
        # C12 gets -1. C12 is at (i_off, k_off+1). So W[(k_off+1)*n + i_off] -= 1.
        # C21 gets -1. C21 is at (i_off+1, k_off). So W[k_off*n + i_off+1] -= 1.
        # C22 gets +1. C22 is at (i_off+1, k_off+1). So W[(k_off+1)*n + i_off+1] += 1.
        
        # Let's implement this logic for all 7 terms.
        
        def add_term(u_indices, v_indices, w_indices, w_coeffs, col):
            for idx in u_indices:
                U[idx, col] = 1
            for idx in v_indices:
                V[idx, col] = 1
            for idx, coeff in zip(w_indices, w_coeffs):
                W[idx, col] = coeff
                
        # Global indices for U: i*m + j
        # Global indices for V: j*p + k
        # Global indices for W: k*n + i
        
        # Term 1: P1 = (A11+A22)(B11+B22)
        u1 = [i_off*m + j_off, (i_off+1)*m + (j_off+1)]
        v1 = [j_off*p + k_off, (j_off+1)*p + (k_off+1)]
        w1 = [k_off*n + i_off, (k_off+1)*n + i_off, k_off*n + (i_off+1), (k_off+1)*n + (i_off+1)]
        w1_coeffs = [1, -1, -1, 1]
        add_term(u1, v1, w1, w1_coeffs, col_start)
        
        # Term 2: P2 = (A21+A22)B11
        # A21 is (i_off+1, j_off), A22 is (i_off+1, j_off+1)
        # B11 is (j_off, k_off)
        # C21 += 1, C22 -= 1
        u2 = [(i_off+1)*m + j_off, (i_off+1)*m + (j_off+1)]
        v2 = [j_off*p + k_off]
        w2 = [k_off*n + (i_off+1), (k_off+1)*n + (i_off+1)]
        w2_coeffs = [1, -1]
        add_term(u2, v2, w2, w2_coeffs, col_start + 1)
        
        # Term 3: P3 = A11(B12-B22)
        # A11 is (i_off, j_off)
        # B12 is (j_off, k_off+1), B22 is (j_off+1, k_off+1)
        # C12 += 1, C22 += 1
        u3 = [i_off*m + j_off]
        v3 = [j_off*p + (k_off+1), (j_off+1)*p + (k_off+1)]
        w3 = [(k_off+1)*n + i_off, (k_off+1)*n + (i_off+1)]
        w3_coeffs = [1, 1]
        add_term(u3, v3, w3, w3_coeffs, col_start + 2)
        
        # Term 4: P4 = A22(B21-B11)
        # A22 is (i_off+1, j_off+1)
        # B21 is (j_off+1, k_off), B11 is (j_off, k_off)
        # C11 += 1, C21 += 1
        u4 = [(i_off+1)*m + (j_off+1)]
        v4 = [(j_off+1)*p + k_off, j_off*p + k_off]
        w4 = [k_off*n + i_off, k_off*n + (i_off+1)]
        w4_coeffs = [1, 1]
        add_term(u4, v4, w4, w4_coeffs, col_start + 3)
        
        # Term 5: P5 = (A11+A12)B22
        # A11 is (i_off, j_off), A12 is (i_off, j_off+1)
        # B22 is (j_off+1, k_off+1)
        # C11 -= 1, C12 += 1
        u5 = [i_off*m + j_off, i_off*m + (j_off+1)]
        v5 = [(j_off+1)*p + (k_off+1)]
        w5 = [k_off*n + i_off, (k_off+1)*n + i_off]
        w5_coeffs = [-1, 1]
        add_term(u5, v5, w5, w5_coeffs, col_start + 4)
        
        # Term 6: P6 = (A21-A11)(B11+B12)
        # A21 is (i_off+1, j_off), A11 is (i_off, j_off)
        # B11 is (j_off, k_off), B12 is (j_off, k_off+1)
        # C22 += 1
        # Wait, C22 = P1 - P2 + P3 + P6.
        # So P6 contributes +1 to C22.
        # Does it contribute to others? No.
        u6 = [(i_off+1)*m + j_off, i_off*m + j_off]
        v6 = [j_off*p + k_off, j_off*p + (k_off+1)]
        w6 = [(k_off+1)*n + (i_off+1)]
        w6_coeffs = [1]
        add_term(u6, v6, w6, w6_coeffs, col_start + 5)
        
        # Term 7: P7 = (A12-A22)(B21+B22)
        # A12 is (i_off, j_off+1), A22 is (i_off+1, j_off+1)
        # B21 is (j_off+1, k_off), B22 is (j_off+1, k_off+1)
        # C11 += 1
        u7 = [i_off*m + (j_off+1), (i_off+1)*m + (j_off+1)]
        v7 = [(j_off+1)*p + k_off, (j_off+1)*p + (k_off+1)]
        w7 = [k_off*n + i_off]
        w7_coeffs = [1]
        add_term(u7, v7, w7, w7_coeffs, col_start + 6)
        
    # Block 1: A1 (i:0-1, j:0-1) * B1 (j:0-1, k:0-1)
    # i_off=0, j_off=0, k_off=0
    add_strassen(0, 0, 0, 0)
    
    # Block 2: A2 (i:0-1, j:2-3) * B4 (j:2-3, k:0-1)
    # i_off=0, j_off=2, k_off=0
    add_strassen(0, 2, 0, 7)
    
    # Block 3: A1 (i:0-1, j:0-1) * B2 (j:0-1, k:2-3)
    # i_off=0, j_off=0, k_off=2
    add_strassen(0, 0, 2, 14)
    
    # Block 4: A2 (i:0-1, j:2-3) * B5 (j:2-3, k:2-3)
    # i_off=0, j_off=2, k_off=2
    add_strassen(0, 2, 2, 21)
    
    # Block 5: A1 (i:0-1, j:0-1) * B3 (j:0-1, k:4-5)
    # i_off=0, j_off=0, k_off=4
    add_strassen(0, 0, 4, 28)
    
    # Block 6: A2 (i:0-1, j:2-3) * B6 (j:2-3, k:4-5)
    # i_off=0, j_off=2, k_off=4
    add_strassen(0, 2, 4, 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
