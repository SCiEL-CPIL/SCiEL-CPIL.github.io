<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_DEADLINE = time.time() + 120.0

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen rank-7 decomposition for 2x2x2 blocks, combined with schoolbook for the rest.
    # Total rank = 49 (for 4x4x4) + 16 (for 4x4x5) = 65
    rank = 65
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen decomposition for 4x4x4
    # Split A (4x4) into 2x2 blocks of 2x2: A00, A01, A10, A11
    # Split B (4x4) into 2x2 blocks of 2x2: B00, B01, B10, B11
    # C = A*B, C is 4x4, split into 2x2 blocks of 2x2: C00, C01, C10, C11
    # For each 2x2 block multiplication, use Strassen rank-7
    
    # Strassen factors for 2x2x2:
    # U factors (input A side):
    # U0 = A00 + A11
    # U1 = A10 + A11
    # U2 = A00
    # U3 = A11
    # U4 = A00 + A01
    # U5 = A10 - A00
    # U6 = A01 - A11
    # V factors (input B side):
    # V0 = B00 + B11
    # V1 = B00
    # V2 = B01 - B11
    # V3 = B10 - B00
    # V4 = B11
    # V5 = B00 + B01
    # V6 = B10 + B11
    # W factors (output C side):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we have 4 block multiplications (each 2x2x2), each with rank 7, total 28
    # But we can share some factors? Actually, for 4x4, the blocks are independent, so rank = 4 * 7 = 28?
    # Wait, no. The 4x4 matrix multiplication can be done with 4 independent 2x2 multiplications?
    # No, that's not right. The 4x4 multiplication is not just 4 independent 2x2 multiplications.
    # Actually, if we partition A into 2x2 blocks of size 2x2, and B into 2x2 blocks of size 2x2,
    # then C[i,j] = sum_k A[i,k]B[k,j] for each 2x2 block.
    # So C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each C block is a sum of two 2x2 multiplications. So we need to compute 8 2x2 multiplications,
    # each with rank 7, total rank = 8 * 7 = 56? That's worse than Strassen 4x4 which is 49.
    
    # Let me just use the known Strassen 4x4 rank-49 decomposition.
    # Actually, the rank of 4x4x4 matrix multiplication is at most 49 (by applying Strassen to 2x2 blocks recursively).
    # But it's known that the rank is between 36 and 49. Let's use 49.
    
    # For simplicity, let's use the schoolbook for the 4x4x4 part (rank 64) and schoolbook for 4x4x5 (rank 20), total 84.
    # That's still better than 144.
    
    # Actually, let's just do schoolbook for all, but that's 144.
    # Let me try a different approach: use FFT or Winograd for the 9x9 part? No, that's not applicable.
    
    # Let's just use the Kronecker product idea from the directive.
    # But I don't know the exact factors. Let me just use schoolbook with a smaller rank.
    
    # Actually, the minimal rank for 4x4x9 is unknown, but schoolbook is 144.
    # Let me try to use the fact that 9 = 4 + 5, and use Strassen for 4x4x4 (rank 49) and schoolbook for 4x4x5 (rank 20), total 69.
    
    # But I need the actual factors. Let me just implement Strassen for 2x2x2 and combine.
    
    # For now, let's just return schoolbook to be safe.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_DEADLINE = time.time() + 120.0

def _strassen_2x2x2():
    """Return U, V, W factors for 2x2x2 Strassen decomposition.
    A is 2x2 indexed as [A00, A01, A10, A11] (row-major: 0=A00, 1=A01, 2=A10, 3=A11)
    B is 2x2 indexed as [B00, B01, B10, B11] (row-major: 0=B00, 1=B01, 2=B10, 3=B11)
    C is 2x2 indexed as [C00, C01, C10, C11] (row-major: 0=C00, 1=C01, 2=C10, 3=C11)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B00)
    # M3 = A11(B01-B11)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 - M1 + M3 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M4 - M6
    
    # U columns (A side):
    # Col 0: A00 + A11
    U[0, 0] = 1; U[3, 0] = 1
    # Col 1: A10 + A11
    U[2, 1] = 1; U[3, 1] = 1
    # Col 2: A00
    U[0, 2] = 1
    # Col 3: A11
    U[3, 3] = 1
    # Col 4: A00 + A01
    U[0, 4] = 1; U[1, 4] = 1
    # Col 5: A10 - A00
    U[2, 5] = 1; U[0, 5] = -1
    # Col 6: A01 - A11
    U[1, 6] = 1; U[3, 6] = -1
    
    # V columns (B side):
    # Col 0: B00 + B11
    V[0, 0] = 1; V[3, 0] = 1
    # Col 1: B00
    V[0, 1] = 1
    # Col 2: B10 - B00
    V[2, 2] = 1; V[0, 2] = -1
    # Col 3: B01 - B11
    V[1, 3] = 1; V[3, 3] = -1
    # Col 4: B11
    V[3, 4] = 1
    # Col 5: B00 + B01
    V[0, 5] = 1; V[1, 5] = 1
    # Col 6: B10 + B11
    V[2, 6] = 1; V[3, 6] = 1
    
    # W columns (C side):
    # Col 0: C00 - C10 + C11 + C01? Let me recheck.
    # C00 = M0 - M1 + M3 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M4 - M6
    # So W[r, c] gives coefficient of M_c in C_r
    # C00: M0:1, M1:-1, M2:0, M3:1, M4:0, M5:1, M6:0
    W[0, 0] = 1; W[0, 1] = -1; W[0, 3] = 1; W[0, 5] = 1
    # C01: M0:0, M1:0, M2:1, M3:0, M4:1, M5:0, M6:0
    W[1, 2] = 1; W[1, 4] = 1
    # C10: M0:0, M1:1, M2:0, M3:1, M4:0, M5:0, M6:0
    W[2, 1] = 1; W[2, 3] = 1
    # C11: M0:1, M1:0, M2:-1, M3:0, M4:1, M5:0, M6:-1
    W[3, 0] = 1; W[3, 2] = -1; W[3, 4] = 1; W[3, 6] = -1
    
    return U, V, W

def _winograd_2x2x3():
    """Return U, V, W factors for 2x2 x 2x3 multiplication.
    A is 2x2 indexed as [A00, A01, A10, A11] (row-major)
    B is 2x3 indexed as [B00, B01, B02, B10, B11, B12] (row-major: 0=B00,1=B01,2=B02,3=B10,4=B11,5=B12)
    C is 2x3 indexed as [C00, C01, C02, C10, C11, C12] (row-major)
    Uses Strassen for first 2 columns (rank 7) + schoolbook for 3rd column (rank 2).
    Total rank = 9.
    """
    rank = 9
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # First 7 columns: Strassen for 2x2 x 2x2 (columns 0,1 of B)
    U_s, V_s, W_s = _strassen_2x2x2()
    # V_s is 4x7 for B[0:4] which corresponds to B00,B01,B10,B11 (indices 0,1,3,4 in 2x3)
    # In 2x3, B indices: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
    # V_s maps to B indices [0, 1, 3, 4] (B00, B01, B10, B11)
    U[:, :7] = U_s
    V[0, :7] = V_s[0, :]  # B00
    V[1, :7] = V_s[1, :]  # B01
    V[3, :7] = V_s[2, :]  # B10
    V[4, :7] = V_s[3, :]  # B11
    # W_s is 4x7 for C indices [0,1,2,3] = C00,C01,C10,C11
    # In 2x3, C indices: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
    W[0, :7] = W_s[0, :]  # C00
    W[1, :7] = W_s[1, :]  # C01
    W[3, :7] = W_s[2, :]  # C10
    W[4, :7] = W_s[3, :]  # C11
    
    # Last 2 columns: schoolbook for 3rd column (column 2 of B, index 2=B02, 5=B12)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # Col 7: A00 * B02 -> C02
    U[0, 7] = 1
    V[2, 7] = 1  # B02
    W[2, 7] = 1  # C02
    # Col 8: A01 * B12 -> C02
    U[1, 8] = 1
    V[5, 8] = 1  # B12
    W[2, 8] = 1  # C02
    # Wait, C12 also needs terms. Let me redo.
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # That's 4 terms, need 4 columns for schoolbook of 2x2 x 2x1.
    # Actually 2x2 x 2x1 has rank 4 (schoolbook) or can we do better?
    # For 2x2 x 2x1: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # This is rank 4 in schoolbook. Can we do rank 3? Probably not easily.
    # Let me use 4 columns for this part.
    
    # Actually, let me reconsider. Let me use rank 4 for the 2x2 x 2x1 part.
    rank = 11  # 7 + 4
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    U[:, :7] = U_s
    V[0, :7] = V_s[0, :]
    V[1, :7] = V_s[1, :]
    V[3, :7] = V_s[2, :]
    V[4, :7] = V_s[3, :]
    W[0, :7] = W_s[0, :]
    W[1, :7] = W_s[1, :]
    W[3, :7] = W_s[2, :]
    W[4, :7] = W_s[3, :]
    
    # Col 7: A00 * B02 -> C02
    U[0, 7] = 1
    V[2, 7] = 1
    W[2, 7] = 1
    # Col 8: A01 * B12 -> C02
    U[1, 8] = 1
    V[5, 8] = 1
    W[2, 8] = 1
    # Col 9: A10 * B02 -> C12
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    # Col 10: A11 * B12 -> C12
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Partition:
    # A (4x4) -> 2x2 blocks of 2x2: A00,A01,A10,A11
    # B (4x9) -> 2x3 blocks of 2x3: B00,B01,B02 (top), B10,B11,B12 (bottom)
    # C (4x9) -> 2x3 blocks of 2x3: C00,C01,C02 (top), C10,C11,C12 (bottom)
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    #
    # Each Ci,j is sum of two 2x2 x 2x3 products.
    # Use Winograd 2x2x3 (rank 11) for each.
    # Total: 6 C-blocks * 2 sub-products * 11 = 132. Still better than 144.
    
    # Actually, let me think if I can share computations.
    # The 12 sub-products are:
    # A00*B00, A00*B01, A00*B02, A01*B10, A01*B11, A01*B12,
    # A10*B00, A10*B01, A10*B02, A11*B10, A11*B11, A11*B12
    # All distinct, so no sharing possible without more complex decomposition.
    
    # Total rank = 12 * 11 = 132
    
    rank = 132
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Get the 2x2x3 decomposition
    Uw, Vw, Ww = _winograd_2x2x3()
    
    # A indices (4x4, row-major): 0=A00,1=A01,2=A10,3=A11,4=A02,5=A03,6=A12,7=A13,8=A20,9=A21,10=A22,11=A23,12=A30,13=A31,14=A32,15=A33
    # Wait, A is 4x4. Row-major: A[i,j] at index i*4+j.
    # A00 block: rows 0-1, cols 0-1: indices 0,1,4,5
    # A01 block: rows 0-1, cols 2-3: indices 2,3,6,7
    # A10 block: rows 2-3, cols 0-1: indices 8,9,12,13
    # A11 block: rows 2-3, cols 2-3: indices 10,11,14,15
    
    # B indices (4x9, row-major): B[i,j] at index i*9+j.
    # B00 block: rows 0-1, cols 0-2: indices 0,1,2,9,10,11
    # B01 block: rows 0-1, cols 3-5: indices 3,4,5,12,13,14
    # B02 block: rows 0-1, cols 6-8: indices 6,7,8,15,16,17
    # B10 block: rows 2-3, cols 0-2: indices 18,19,20,27,28,29
    # B11 block: rows 2-3, cols 3-5: indices 21,22,23,30,31,32
    # B12 block: rows 2-3, cols 6-8: indices 24,25,26,33,34,35
    
    # C indices (4x9, row-major): C[i,j] at index i*9+j.
    # C00 block: rows 0-1, cols 0-2: indices 0,1,2,9,10,11
    # C01 block: rows 0-1, cols 3-5: indices 3,4,5,12,13,14
    # C02 block: rows 0-1, cols 6-8: indices 6,7,8,15,16,17
    # C10 block: rows 2-3, cols 0-2: indices 18,19,20,27,28,29
    # C11 block: rows 2-3, cols 3-5: indices 21,22,23,30,31,32
    # C12 block: rows 2-3, cols 6-8: indices 24,25,26,33,34,35
    
    # For each sub-product, map local indices to global indices.
    # A local: 0=A00,1=A01,2=A10,3=A11 (within 2x2 block)
    # B local: 0=B00,1=B01,2=B02,3=B10,4=B11,5=B12 (within 2x3 block)
    # C local: 0=C00,1=C01,2=C02,3=C10,4=C11,5=C12 (within 2x3 block)
    
    # Define the 12 sub-products:
    # (A_block, B_block, C_block)
    # A_block: 0=A00, 1=A01, 2=A10, 3=A11
    # B_block: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
    # C_block: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
    
    subproducts = [
        # (A_block, B_block, C_block)
        (0, 0, 0),  # A00*B00 -> C00
        (0, 1, 1),  # A00*B01 -> C01
        (0, 2, 2),  # A00*B02 -> C02
        (1, 3, 0),  # A01*B10 -> C00
        (1, 4, 1),  # A01*B11 -> C01
        (1, 5, 2),  # A01*B12 -> C02
        (2, 0, 3),  # A10*B00 -> C10
        (2, 1, 4),  # A10*B01 -> C11
        (2, 2, 5),  # A10*B02 -> C12
        (3, 3, 3),  # A11*B10 -> C10
        (3, 4, 4),  # A11*B11 -> C11
        (3, 5, 5),  # A11*B12 -> C12
    ]
    
    # Global index mappings
    # A: block -> 4 local indices in global A
    A_block_to_global = {
        0: [0, 1, 4, 5],    # A00
        1: [2, 3, 6, 7],    # A01
        2: [8, 9, 12, 13],  # A10
        3: [10, 11, 14, 15],# A11
    }
    # B: block -> 6 local indices in global B
    B_block_to_global = {
        0: [0, 1, 2, 9, 10, 11],    # B00
        1: [3, 4, 5, 12, 13, 14],   # B01
        2: [6, 7, 8, 15, 16, 17],   # B02
        3: [18, 19, 20, 27, 28, 29],# B10
        4: [21, 22, 23, 30, 31, 32],# B11
        5: [24, 25, 26, 33, 34, 35],# B12
    }
    # C: block -> 6 local indices in global C
    C_block_to_global = {
        0: [0, 1, 2, 9, 10, 11],    # C00
        1: [3, 4, 5, 12, 13, 14],   # C01
        2: [6, 7, 8, 15, 16, 17],   # C02
        3: [18, 19, 20, 27, 28, 29],# C10
        4: [21, 22, 23, 30, 31, 32],# C11
        5: [24, 25, 26, 33, 34, 35],# C12
    }
    
    col = 0
    for (a_blk, b_blk, c_blk) in subproducts:
        a_globals = A_block_to_global[a_blk]
        b_globals = B_block_to_global[b_blk]
        c_globals = C_block_to_global[c_blk]
        
        for r in range(rank // 12):  # 11 per sub-product
            # U: local A index -> global A
            for local_a, val in enumerate(Uw[:, r]):
                if val != 0:
                    U[a_globals[local_a], col] = val
            
            # V: local B index -> global B
            for local_b, val in enumerate(Vw[:, r]):
                if val != 0:
                    V[b_globals[local_b], col] = val
            
            # W: local C index -> global C
            for local_c, val in enumerate(Ww[:, r]):
                if val != 0:
                    W[c_globals[local_c], col] = val
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE

Wait, I made an error. The `_winograd_2x2x3` function returns rank 11, but I'm using `rank // 12` which would be 132//12 = 11. That's correct. But let me double-check the logic.

Actually, I realize there might be an issue with my Winograd decomposition. Let me verify it's correct by checking that the Strassen part correctly handles the 2x2 x 2x2 case for the first two columns.

Let me also add a verification step to make sure the decomposition is correct before returning.

