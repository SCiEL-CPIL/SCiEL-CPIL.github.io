import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Returns a decomposition for 4x5 by 5x6 matrix multiplication.
    Strategy: Hybrid block decomposition.
    1. Split the 5x5 middle dimension into 4+1.
    2. For the 4x4x4 part (i in 0..3, j in 0..3, k in 0..3):
       Use Strassen's algorithm for 2x2 blocks.
       The 4x4 multiplication is treated as multiplying four 2x2 blocks.
       However, Strassen reduces the number of 2x2 multiplications from 8 to 7.
       Each 2x2 multiplication is then implemented using 7 scalar multiplications (rank 7).
       Total rank for 4x4x4 = 7 * 7 = 49.
    3. For the 4x4x2 part (i in 0..3, j in 0..3, k in 4..5):
       Use Schoolbook. Rank = 4 * 4 * 2 = 32.
    4. For the 4x1x6 part (i in 0..3, j=4, k in 0..5):
       Use Schoolbook (outer product). Rank = 4 * 1 * 6 = 24.
    
    Total Rank = 49 + 32 + 24 = 105.
    """
    n, m, p = 4, 5, 6
    
    # Shapes
    U_dim = n * m  # 20
    V_dim = m * p  # 30
    W_dim = p * n  # 24
    
    # Rank components
    r_strassen_444 = 49
    r_school_442 = 32
    r_school_416 = 24
    total_rank = r_strassen_444 + r_school_442 + r_school_416
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(U_dim)]
    V = [[0] * total_rank for _ in range(V_dim)]
    W = [[0] * total_rank for _ in range(W_dim)]
    
    r = 0
    
    # --- Part 1: Strassen for 4x4x4 ---
    # Indices: i in 0..3, j in 0..3, k in 0..3
    # We split A (4x4), B (4x4), C (4x4) into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # C = A * B = [[C11, C12], [C21, C22]]
    
    # Strassen formulas for the 7 products (M1..M7):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B22 - B21)
    # M4 = A22 * (B11 - B12)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Result blocks:
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # Each Mi is a 2x2 matrix product. We implement each 2x2 product using 7 scalar terms (standard Strassen for 2x2).
    # Let's define a helper to add the 7 scalar terms for a specific 2x2x2 multiplication defined by linear combinations.
    # Instead of generic helpers, we will explicitly construct the 49 terms.
    # For each of the 7 Strassen terms (M1..M7), we have a 2x2 multiplication.
    # Let's map the 2x2 blocks to global indices.
    # A11: i=0,1; j=0,1
    # A12: i=0,1; j=2,3
    # A21: i=2,3; j=0,1
    # A22: i=2,3; j=2,3
    # Similarly for B and C.
    
    # We will iterate through the 7 Strassen terms.
    # For each term, we define the coefficients for A and B elements.
    # Then we use the standard 7-term Strassen decomposition for the 2x2 multiplication.
    
    # To keep it simple and robust, we will pre-compute the 49 (u, v, w) triples.
    # A 2x2x2 Strassen decomposition has 7 terms.
    # Let's define the 7 terms for a generic 2x2 multiplication of X (2x2) and Y (2x2) resulting in Z (2x2).
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    
    # Strassen terms for 2x2:
    # S1 = (x00 + x11)(y00 + y11)
    # S2 = (x10 + x11)y00
    # S3 = x00(y11 - y10)
    # S4 = x11(y00 - y01)
    # S5 = (x00 + x01)y11
    # S6 = (x10 - x00)(y00 + y01)
    # S7 = (x01 - x11)(y10 + y11)
    
    # Z00 = S1 - S4 + S6 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S3 + S5 + S7
    
    # We need to map the elements of X and Y to the global U and V indices.
    # For each of the 7 outer Strassen terms (M1..M7), X and Y are linear combinations of the 2x2 blocks.
    
    # Let's create a list of operations.
    # Each operation is a scalar multiplication: u_idx * v_idx -> w_idx with coefficient.
    # We will build a list of (u_idx, v_idx, w_idx, coeff) tuples.
    
    ops = []
    
    # Helper to add the 7 Strassen terms for a 2x2 multiplication
    # x_coeffs: list of 4 (u_idx, coeff) for x00, x01, x10, x11
    # y_coeffs: list of 4 (v_idx, coeff) for y00, y01, y10, y11
    # z_coeffs: list of 4 (w_idx, coeff) for z00, z01, z10, z11 (these are the target W indices with signs from the outer Strassen)
    # Note: The outer Strassen determines the signs for Z00, Z01, Z10, Z11.
    # The inner Strassen determines how X and Y combine.
    
    def add_2x2_strassen_ops(x_map, y_map, z_map):
        """
        x_map: dict {0: [(u_idx, coeff), ...], 1: ..., 2: ..., 3: ...} for x00, x01, x10, x11
        y_map: dict {0: [(v_idx, coeff), ...], ...} for y00, y01, y10, y11
        z_map: dict {0: [(w_idx, coeff), ...], ...} for z00, z01, z10, z11
        """
        # S1 = (x00 + x11)(y00 + y11)
        # Expansion: x00*y00 + x00*y11 + x11*y00 + x11*y11
        # We need to add these to the ops list.
        # But wait, Strassen combines them into one multiplication term in the rank count?
        # No, in the tensor decomposition, each scalar multiplication is a rank-1 term.
        # The "Strassen trick" is that we compute S1 = (x00+x11)(y00+y11) as a single product of two linear forms.
        # So we add ONE term to the rank:
        # U_vector = sum of x parts, V_vector = sum of y parts.
        # W_vector = sum of z parts with appropriate signs.
        
        # Term S1:
        u_terms = []
        for (idx, coeff) in x_map[0]: u_terms.append((idx, coeff)) # x00
        for (idx, coeff) in x_map[3]: u_terms.append((idx, coeff)) # x11
        v_terms = []
        for (idx, coeff) in y_map[0]: v_terms.append((idx, coeff)) # y00
        for (idx, coeff) in y_map[3]: v_terms.append((idx, coeff)) # y11
        
        # W contribution for S1:
        # Z00 += S1, Z11 += S1
        w_terms = []
        for (idx, coeff) in z_map[0]: w_terms.append((idx, coeff)) # z00
        for (idx, coeff) in z_map[3]: w_terms.append((idx, coeff)) # z11
        
        ops.extend([(u, v, w, cu * cv * cw) for u, cu in u_terms for v, cv in v_terms for w, cw in w_terms])
        
        # Term S2: (x10 + x11)y00
        u_terms = []
        for (idx, coeff) in x_map[2]: u_terms.append((idx, coeff)) # x10
        for (idx, coeff) in x_map[3]: u_terms.append((idx, coeff)) # x11
        v_terms = []
        for (idx, coeff) in y_map[0]: v_terms.append((idx, coeff)) # y00
        
        # W contribution for S2:
        # Z10 += S2
        w_terms = []
        for (idx, coeff) in z_map[2]: w_terms.append((idx, coeff)) # z10
        
        ops.extend([(u, v, w, cu * cv * cw) for u, cu in u_terms for v, cv in v_terms for w, cw in w_terms])
        
        # Term S3: x00(y11 - y10)
        u_terms = []
        for (idx, coeff) in x_map[0]: u_terms.append((idx, coeff)) # x00
        v_terms = []
        for (idx, coeff) in y_map[3]: v_terms.append((idx, coeff)) # y11
        for (idx, coeff) in y_map[2]: v_terms.append((idx, -coeff)) # y10 with -1
        
        # W contribution for S3:
        # Z01 += S3, Z11 -= S3
        w_terms = []
        for (idx, coeff) in z_map[1]: w_terms.append((idx, coeff)) # z01
        for (idx, coeff) in z_map[3]: w_terms.append((idx, -coeff)) # z11
        
        ops.extend([(u, v, w, cu * cv * cw) for u, cu in u_terms for v, cv in v_terms for w, cw in w_terms])
        
        # Term S4: x11(y00 - y01)
        u_terms = []
        for (idx, coeff) in x_map[3]: u_terms.append((idx, coeff)) # x11
        v_terms = []
        for (idx, coeff) in y_map[0]: v_terms.append((idx, coeff)) # y00
        for (idx, coeff) in y_map[1]: v_terms.append((idx, -coeff)) # y01 with -1
        
        # W contribution for S4:
        # Z00 -= S4, Z10 += S4
        w_terms = []
        for (idx, coeff) in z_map[0]: w_terms.append((idx, -coeff)) # z00
        for (idx, coeff) in z_map[2]: w_terms.append((idx, coeff)) # z10
        
        ops.extend([(u, v, w, cu * cv * cw) for u, cu in u_terms for v, cv in v_terms for w, cw in w_terms])
        
        # Term S5: (x00 + x01)y11
        u_terms = []
        for (idx, coeff) in x_map[0]: u_terms.append((idx, coeff)) # x00
        for (idx, coeff) in x_map[1]: u_terms.append((idx, coeff)) # x01
        v_terms = []
        for (idx, coeff) in y_map[3]: v_terms.append((idx, coeff)) # y11
        
        # W contribution for S5:
        # Z01 += S5, Z11 += S5
        w_terms = []
        for (idx, coeff) in z_map[1]: w_terms.append((idx, coeff)) # z01
        for (idx, coeff) in z_map[3]: w_terms.append((idx, coeff)) # z11
        
        ops.extend([(u, v, w, cu * cv * cw) for u, cu in u_terms for v, cv in v_terms for w, cw in w_terms])
        
        # Term S6: (x10 - x00)(y00 + y01)
        u_terms = []
        for (idx, coeff) in x_map[2]: u_terms.append((idx, coeff)) # x10
        for (idx, coeff) in x_map[0]: u_terms.append((idx, -coeff)) # x00 with -1
        v_terms = []
        for (idx, coeff) in y_map[0]: v_terms.append((idx, coeff)) # y00
        for (idx, coeff) in y_map[1]: v_terms.append((idx, coeff)) # y01
        
        # W contribution for S6:
        # Z00 += S6, Z10 -= S6
        w_terms = []
        for (idx, coeff) in z_map[0]: w_terms.append((idx, coeff)) # z00
        for (idx, coeff) in z_map[2]: w_terms.append((idx, -coeff)) # z10
        
        ops.extend([(u, v, w, cu * cv * cw) for u, cu in u_terms for v, cv in v_terms for w, cw in w_terms])
        
        # Term S7: (x01 - x11)(y10 + y11)
        u_terms = []
        for (idx, coeff) in x_map[1]: u_terms.append((idx, coeff)) # x01
        for (idx, coeff) in x_map[3]: u_terms.append((idx, -coeff)) # x11 with -1
        v_terms = []
        for (idx, coeff) in y_map[2]: v_terms.append((idx, coeff)) # y10
        for (idx, coeff) in y_map[3]: v_terms.append((idx, coeff)) # y11
        
        # W contribution for S7:
        # Z00 += S7, Z11 += S7
        w_terms = []
        for (idx, coeff) in z_map[0]: w_terms.append((idx, coeff)) # z00
        for (idx, coeff) in z_map[3]: w_terms.append((idx, coeff)) # z11
        
        ops.extend([(u, v, w, cu * cv * cw) for u, cu in u_terms for v, cv in v_terms for w, cw in w_terms])

    # Now define the 7 outer Strassen terms for 4x4.
    # A blocks:
    # A11: i=0,1; j=0,1. U indices: i*5+j
    # A12: i=0,1; j=2,3
    # A21: i=2,3; j=0,1
    # A22: i=2,3; j=2,3
    
    # B blocks:
    # B11: j=0,1; k=0,1. V indices: j*6+k
    # B12: j=0,1; k=2,3
    # B21: j=2,3; k=0,1
    # B22: j=2,3; k=2,3
    
    # C blocks (W indices: k*4+i):
    # C11: k=0,1; i=0,1
    # C12: k=0,1; i=2,3
    # C21: k=2,3; i=0,1
    # C22: k=2,3; i=2,3
    
    # We need to map the 2x2 elements of X and Y to the global indices.
    # For each outer term, X and Y are sums of blocks.
    # Example: M1 = (A11 + A22)(B11 + B22)
    # X = A11 + A22.
    # X00 = A11_00 + A22_00 = A[0,0] + A[2,2]
    # X01 = A11_01 + A22_01 = A[0,1] + A[2,3]
    # X10 = A11_10 + A22_10 = A[1,0] + A[3,2]
    # X11 = A11_11 + A22_11 = A[1,1] + A[3,3]
    
    # Let's create a helper to get the map for a block sum.
    def get_block_map(block_names, matrix_type):
        # matrix_type: 'A' or 'B'
        # block_names: e.g., ['A11', 'A22']
        # Returns a dict for x_map or y_map
        # Keys 0,1,2,3 correspond to 00, 01, 10, 11 within the 2x2 structure.
        # But here the "2x2 structure" is the block itself.
        # So for a single block like A11, the elements are:
        # 00: A[0,0]
        # 01: A[0,1]
        # 10: A[1,0]
        # 11: A[1,1]
        
        # If we sum A11 + A22:
        # 00: A[0,0] + A[2,2]
        # 01: A[0,1] + A[2,3]
        # 10: A[1,0] + A[3,2]
        # 11: A[1,1] + A[3,3]
        
        # We will build the map manually for each of the 7 terms.
        pass

    # Term M1: (A11 + A22)(B11 + B22)
    # X = A11 + A22
    x_map = {
        0: [(0*5+0, 1), (2*5+2, 1)], # A00 + A22
        1: [(0*5+1, 1), (2*5+3, 1)], # A01 + A23
        2: [(1*5+0, 1), (3*5+2, 1)], # A10 + A32
        3: [(1*5+1, 1), (3*5+3, 1)]  # A11 + A33
    }
    # Y = B11 + B22
    y_map = {
        0: [(0*6+0, 1), (2*6+0, 1)], # B00 + B20
        1: [(0*6+1, 1), (2*6+1, 1)], # B01 + B21
        2: [(1*6+0, 1), (3*6+0, 1)], # B10 + B30
        3: [(1*6+1, 1), (3*6+1, 1)]  # B11 + B31
    }
    # Z = C11 + C22 (with signs from C11 = M1..., C22 = M1...)
    # C11 gets +M1, C22 gets +M1
    z_map = {
        0: [(0*4+0, 1), (1*4+1, 1)], # C00 + C11? No.
        # C11 is k=0,1; i=0,1.
        # C11_00 = C[0,0] = W[0*4+0]
        # C11_01 = C[0,1] = W[0*4+1]
        # C11_10 = C[1,0] = W[1*4+0]
        # C11_11 = C[1,1] = W[1*4+1]
        # C22 is k=2,3; i=2,3.
        # C22_00 = C[2,2] = W[2*4+2]
        # C22_01 = C[2,3] = W[2*4+3]
        # C22_10 = C[3,2] = W[3*4+2]
        # C22_11 = C[3,3] = W[3*4+3]
        
        # So Z00 (top-left of C11) gets +1, Z11 (bottom-right of C22) gets +1.
        # Wait, the z_map in add_2x2_strassen_ops expects the coefficients for the 2x2 result Z.
        # Z is the 2x2 block result of the inner Strassen.
        # The inner Strassen produces a 2x2 matrix Z.
        # This Z is added to the global C.
        # For M1, Z contributes to C11 (top-left) and C22 (bottom-right).
        # So Z00 (top-left of Z) goes to C11_00.
        # Z01 (top-right of Z) goes to C11_01.
        # Z10 (bottom-left of Z) goes to C11_10.
        # Z11 (bottom-right of Z) goes to C11_11 AND C22_00?
        
        # NO. The Strassen formula is for a single 2x2 multiplication.
        # M1 is a 2x2 matrix.
        # C11 = M1 - M4 + M6 + M7.
        # C22 = M1 - M3 + M5 + M7.
        
        # So M1 (which is a 2x2 matrix) is added to C11 and C22.
        # The 2x2 matrix M1 has elements M1_00, M1_01, M1_10, M1_11.
        # M1_00 is added to C11_00 and C22_00?
        # No, C11 is a 2x2 block. C22 is a 2x2 block.
        # M1 is a 2x2 matrix.
        # The formula C11 = ... means the entire 2x2 block C11 is the sum of these 2x2 matrices.
        # So M1_00 contributes to C11_00.
        # M1_01 contributes to C11_01.
        # M1_10 contributes to C11_10.
        # M1_11 contributes to C11_11.
        # AND
        # M1_00 contributes to C22_00.
        # M1_01 contributes to C22_01.
        # M1_10 contributes to C22_10.
        # M1_11 contributes to C22_11.
        
        # Therefore, for the z_map in the inner Strassen call:
        # z_map[0] (Z00) should map to C11_00 and C22_00.
        # z_map[1] (Z01) should map to C11_01 and C22_01.
        # z_map[2] (Z10) should map to C11_10 and C22_10.
        # z_map[3] (Z11) should map to C11_11 and C22_11.
        
        0: [(0*4+0, 1), (2*4+2, 1)], # C11_00, C22_00
        1: [(0*4+1, 1), (2*4+3, 1)], # C11_01, C22_01
        2: [(1*4+0, 1), (3*4+2, 1)], # C11_10, C22_10
        3: [(1*4+1, 1), (3*4+3, 1)]  # C11_11, C22_11
    }
    add_2x2_strassen_ops(x_map, y_map, z_map)
    
    # Term M2: (A21 + A22)B11
    # X = A21 + A22
    x_map = {
        0: [(2*5+0, 1), (2*5+2, 1)], # A20 + A22
        1: [(2*5+1, 1), (2*5+3, 1)], # A21 + A23
        2: [(3*5+0, 1), (3*5+2, 1)], # A30 + A32
        3: [(3*5+1, 1), (3*5+3, 1)]  # A31 + A33
    }
    # Y = B11
    y_map = {
        0: [(0*6+0, 1)],
        1: [(0*6+1, 1)],
        2: [(1*6+0, 1)],
        3: [(1*6+1, 1)]
    }
    # Z contributes to C21.
    # C21 is k=2,3; i=0,1.
    # C21_00 = C[2,0] = W[2*4+0]
    # C21_01 = C[2,1] = W[2*4+1]
    # C21_10 = C[3,0] = W[3*4+0]
    # C21_11 = C[3,1] = W[3*4+1]
    z_map = {
        0: [(2*4+0, 1)],
        1: [(2*4+1, 1)],
        2: [(3*4+0, 1)],
        3: [(3*4+1, 1)]
    }
    add_2x2_strassen_ops(x_map, y_map, z_map)
    
    # Term M3: A11(B22 - B21)
    # X = A11
    x_map = {
        0: [(0*5+0, 1)],
        1: [(0*5+1, 1)],
        2: [(1*5+0, 1)],
        3: [(1*5+1, 1)]
    }
    # Y = B22 - B21
    y_map = {
        0: [(2*6+0, 1), (2*6+0, -1)], # B20 - B20? No.
        # B22: j=2,3; k=2,3. B21: j=2,3; k=0,1.
        # B22_00 = B[2,2], B21_00 = B[2,0]
        # B22_01 = B[2,3], B21_01 = B[2,1]
        # B22_10 = B[3,2], B21_10 = B[3,0]
        # B22_11 = B[3,3], B21_11 = B[3,1]
        0: [(2*6+2, 1), (2*6+0, -1)], # B22 - B21
        1: [(2*6+3, 1), (2*6+1, -1)],
        2: [(3*6+2, 1), (3*6+0, -1)],
        3: [(3*6+3, 1), (3*6+1, -1)]
    }
    # Z contributes to C12 (+) and C22 (-).
    # C12: k=0,1; i=2,3.
    # C22: k=2,3; i=2,3.
    z_map = {
        0: [(0*4+2, 1), (2*4+2, -1)], # C12_00, C22_00
        1: [(0*4+3, 1), (2*4+3, -1)], # C12_01, C22_01
        2: [(1*4+2, 1), (3*4+2, -1)], # C12_10, C22_10
        3: [(1*4+3, 1), (3*4+3, -1)]  # C12_11, C22_11
    }
    add_2x2_strassen_ops(x_map, y_map, z_map)
    
    # Term M4: A22(B11 - B12)
    # X = A22
    x_map = {
        0: [(2*5+2, 1)],
        1: [(2*5+3, 1)],
        2: [(3*5+2, 1)],
        3: [(3*5+3, 1)]
    }
    # Y = B11 - B12
    # B11: j=0,1; k=0,1. B12: j=0,1; k=2,3.
    y_map = {
        0: [(0*6+0, 1), (0*6+2, -1)],
        1: [(0*6+1, 1), (0*6+3, -1)],
        2: [(1*6+0, 1), (1*6+2, -1)],
        3: [(1*6+1, 1), (1*6+3, -1)]
    }
    # Z contributes to C11 (-) and C21 (+).
    z_map = {
        0: [(0*4+0, -1), (2*4+0, 1)],
        1: [(0*4+1, -1), (2*4+1, 1)],
        2: [(1*4+0, -1), (3*4+0, 1)],
        3: [(1*4+1, -1), (3*4+1, 1)]
    }
    add_2x2_strassen_ops(x_map, y_map, z_map)
    
    # Term M5: (A11 + A12)B22
    # X = A11 + A12
    x_map = {
        0: [(0*5+0, 1), (0*5+2, 1)],
        1: [(0*5+1, 1), (0*5+3, 1)],
        2: [(1*5+0, 1), (1*5+2, 1)],
        3: [(1*5+1, 1), (1*5+3, 1)]
    }
    # Y = B22
    y_map = {
        0: [(2*6+2, 1)],
        1: [(2*6+3, 1)],
        2: [(3*6+2, 1)],
        3: [(3*6+3, 1)]
    }
    # Z contributes to C12 (+) and C22 (+).
    z_map = {
        0: [(0*4+2, 1), (2*4+2, 1)],
        1: [(0*4+3, 1), (2*4+3, 1)],
        2: [(1*4+2, 1), (3*4+2, 1)],
        3: [(1*4+3, 1), (3*4+3, 1)]
    }
    add_2x2_strassen_ops(x_map, y_map, z_map)
    
    # Term M6: (A21 - A11)(B11 + B12)
    # X = A21 - A11
    x_map = {
        0: [(2*5+0, 1), (0*5+0, -1)],
        1: [(2*5+1, 1), (0*5+1, -1)],
        2: [(3*5+0, 1), (1*5+0, -1)],
        3: [(3*5+1, 1), (1*5+1, -1)]
    }
    # Y = B11 + B12
    y_map = {
        0: [(0*6+0, 1), (0*6+2, 1)],
        1: [(0*6+1, 1), (0*6+3, 1)],
        2: [(1*6+0, 1), (1*6+2, 1)],
        3: [(1*6+1, 1), (1*6+3, 1)]
    }
    # Z contributes to C11 (+) and C21 (-).
    z_map = {
        0: [(0*4+0, 1), (2*4+0, -1)],
        1: [(0*4+1, 1), (2*4+1, -1)],
        2: [(1*4+0, 1), (3*4+0, -1)],
        3: [(1*4+1, 1), (3*4+1, -1)]
    }
    add_2x2_strassen_ops(x_map, y_map, z_map)
    
    # Term M7: (A12 - A22)(B21 + B22)
    # X = A12 - A22
    x_map = {
        0: [(0*5+2, 1), (2*5+2, -1)],
        1: [(0*5+3, 1), (2*5+3, -1)],
        2: [(1*5+2, 1), (3*5+2, -1)],
        3: [(1*5+3, 1), (3*5+3, -1)]
    }
    # Y = B21 + B22
    y_map = {
        0: [(2*6+0, 1), (2*6+2, 1)],
        1: [(2*6+1, 1), (2*6+3, 1)],
        2: [(3*6+0, 1), (3*6+2, 1)],
        3: [(3*6+1, 1), (3*6+3, 1)]
    }
    # Z contributes to C11 (+) and C22 (+).
    z_map = {
        0: [(0*4+0, 1), (2*4+2, 1)],
        1: [(0*4+1, 1), (2*4+3, 1)],
        2: [(1*4+0, 1), (3*4+2, 1)],
        3: [(1*4+1, 1), (3*4+3, 1)]
    }
    add_2x2_strassen_ops(x_map, y_map, z_map)
    
    # Now populate U, V, W with the ops
    # Each op is (u_idx, v_idx, w_idx, coeff)
    # We need to aggregate coefficients for the same (u, v, w) triple?
    # No, each op corresponds to a rank-1 term.
    # But the ops list contains all the scalar multiplications expanded from the linear forms.
    # Wait, the `ops` list contains tuples like (u, v, w, coeff).
    # This represents coeff * U[u] * V[v] * W[w].
    # We can just add these to the matrices.
    
    for u, v, w, coeff in ops:
        U[u][r] = coeff
        V[v][r] = coeff
        W[w][r] = coeff
        r += 1
        
    # Check r
    assert r == 49, f"Strassen rank mismatch: {r}"
    
    # --- Part 2: Schoolbook for 4x4x2 ---
    # i in 0..3, j in 0..3, k in 4..5
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    assert r == 49 + 32, f"Schoolbook 442 rank mismatch: {r}"
    
    # --- Part 3: Schoolbook for 4x1x6 ---
    # i in 0..3, j=4, k in 0..5
    for i in range(4):
        for k in range(6):
            U[i * m + 4][r] = 1
            V[4 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    assert r == total_rank, f"Total rank mismatch: {r}"
    
    return {"U": U, "V": V, "W": W}

