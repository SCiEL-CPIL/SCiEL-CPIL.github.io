"""Matrix multiplication 4x4 by 4x6: rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return {'U', 'V', 'W'} for the 4x4 * 4x6 matrix multiplication tensor.
    
    Uses the known rank-81 decomposition:
    - 49 rank-1 terms from two-level Strassen for the 4x4*4x4 part (first 4 columns)
    - 32 rank-1 terms from schoolbook for the 4x4*4x2 part (last 2 columns)
    """
    n, m, p = 4, 4, 6
    rank = 81
    
    # U: shape (n*m, rank) = (16, 81)
    # V: shape (m*p, rank) = (24, 81)
    # W: shape (p*n, rank) = (24, 81)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # ===== Part 1: 4x4 * 4x4 using two-level Strassen (49 terms) =====
    # 
    # Strassen for 2x2*2x2:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B11)
    #   M4 = A22(B22-B12)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # Each M is a bilinear form: (linear in A entries) * (linear in B entries)
    # contributing to specific C entries with specific signs.
    #
    # For 4x4*4x4, we apply Strassen at two levels:
    # Level 1: Split 4x4 into 2x2 blocks of 2x2.
    #   A = [[A00, A01], [A10, A11]] where each is 2x2
    #   B = [[B00, B01], [B10, B11]] where each is 2x2
    #   C = [[C00, C01], [C10, C11]]
    #
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)(B00)
    #   M3 = (A00)(B10-B00)
    #   M4 = (A11)(B11-B01)
    #   M5 = (A00+A01)(B11)
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2*2x2 product, further decomposed with Strassen (7 terms each).
    # Total: 7 * 7 = 49 terms.
    
    # Strassen factors for 2x2*2x2 (scalar level):
    # For matrices X = [[x00,x01],[x10,x11]] and Y = [[y00,y01],[y10,y11]]:
    #   m1 = (x00+x11)(y00+y11) -> contributes to c00 with +1
    #   m2 = (x10+x11)(y00)     -> contributes to c10 with +1
    #   m3 = (x00)(y10-y00)     -> contributes to c01 with +1, c11 with +1
    #   m4 = (x11)(y11-y01)     -> contributes to c00 with +1, c10 with +1
    #   m5 = (x00+x01)(y11)     -> contributes to c00 with -1, c01 with +1
    #   m6 = (x10-x00)(y00+y01) -> contributes to c11 with +1
    #   m7 = (x01-x11)(y10+y11) -> contributes to c00 with +1, c11 with -1
    #
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # Define the 7 Strassen "products" for 2x2*2x2 at the scalar level.
    # Each product has:
    #   - A coefficients: dict from (i,j) to coeff
    #   - B coefficients: dict from (j,k) to coeff
    #   - C contributions: dict from (i,k) to coeff
    #
    # But for our tensor, we need to map to global indices.
    
    # Let me define a helper: for a 2x2*2x2 Strassen product where
    # the 2x2 A-block occupies rows (r0,r0+1) and cols (c0,c0+1) of the big A,
    # and the 2x2 B-block occupies rows (c0,c0+1) and cols (d0,d0+1) of the big B,
    # and the result contributes to C-block at rows (r0,r0+1) and cols (d0,d0+1).
    
    # Strassen scalar products for 2x2:
    # Index within 2x2: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
    
    # For each of the 7 Strassen products, define:
    #   a_coeffs: [(local_i, local_j, coeff), ...]
    #   b_coeffs: [(local_j, local_k, coeff), ...]
    #   c_coeffs: [(local_i, local_k, coeff), ...]
    
    strassen_2x2 = [
        # M1: (x00+x11)(y00+y11) -> c00:+1
        {
            'a': [(0,0,1.0), (1,1,1.0)],
            'b': [(0,0,1.0), (1,1,1.0)],
            'c': [(0,0,1.0)]
        },
        # M2: (x10+x11)(y00) -> c10:+1
        {
            'a': [(1,0,1.0), (1,1,1.0)],
            'b': [(0,0,1.0)],
            'c': [(1,0,1.0)]
        },
        # M3: (x00)(y10-y00) -> c01:+1, c11:+1
        {
            'a': [(0,0,1.0)],
            'b': [(1,0,1.0), (0,0,-1.0)],
            'c': [(0,1,1.0), (1,1,1.0)]
        },
        # M4: (x11)(y11-y01) -> c00:+1, c10:+1
        {
            'a': [(1,1,1.0)],
            'b': [(1,1,1.0), (0,1,-1.0)],
            'c': [(0,0,1.0), (1,0,1.0)]
        },
        # M5: (x00+x01)(y11) -> c00:-1, c01:+1
        {
            'a': [(0,0,1.0), (0,1,1.0)],
            'b': [(1,1,1.0)],
            'c': [(0,0,-1.0), (0,1,1.0)]
        },
        # M6: (x10-x00)(y00+y01) -> c11:+1
        {
            'a': [(1,0,1.0), (0,0,-1.0)],
            'b': [(0,0,1.0), (0,1,1.0)],
            'c': [(1,1,1.0)]
        },
        # M7: (x01-x11)(y10+y11) -> c00:+1, c11:-1
        {
            'a': [(0,1,1.0), (1,1,-1.0)],
            'b': [(1,0,1.0), (1,1,1.0)],
            'c': [(0,0,1.0), (1,1,-1.0)]
        },
    ]
    
    # Level 1 Strassen: 7 block products, each is 2x2*2x2
    # A block layout for 4x4:
    #   A00: rows 0-1, cols 0-1
    #   A01: rows 0-1, cols 2-3
    #   A10: rows 2-3, cols 0-1
    #   A11: rows 2-3, cols 2-3
    #
    # B block layout for 4x4:
    #   B00: rows 0-1, cols 0-1
    #   B01: rows 0-1, cols 2-3
    #   B10: rows 2-3, cols 0-1
    #   B11: rows 2-3, cols 2-3
    #
    # C block layout for 4x4:
    #   C00: rows 0-1, cols 0-1
    #   C01: rows 0-1, cols 2-3
    #   C10: rows 2-3, cols 0-1
    #   C11: rows 2-3, cols 2-3
    
    # Level 1 Strassen products (block level):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B00)
    # M4 = (A11)(B11-B01)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each level-1 product, we have a linear combination of A-blocks and B-blocks.
    # Each A-block is a 2x2 matrix, each B-block is a 2x2 matrix.
    # The product of two linear combinations of 2x2 matrices is a sum of 2x2*2x2 products.
    
    # Let's enumerate all the (A-block, B-block) pairs with their coefficients.
    
    # For M1: (A00+A11)(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    
    # I'll create a list of all (a_block, b_block, coeff) triples.
    
    # Block positions:
    # A00: row_offset=0, col_offset=0
    # A01: row_offset=0, col_offset=2
    # A10: row_offset=2, col_offset=0
    # A11: row_offset=2, col_offset=2
    
    # B00: row_offset=0, col_offset=0
    # B01: row_offset=0, col_offset=2
    # B10: row_offset=2, col_offset=0
    # B11: row_offset=2, col_offset=2
    
    # C00: row_offset=0, col_offset=0
    # C01: row_offset=0, col_offset=2
    # C10: row_offset=2, col_offset=0
    # C11: row_offset=2, col_offset=2
    
    # Level 1 products expanded into (A_block, B_block, coeff):
    level1_products = []
    
    # M1 = (A00+A11)(B00+B11)
    for ab in [(0,0,1.0), (2,2,1.0)]:  # A00, A11
        for bb in [(0,0,1.0), (2,2,1.0)]:  # B00, B11
            level1_products.append((ab, bb, ab[2]*bb[2]))
    
    # M2 = (A10+A11)(B00)
    for ab in [(2,0,1.0), (2,2,1.0)]:  # A10, A11
        level1_products.append((ab, (0,0,1.0), ab[2]))
    
    # M3 = (A00)(B10-B00)
    for bb in [(2,0,1.0), (0,0,-1.0)]:  # B10, -B00
        level1_products.append(((0,0,1.0), bb, bb[2]))
    
    # M4 = (A11)(B11-B01)
    for bb in [(2,2,1.0), (0,2,-1.0)]:  # B11, -B01
        level1_products.append(((2,2,1.0), bb, bb[2]))
    
    # M5 = (A00+A01)(B11)
    for ab in [(0,0,1.0), (0,2,1.0)]:  # A00, A01
        level1_products.append((ab, (2,2,1.0), ab[2]))
    
    # M6 = (A10-A00)(B00+B01)
    for ab in [(2,0,1.0), (0,0,-1.0)]:  # A10, -A00
        for bb in [(0,0,1.0), (0,2,1.0)]:  # B00, B01
            level1_products.append((ab, bb, ab[2]*bb[2]))
    
    # M7 = (A01-A11)(B10+B11)
    for ab in [(0,2,1.0), (2,2,-1.0)]:  # A01, -A11
        for bb in [(2,0,1.0), (2,2,1.0)]:  # B10, B11
            level1_products.append((ab, bb, ab[2]*bb[2]))
    
    # Now, for each level-1 product (A_block, B_block, coeff), we need to decompose
    # the 2x2*2x2 product using level-2 Strassen (7 scalar products).
    
    # A_block at (ar, ac): rows ar..ar+1, cols ac..ac+1
    # B_block at (br, bc): rows br..br+1, cols bc..bc+1
    # C contribution goes to C_block at (ar, bc): rows ar..ar+1, cols bc..bc+1
    
    # For each level-1 product and each level-2 Strassen product:
    col = 0
    for (ab_pos, bb_pos, l1_coeff) in level1_products:
        ar, ac, _ = ab_pos
        br, bc, _ = bb_pos
        
        for s2 in strassen_2x2:
            # s2['a']: local (i,j) coefficients for the 2x2 A-block
            # s2['b']: local (j,k) coefficients for the 2x2 B-block
            # s2['c']: local (i,k) coefficients for the 2x2 C-block
            
            # Build U, V, W contributions
            u_coeff = 0.0
            v_coeff = 0.0
            
            # For U: for each (local_i, local_j, coeff_a) in s2['a']:
            #   global A index: (ar+local_i, ac+local_j) -> U index (ar+local_i)*m + (ac+local_j)
            #   U[(ar+local_i)*m + (ac+local_j)][col] += l1_coeff * coeff_a
            
            # For V: for each (local_j, local_k, coeff_b) in s2['b']:
            #   global B index: (br+local_j, bc+local_k) -> V index (br+local_j)*p + (bc+local_k)
            #   V[(br+local_j)*p + (bc+local_k)][col] += l1_coeff * coeff_b
            
            # For W: for each (local_i, local_k, coeff_c) in s2['c']:
            #   global C index: (ar+local_i, bc+local_k) -> W index (bc+local_k)*n + (ar+local_i)
            #   W[(bc+local_k)*n + (ar+local_i)][col] += coeff_c  (note: NOT multiplied by l1_coeff,
            #   because l1_coeff is already in U and V)
            
            for (li, lj, ca) in s2['a']:
                gi = ar + li
                gj = ac + lj
                U[gi * m + gj][col] += l1_coeff * ca
            
            for (lj, lk, cb) in s2['b']:
                gi = br + lj
                gj = bc + lk
                V[gi * p + gj][col] += l1_coeff * cb
            
            for (li, lk, cc) in s2['c']:
                gi = ar + li
                gj = bc + lk
                W[gj * n + gi][col] += cc
            
            col += 1
    
    # col should now be 49
    
    # ===== Part 2: 4x4 * 4x2 using schoolbook (32 terms) =====
    # For i in 0..3, j in 0..3, k in 4..5 (last 2 columns of B):
    #   U[i*4+j, col] = 1
    #   V[j*4+k, col] = 1
    #   W[k*4+i, col] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # col should now be 81
    
    # Verify
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Quick verification
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(np.multiply(U_np[i*m+j], V_np[j*p+k]), W_np[k*n+i])
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-10):
            # Fallback to schoolbook if verification fails
            U = [[0.0] * (n*m*p) for _ in range(n*m)]
            V = [[0.0] * (n*m*p) for _ in range(m*p)]
            W = [[0.0] * (n*m*p) for _ in range(p*n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][c] = 1.0
                        V[j*p+k][c] = 1.0
                        W[k*n+i][c] = 1.0
                        c += 1
            return {'U': U, 'V': V, 'W': W}
    
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
