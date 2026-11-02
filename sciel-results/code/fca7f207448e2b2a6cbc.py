import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,4,4}.
    Uses a Strassen-based block decomposition to reduce the rank below the schoolbook 32.
    The 2x4 * 4x4 product is viewed as 2x2 blocks of size 1x2 * 2x2.
    Each block uses Strassen's 7-multiplication algorithm.
    Total rank = 4 blocks * 7 = 28.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # We view the matrices as:
    # A is 2x4: split into 2 blocks of size 1x2 along columns?
    # Actually, standard block decomposition for (2,4,4):
    # Split A (2x4) into 2 row-blocks of size 1x4? No, that's not square.
    # Let's split the middle dimension m=4 into two halves of size 2.
    # A is 2x4 -> A_left (2x2), A_right (2x2)
    # B is 4x4 -> B_top (2x4), B_bottom (2x4)
    # C = A_left * B_top + A_right * B_bottom
    # This is sum of two 2x2 * 2x4 products.
    # Each 2x2 * 2x4 product can be done with rank 2 * 2 * 4 / 2? No.
    # Rank of 2x2 * 2x4 is at most 2*2*4 = 16, but Strassen on 2x2 blocks?
    # Let's split B (4x4) into 2x2 blocks too?
    # B is 4x4 -> B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2)
    # A is 2x4 -> A1 (2x2), A2 (2x2)
    # C is 2x4 -> C1 (2x2), C2 (2x2)
    # C1 = A1 * B11 + A2 * B21
    # C2 = A1 * B12 + A2 * B22
    # This is two independent 2x2 * 2x2 products summed? No, it's a sum of two products.
    # We can use Strassen for each sum?
    # Actually, let's just use the Strassen algorithm for 2x2 * 2x2 four times?
    # No, C1 = A1 B11 + A2 B21. This is a bilinear map from (A1, A2, B11, B21) to C1.
    # This is equivalent to multiplying a 2x4 matrix (A1 A2) by a 4x2 matrix (B11; B21).
    # The rank of 2x4 * 4x2 is at most 2*4*2 = 16, but can be lower.
    # Actually, the rank of 2x4 * 4x2 is 2*4*2 / 2 = 8? No, Strassen for 2x2 * 2x2 is 7.
    # Let's use the standard Strassen decomposition for 2x2 * 2x2.
    # We have 4 such products:
    # P1 = A1 B11
    # P2 = A2 B21
    # P3 = A1 B12
    # P4 = A2 B22
    # C1 = P1 + P2
    # C2 = P3 + P4
    # If we do each with rank 7, total rank is 28.
    # But we can do better by combining?
    # Let's just implement the 4 separate Strassen decompositions.
    # Each 2x2 * 2x2 product uses 7 scalar multiplications.
    # Total 28 multiplications.
    
    # Strassen for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p, q], [r, s]]
    # p = (a+d)(e+h) - (c+d)h + d(e+g)  ... wait, standard is:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+a)(e+g)
    # M7 = (b+d)(f+h)
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M3 + M5 - M7
    
    # We need to map these to the global indices.
    # Global A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # A1 is A[:, 0:2], A2 is A[:, 2:4].
    # Global B is 4x4. Rows 0,1,2,3. Cols 0,1,2,3.
    # B11 is B[0:2, 0:2], B12 is B[0:2, 2:4], B21 is B[2:4, 0:2], B22 is B[2:4, 2:4].
    # Global C is 2x4. Rows 0,1. Cols 0,1,2,3.
    # C1 is C[:, 0:2], C2 is C[:, 2:4].
    
    # For each of the 4 products (A1*B11, A2*B21, A1*B12, A2*B22), we apply Strassen.
    # Let's define the 4 products explicitly.
    # Product 0: A1 * B11 -> C1
    # Product 1: A2 * B21 -> C1
    # Product 2: A1 * B12 -> C2
    # Product 3: A2 * B22 -> C2
    
    # For each product, we have 7 terms.
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 8 x 28
    
    # Index mapping:
    # U row: i * m + j. i in [0,1], j in [0,3].
    # V row: j * p + k. j in [0,3], k in [0,3].
    # W row: k * n + i. k in [0,3], i in [0,1].
    
    # Strassen terms for a 2x2 * 2x2 product:
    # Let A_sub be 2x2, B_sub be 2x2, C_sub be 2x2.
    # A_sub indices (local i, local j) -> global (i, j_offset + local j)
    # B_sub indices (local j, local k) -> global (j_offset + local j, k_offset + local k)
    # C_sub indices (local i, local k) -> global (i, k_offset + local k)
    
    # Product 0: A1 (cols 0,1) * B11 (rows 0,1, cols 0,1) -> C1 (cols 0,1)
    # j_offset_A = 0, j_offset_B = 0, k_offset_C = 0
    # Product 1: A2 (cols 2,3) * B21 (rows 2,3, cols 0,1) -> C1 (cols 0,1)
    # j_offset_A = 2, j_offset_B = 2, k_offset_C = 0
    # Product 2: A1 (cols 0,1) * B12 (rows 0,1, cols 2,3) -> C2 (cols 2,3)
    # j_offset_A = 0, j_offset_B = 0, k_offset_C = 2
    # Product 3: A2 (cols 2,3) * B22 (rows 2,3, cols 2,3) -> C2 (cols 2,3)
    # j_offset_A = 2, j_offset_B = 2, k_offset_C = 2
    
    products = [
        (0, 0, 0),  # (j_offset_A, j_offset_B, k_offset_C)
        (2, 2, 0),
        (0, 0, 2),
        (2, 2, 2)
    ]
    
    # Strassen formulas (coefficients for A, B, C)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00+B01)
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10+A00)(B00+B10)
    # M7 = (A01+A11)(B01+B11)
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 - M4 + M6
    # C11 = M1 - M3 + M5 - M7
    
    # We need to express each M as a sum of products of linear forms in A and B.
    # For example, M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11.
    # This is a sum of 4 rank-1 terms.
    # But Strassen uses 7 products, each of which is a product of linear forms.
    # Each linear form is a sum of A entries and a sum of B entries.
    # So each M is a product of two linear forms, which is a sum of rank-1 terms.
    # But in the tensor decomposition, each rank-1 term corresponds to a single scalar multiplication.
    # So we need to expand each M into its constituent rank-1 terms.
    # Total rank is the sum of the number of rank-1 terms in all 7 M's.
    # M1 has 4 terms, M2 has 4, M3 has 2, M4 has 2, M5 has 2, M6 has 4, M7 has 4.
    # Total 22 terms per 2x2 product.
    # 4 products * 22 = 88 terms.
    # This is worse than schoolbook 32.
    
    # Wait, the rank of the tensor is the minimum number of rank-1 terms.
    # Strassen shows that the bilinear complexity is 7, meaning we can do it with 7 scalar multiplications.
    # But each scalar multiplication is of linear forms, not single entries.
    # In the tensor decomposition, a rank-1 term is u v^T w^T, where u, v, w are vectors.
    # If u and v have multiple non-zero entries, it still counts as 1 rank term.
    # So we don't need to expand M1 into 4 terms. M1 itself is a rank-1 term in the tensor?
    # No, M1 is a product of two linear forms. The term in the tensor is (linear form in A) * (linear form in B) * (linear form in C).
    # So each M corresponds to one rank-1 term in the tensor decomposition.
    # Therefore, the rank is 7 per 2x2 product.
    # Total rank 28.
    
    # Let's construct the 7 terms for each product.
    # For a given product with offsets (ja, jb, kc):
    # A indices: (0, ja+0), (0, ja+1), (1, ja+0), (1, ja+1)
    # B indices: (jb+0, kc+0), (jb+0, kc+1), (jb+1, kc+0), (jb+1, kc+1)
    # C indices: (0, kc+0), (0, kc+1), (1, kc+0), (1, kc+1)
    
    # We'll create a list of (U_vec, V_vec, W_vec) for each term.
    # U_vec is length 8, V_vec is length 16, W_vec is length 8.
    
    def get_u_idx(i, j):
        return i * m + j
    
    def get_v_idx(j, k):
        return j * p + k
    
    def get_w_idx(k, i):
        return k * n + i
    
    terms = []
    
    for (ja, jb, kc) in products:
        # Local indices for A: (0,0), (0,1), (1,0), (1,1)
        # Global A indices:
        a00 = get_u_idx(0, ja + 0)
        a01 = get_u_idx(0, ja + 1)
        a10 = get_u_idx(1, ja + 0)
        a11 = get_u_idx(1, ja + 1)
        
        # Local indices for B: (0,0), (0,1), (1,0), (1,1)
        # Global B indices:
        b00 = get_v_idx(jb + 0, kc + 0)
        b01 = get_v_idx(jb + 0, kc + 1)
        b10 = get_v_idx(jb + 1, kc + 0)
        b11 = get_v_idx(jb + 1, kc + 1)
        
        # Local indices for C: (0,0), (0,1), (1,0), (1,1)
        # Global C indices:
        c00 = get_w_idx(kc + 0, 0)
        c01 = get_w_idx(kc + 1, 0)
        c10 = get_w_idx(kc + 0, 1)
        c11 = get_w_idx(kc + 1, 1)
        
        # M1 = (A00+A11)(B00+B11)
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: c00=1, c11=1
        terms.append(([a00, a11], [b00, b11], [c00, c11]))
        
        # M2 = (A10+A11)(B00+B01)
        terms.append(([a10, a11], [b00, b01], [c10, c11]))
        
        # M3 = A00(B01-B11)
        terms.append(([a00], [b01, b11], [c01, c11]))
        # Wait, M3 = A00 * (B01 - B11).
        # This is A00*B01 - A00*B11.
        # So it's two rank-1 terms?
        # No, in the tensor decomposition, the term is (A00) * (B01 - B11) * (C01 + C11)?
        # No, the C part is determined by where M3 contributes.
        # M3 contributes to C01 and C11.
        # C01 = M3 + M5
        # C11 = M1 - M3 + M5 - M7
        # So M3 contributes +1 to C01 and -1 to C11.
        # So the W vector for M3 has 1 at c01 and -1 at c11.
        # The V vector has 1 at b01 and -1 at b11.
        # The U vector has 1 at a00.
        # So it's a single rank-1 term with negative entries.
        
        # Let's correct M3.
        # U: a00=1
        # V: b01=1, b11=-1
        # W: c01=1, c11=-1
        terms[-1] = ([a00], [b01, b11], [c01, c11])
        
        # M4 = A11(B10-B00)
        # U: a11=1
        # V: b10=1, b00=-1
        # W: c00=-1, c10=-1
        terms.append(([a11], [b10, b00], [c00, c10]))
        
        # M5 = (A00+A01)B11
        # U: a00=1, a01=1
        # V: b11=1
        # W: c01=1, c11=1
        terms.append(([a00, a01], [b11], [c01, c11]))
        
        # M6 = (A10+A00)(B00+B10)
        # U: a10=1, a00=1
        # V: b00=1, b10=1
        # W: c00=1, c10=1
        terms.append(([a10, a00], [b00, b10], [c00, c10]))
        
        # M7 = (A01+A11)(B01+B11)
        # U: a01=1, a11=1
        # V: b01=1, b11=1
        # W: c00=1, c11=-1
        terms.append(([a01, a11], [b01, b11], [c00, c11]))
    
    # Now, we have 28 terms.
    # Each term is (U_indices, V_indices, W_indices) with coefficients 1 or -1.
    # We need to fill U, V, W matrices.
    
    for c, (u_idxs, v_idxs, w_idxs) in enumerate(terms):
        for idx in u_idxs:
            U[idx, c] += 1.0
        for idx in v_idxs:
            V[idx, c] += 1.0
        for idx in w_idxs:
            W[idx, c] += 1.0
    
    # Check for negative coefficients in V and W for M3, M4, M7.
    # M3: V has b01=1, b11=-1. W has c01=1, c11=-1.
    # M4: V has b10=1, b00=-1. W has c00=-1, c10=-1.
    # M7: W has c00=1, c11=-1.
    
    # We need to handle the signs correctly.
    # Let's re-do the terms with explicit signs.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0
    for (ja, jb, kc) in products:
        a00 = get_u_idx(0, ja + 0)
        a01 = get_u_idx(0, ja + 1)
        a10 = get_u_idx(1, ja + 0)
        a11 = get_u_idx(1, ja + 1)
        
        b00 = get_v_idx(jb + 0, kc + 0)
        b01 = get_v_idx(jb + 0, kc + 1)
        b10 = get_v_idx(jb + 1, kc + 0)
        b11 = get_v_idx(jb + 1, kc + 1)
        
        c00 = get_w_idx(kc + 0, 0)
        c01 = get_w_idx(kc + 1, 0)
        c10 = get_w_idx(kc + 0, 1)
        c11 = get_w_idx(kc + 1, 1)
        
        # M1
        U[a00, c] = 1; U[a11, c] = 1
        V[b00, c] = 1; V[b11, c] = 1
        W[c00, c] = 1; W[c11, c] = 1
        c += 1
        
        # M2
        U[a10, c] = 1; U[a11, c] = 1
        V[b00, c] = 1; V[b01, c] = 1
        W[c10, c] = 1; W[c11, c] = 1
        c += 1
        
        # M3
        U[a00, c] = 1
        V[b01, c] = 1; V[b11, c] = -1
        W[c01, c] = 1; W[c11, c] = -1
        c += 1
        
        # M4
        U[a11, c] = 1
        V[b10, c] = 1; V[b00, c] = -1
        W[c00, c] = -1; W[c10, c] = -1
        c += 1
        
        # M5
        U[a00, c] = 1; U[a01, c] = 1
        V[b11, c] = 1
        W[c01, c] = 1; W[c11, c] = 1
        c += 1
        
        # M6
        U[a10, c] = 1; U[a00, c] = 1
        V[b00, c] = 1; V[b10, c] = 1
        W[c00, c] = 1; W[c10, c] = 1
        c += 1
        
        # M7
        U[a01, c] = 1; U[a11, c] = 1
        V[b01, c] = 1; V[b11, c] = 1
        W[c00, c] = 1; W[c11, c] = -1
        c += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
