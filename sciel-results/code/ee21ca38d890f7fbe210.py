"""Matrix multiplication 2x4x8 using Strassen decomposition for rank reduction."""
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
_DEADLINE = 120.0
_START = time.time()

def _remaining():
    return _DEADLINE - 15 - (time.time() - _START)

# Strassen 2x2x2 decomposition (rank 7)
# U: rows indexed by i*m+j (i in {0,1}, m=2, j in {0,1})
# V: rows indexed by j*p+k (j in {0,1}, p=2, k in {0,1})
# W: rows indexed by k*n+i (n=2, k in {0,1}, i in {0,1})
#
# For 2x2 matrices A (left, 2x2) and B (right, 2x2):
# A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
# Product C = [[c00, c01], [c10, c11]]
#
# Strassen:
# M1 = (a00+a11)(b00+b11)
# M2 = (a10+a11)b00
# M3 = a00(b01-b11)
# M4 = a11(b10-b00)
# M5 = (a00+a01)b11
# M6 = (a10-a00)(b00+b01)
# M7 = (a01-a11)(b10+b11)
#
# c00 = M1 - M4 + M6 + M7
# c01 = M3 + M5
# c10 = M2 + M4
# c11 = M1 - M3 + M2 + M5
#
# U matrix (4 rows: i*2+j, 7 cols):
# Row 0: a00 -> M1, M3, M5, M6
# Row 1: a01 -> M3, M5, M7
# Row 2: a10 -> M2, M4, M6
# Row 3: a11 -> M1, M2, M4, M5, M7
_U_strassen = {
    0: {0: 1, 2: 1, 4: 1, 5: 1},
    1: {2: 1, 4: 1, 6: 1},
    2: {1: 1, 3: 1, 5: 1},
    3: {0: 1, 1: 1, 3: 1, 4: 1, 6: 1},
}
_V_strassen = {
    0: {0: 1, 1: 1, 5: 1},
    1: {0: 1, 4: 1, 6: 1},
    2: {1: 1, 3: -1},
    3: {0: 1, 4: -1, 6: -1},
}
# W: rows k*n+i, n=2, so row = 2*k + i
# c00 (k=0,i=0, row 0): M1 - M4 + M6 + M7
# c01 (k=0,i=1, row 1): M3 + M5
# c10 (k=1,i=0, row 2): M2 + M4
# c11 (k=1,i=1, row 3): M1 - M3 + M2 + M5
_W_strassen = {
    0: {0: 1, 3: -1, 5: 1, 6: 1},
    1: {2: 1, 4: 1},
    2: {1: 1, 3: 1},
    3: {0: 1, 2: -1, 1: 1, 4: 1},
}


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Decompose as: M_{2,4,8} = M_{2,2,2} (Strassen) tensor M_{1,2,4} (schoolbook)
    # This gives rank 7 * (1*2*4) = 7*8 = 56
    # But we need to verify the tensor structure maps correctly.
    
    # Actually, let's use a simpler approach:
    # M_{2,4,8} can be viewed as block structure.
    # A is 2x4, B is 4x8.
    # Split A into two 2x2 blocks: A = [A_left, A_right]
    # Split B into four 4x2 blocks: B = [B00, B01, B10, B11] (2x2 blocks of 2x2)
    # Wait, B is 4x8. Split into 2x4 blocks of 2x4? No.
    # Let's think of it differently.
    # A is 2x4, B is 4x8.
    # C = A @ B is 2x8.
    # 
    # Use Strassen on the 2x2 block structure:
    # A = [[A00, A01], [A10, A11]] where each Aij is 1x2 (since n=2, m=4, split into 2x2 blocks of size 1x2)
    # B = [[B00, B01], [B10, B11]] where each Bji is 2x4 (since m=4, p=8, split into 2x2 blocks of size 2x4)
    # C = [[C00, C01], [C10, C11]] where each Cki is 1x4
    #
    # Strassen applies to 2x2 matrix multiplication where each entry is a scalar.
    # Here, "entries" are submatrices, so we need the submatrix multiplication to be commutative 
    # or we need to be careful. Since submatrix multiplication is not generally commutative,
    # standard Strassen may not directly apply unless we can reorder.
    #
    # For 1x2 * 2x4 = 1x4, this is a row vector times a matrix, which is well-defined.
    # The products like (A00+A11) are 1x2, and (B00+B11) are 2x4, so (A00+A11)(B00+B11) is 1x4. Good.
    #
    # So we can use Strassen on the 2x2 block level!
    # The "field" here is the ring of 1x2 * 2x4 = 1x4 multiplications, but since all submatrix
    # products result in 1x4, and additions are element-wise, this works.
    
    # Strassen rank-7 for the 2x2 block structure:
    # U_block: 2x2 "matrix" of 1x2 blocks -> 4 rows (each 1x2), so 4*1 = 4 rows? No.
    # Actually, U has shape (n*m, rank) = (2*4, rank) = (8, rank).
    # 
    # Let me construct this properly.
    # 
    # Block indices: i in {0,1} for row of A (2 rows), j in {0,1} for column-block of A (2 blocks of 2 cols)
    # So A[i, j*2:(j+1)*2] is Aij, a 1x2 row.
    # 
    # B has m=4 rows, p=8 cols. Split into 2x2 blocks:
    # B[j*2:(j+1)*2, k*4:(k+1)*4] is Bjk, a 2x4 block. j in {0,1}, k in {0,1}.
    # 
    # C[i, k*4:(k+1)*4] = C[i,k], a 1x4 block.
    # 
    # Strassen:
    # M1 = (A00+A11)(B00+B11) -> 1x4
    # M2 = (A10+A11)B00 -> 1x4
    # M3 = A00(B01-B11) -> 1x4
    # M4 = A11(B10-B00) -> 1x4
    # M5 = (A00+A01)B11 -> 1x4
    # M6 = (A10-A00)(B00+B01) -> 1x4
    # M7 = (A01-A11)(B10+B11) -> 1x4
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M5
    #
    # Now, each Mj is a 1x4 vector computed as (1x2) * (2x4).
    # This is a standard matrix product of size 1x2 * 2x4.
    # The rank-1 decomposition of a 1x2 * 2x4 product has rank 2*4 = 8? No.
    # The tensor for 1x2 * 2x4: U has 1*2=2 rows, V has 2*4=8 rows, W has 4*1=4 rows.
    # Schoolbook rank = 1*2*4 = 8.
    #
    # So total rank = 7 (Strassen) * 8 (schoolbook for 1x2*2x4) = 56.
    
    rank_total = 56
    U = [[0] * rank_total for _ in range(n * m)]  # 8 rows
    V = [[0] * rank_total for _ in range(m * p)]  # 32 rows
    W = [[0] * rank_total for _ in range(p * n)]  # 16 rows
    
    # Strassen entries: each is a 1x2 * 2x4 product
    # For each Strassen term t in 0..6, and each schoolbook term s in 0..7:
    # col = t * 8 + s
    
    # Schoolbook for 1x2 * 2x4:
    # A_block is 1x2: entries at positions (0,0), (0,1) -> row index 0*2+0=0, 0*2+1=1
    # B_block is 2x4: entries at (j,k) for j in {0,1}, k in {0,1,2,3} -> row index j*4+k
    # C_block is 1x4: entries at (0,k) for k in {0,1,2,3} -> row index k*1+0 = k
    
    # Strassen U contributions (for A blocks):
    # Aij is at row i*2 + j_block in U, where j_block is the 2 columns.
    # Wait, A is 2x4. Row i (0 or 1), columns 0-3.
    # Block A00 = A[0, 0:2], A01 = A[0, 2:4]
    # Block A10 = A[1, 0:2], A11 = A[1, 2:4]
    # U row for A[i, col] is i*4 + col (since n*m = 2*4 = 8)
    # A00: rows 0, 1 (i=0, cols 0,1)
    # A01: rows 2, 3 (i=0, cols 2,3)
    # A10: rows 4, 5 (i=1, cols 0,1)
    # A11: rows 6, 7 (i=1, cols 2,3)
    
    # V rows for B blocks:
    # B is 4x8. Block Bjk is B[2j:2j+2, 4k:4k+4]
    # B00: rows 0-1, cols 0-3 -> V rows 0*8+0=0 to 1*8+3=11? No.
    # V row index = row * p + col = row * 8 + col
    # B00: V rows 0*8+0=0, 1*8+0=8, 0*8+1=1, 1*8+1=9, 0*8+2=2, 1*8+2=10, 0*8+3=3, 1*8+3=11
    # So B00 corresponds to V rows [0,1,2,3,8,9,10,11]
    # B01: cols 4-7 -> V rows [4,5,6,7,12,13,14,15]
    # B10: rows 2-3, cols 0-3 -> V rows [16,17,18,19,24,25,26,27]
    # B11: rows 2-3, cols 4-7 -> V rows [20,21,22,23,28,29,30,31]
    
    # W rows for C blocks:
    # C is 2x8. C[i, 4k:4k+4]
    # W row index = k*n + i = k*2 + i (since p*n = 8*2 = 16, but indexing is k*n+i where k is col-block, i is row)
    # Wait, W[k*n+i, r] where k is output col index, i is output row index.
    # C[i, col] -> W row = col * n + i = col * 2 + i
    # C00: i=0, cols 0-3 -> W rows 0,1,2,3 (col*2+0 = 0,2,4,6? No, col*2+0 for col=0,1,2,3 -> 0,2,4,6)
    # Hmm, let me re-read: W[k*n+i, r] where k is the column index of C, i is the row index.
    # So W row = k * n + i. For C[i, k], row = k*2 + i.
    # C00 block: i=0, k=0,1,2,3 -> W rows 0, 2, 4, 6
    # C01 block: i=0, k=4,5,6,7 -> W rows 8, 10, 12, 14
    # C10 block: i=1, k=0,1,2,3 -> W rows 1, 3, 5, 7
    # C11 block: i=1, k=4,5,6,7 -> W rows 9, 11, 13, 15
    
    # Strassen terms and their A and B block combinations:
    # t: (A_coefficients, B_coefficients)
    # A_coeffs: {block: coeff} where block is '00','01','10','11'
    # B_coeffs: {block: coeff} where block is '00','01','10','11'
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}),
        # M2: (A10+A11)B00
        ({'10': 1, '11': 1}, {'00': 1}),
        # M3: A00(B01-B11)
        ({'00': 1}, {'01': 1, '11': -1}),
        # M4: A11(B10-B00)
        ({'11': 1}, {'10': 1, '00': -1}),
        # M5: (A00+A01)B11
        ({'00': 1, '01': 1}, {'11': 1}),
        # M6: (A10-A00)(B00+B01)
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}),
        # M7: (A01-A11)(B10+B11)
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}),
    ]
    
    # C combination:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M5
    c_combinations = {
        '00': [(0, 1), (3, -1), (5, 1), (6, 1)],
        '01': [(2, 1), (4, 1)],
        '10': [(1, 1), (3, 1)],
        '11': [(0, 1), (2, -1), (1, 1), (4, 1)],
    }
    
    # A block to U rows:
    a_block_rows = {
        '00': [0, 1],
        '01': [2, 3],
        '10': [4, 5],
        '11': [6, 7],
    }
    
    # B block to V rows:
    b_block_rows = {
        '00': [0, 8, 1, 9, 2, 10, 3, 11],
        '01': [4, 12, 5, 13, 6, 14, 7, 15],
        '10': [16, 24, 17, 25, 18, 26, 19, 27],
        '11': [20, 28, 21, 29, 22, 30, 23, 31],
    }
    
    # For each Strassen term and schoolbook sub-term:
    for t_idx, (a_coeffs, b_coeffs) in enumerate(strassen_terms):
        for a_block, a_coeff in a_coeffs.items():
            for b_block, b_coeff in b_coeffs.items():
                # Schoolbook for 1x2 * 2x4: 8 terms
                # A_block is 1x2: a_row in {0,1} (local)
                # B_block is 2x4: b_row in {0,1}, b_col in {0,1,2,3} (local)
                # C_block is 1x4: c_col in {0,1,2,3} (local)
                for a_local in range(2):
                    for b_local_row in range(2):
                        for b_local_col in range(4):
                            col = t_idx * 8 + a_local * 8 + b_local_row * 4 + b_local_col
                            if col >= rank_total:
                                continue
                            # U: a_block_rows[a_block][a_local]
                            u_row = a_block_rows[a_block][a_local]
                            U[u_row][col] += a_coeff
                            # V: b_block_rows[b_block][b_local_row * 4 + b_local_col]
                            v_row = b_block_rows[b_block][b_local_row * 4 + b_local_col]
                            V[v_row][col] += b_coeff
                            # W: c_block is C[i, k*4 + b_local_col]
                            # For each C output block that uses this M term:
                            for c_block, comb in c_combinations.items():
                                for m_idx, w_coeff in comb:
                                    if m_idx == t_idx:
                                        # c_block determines i and k_base
                                        if c_block == '00':
                                            i, k_base = 0, 0
                                        elif c_block == '01':
                                            i, k_base = 0, 4
                                        elif c_block == '10':
                                            i, k_base = 1, 0
                                        else:  # '11'
                                            i, k_base = 1, 4
                                        k = k_base + b_local_col
                                        w_row = k * n + i
                                        W[w_row][col] += w_coeff
    
    return {"U": U, "V": V, "W": W}
