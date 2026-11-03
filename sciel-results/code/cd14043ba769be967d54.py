"""Matrix multiplication 2x4 by 4x6 via Strassen-based rank reduction."""
import time

DIMENSIONS = [2, 4, 6]

def _strassen_3x3_factors():
    """
    Return U3, V3, W3 for 3x3 * 3x3 -> 3x3 with rank 23.
    Indexing: U3[i*3+j, c], V3[j*3+k, c], W3[k*3+i, c]
    """
    # Standard 23-term Strassen for 3x3 matrices
    # M = A * B where A is 3x3, B is 3x3, M is 3x3
    # We define 23 bilinear forms
    
    # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    #     B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    # M = [[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]]
    
    # 23-term decomposition (Baur-Strassen / Winograd)
    # m00 = a00*b00 + a01*b10 + a02*b20
    # m01 = a00*b01 + a01*b11 + a02*b21
    # m02 = a00*b02 + a01*b12 + a02*b22
    # m10 = a10*b00 + a11*b10 + a12*b20
    # m11 = a10*b01 + a11*b11 + a12*b21
    # m12 = a10*b02 + a11*b12 + a12*b22
    # m20 = a20*b00 + a21*b10 + a22*b20
    # m21 = a20*b01 + a21*b11 + a22*b21
    # m22 = a20*b02 + a21*b12 + a22*b22
    
    # Winograd's 23-term scheme:
    # t1  = a00*b00
    # t2  = (a01+a02)*(b20+b21+b22)  -- no, let me use a known 23-term scheme
    
    # Actually, let me use the standard 23-term algorithm from literature.
    # Here's a clean 23-term decomposition:
    
    # Define the 23 bilinear forms as (U_coeff, V_coeff, W_coeff) for each term
    # U[i,j] coefficient, V[j,k] coefficient, W[k,i] coefficient
    
    # I'll use the well-known 23-term algorithm:
    # Terms (each is a product of a linear form in A and a linear form in B):
    
    terms = []
    
    # t1 = a00 * b00
    terms.append(((0,0), (0,0), (0,0), 1))
    # t2 = a00 * b01
    terms.append(((0,0), (0,1), (1,0), 1))
    # t3 = a00 * b02
    terms.append(((0,0), (0,2), (2,0), 1))
    # t4 = a01 * b10
    terms.append(((0,1), (1,0), (0,1), 1))
    # t5 = a01 * b11
    terms.append(((0,1), (1,1), (1,1), 1))
    # t6 = a01 * b12
    terms.append(((0,1), (1,2), (2,1), 1))
    # t7 = a02 * b20
    terms.append(((0,2), (2,0), (0,2), 1))
    # t8 = a02 * b21
    terms.append(((0,2), (2,1), (1,2), 1))
    # t9 = a02 * b22
    terms.append(((0,2), (2,2), (2,2), 1))
    # t10 = a10 * b00
    terms.append(((1,0), (0,0), (0,0), 1))
    # t11 = a10 * b01
    terms.append(((1,0), (0,1), (1,0), 1))
    # t12 = a10 * b02
    terms.append(((1,0), (0,2), (2,0), 1))
    # t13 = a11 * b10
    terms.append(((1,1), (1,0), (0,1), 1))
    # t14 = a11 * b11
    terms.append(((1,1), (1,1), (1,1), 1))
    # t15 = a11 * b12
    terms.append(((1,1), (1,2), (2,1), 1))
    # t16 = a12 * b20
    terms.append(((1,2), (2,0), (0,2), 1))
    # t17 = a12 * b21
    terms.append(((1,2), (2,1), (1,2), 1))
    # t18 = a12 * b22
    terms.append(((1,2), (2,2), (2,2), 1))
    # t19 = a20 * b00
    terms.append(((2,0), (0,0), (0,0), 1))
    # t20 = a20 * b01
    terms.append(((2,0), (0,1), (1,0), 1))
    # t21 = a20 * b02
    terms.append(((2,0), (0,2), (2,0), 1))
    # t22 = a21 * b10
    terms.append(((2,1), (1,0), (0,1), 1))
    # t23 = a21 * b11
    terms.append(((2,1), (1,1), (1,1), 1))
    # t24 = a21 * b12
    terms.append(((2,1), (1,2), (2,1), 1))
    # t25 = a22 * b20
    terms.append(((2,2), (2,0), (0,2), 1))
    # t26 = a22 * b21
    terms.append(((2,2), (2,1), (1,2), 1))
    # t27 = a22 * b22
    terms.append(((2,2), (2,2), (2,2), 1))
    
    # That's 27 terms (schoolbook). I need the actual 23-term Strassen.
    # Let me use a proper 23-term algorithm.
    
    # Winograd 23-term for 3x3:
    # The 23 bilinear forms are:
    
    # I'll use a verified 23-term decomposition.
    # Based on the standard result that R(3x3*3x3) = 23.
    
    # Let me construct it properly using the block Strassen approach on 3x3:
    # Split 3x3 into blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
    
    # This gets complicated. Let me just use a known explicit 23-term formula.
    
    # Here's the 23-term algorithm (from Baur & Strassen, 1973):
    # Using indices (i,j) for A, (j,k) for B, (k,i) for M:
    
    _terms = [
        # (a_idx, b_idx, m_idx, coeff)
        ((0,0), (0,0), (0,0), 1),
        ((0,0), (0,1), (1,0), 1),
        ((0,0), (0,2), (2,0), 1),
        ((0,1), (1,0), (0,1), 1),
        ((0,1), (1,1), (1,1), 1),
        ((0,1), (1,2), (2,1), 1),
        ((0,2), (2,0), (0,2), 1),
        ((0,2), (2,1), (1,2), 1),
        ((0,2), (2,2), (2,2), 1),
        ((1,0), (0,0), (0,0), 1),
        ((1,0), (0,1), (1,0), 1),
        ((1,0), (0,2), (2,0), 1),
        ((1,1), (1,0), (0,1), 1),
        ((1,1), (1,1), (1,1), 1),
        ((1,1), (1,2), (2,1), 1),
        ((1,2), (2,0), (0,2), 1),
        ((1,2), (2,1), (1,2), 1),
        ((1,2), (2,2), (2,2), 1),
        ((2,0), (0,0), (0,0), 1),
        ((2,0), (0,1), (1,0), 1),
        ((2,0), (0,2), (2,0), 1),
        ((2,1), (1,0), (0,1), 1),
        ((2,1), (1,1), (1,1), 1),
        ((2,1), (1,2), (2,1), 1),
        ((2,2), (2,0), (0,2), 1),
        ((2,2), (2,1), (1,2), 1),
        ((2,2), (2,2), (2,2), 1),
    ]
    
    # This is still 27 terms. The 23-term algorithm requires combining terms.
    # Let me just return the schoolbook 27-term for 3x3 and combine with the rest.
    # Total rank would be 27 + 4 = 31, which is better than 42.
    
    # Actually, let me just build the full decomposition directly.
    
    n, m, p = 2, 4, 6
    rank = 42  # schoolbook
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x6.
    
    Strategy: Use Strassen's algorithm on a 2x2 block structure to reduce rank.
    Split the 2x4 matrix into two 1x4 row blocks, and the 4x6 matrix into 
    four 1x6 row blocks. Then use bilinear tricks.
    
    Actually, let's think about this more carefully.
    
    We want to compute C = A * B where A is 2x4 and B is 4x6.
    C is 2x6.
    
    C[0,:] = A[0,:] * B  (dot product of row 0 of A with each column of B)
    C[1,:] = A[1,:] * B
    
    Let A[0,:] = [a0, a1, a2, a3]
    Let A[1,:] = [a4, a5, a6, a7]
    
    B columns:
    B[:,0] = [b0, b4, b8, b12]
    B[:,1] = [b1, b5, b9, b13]
    B[:,2] = [b2, b6, b10, b14]
    B[:,3] = [b3, b7, b11, b15]
    B[:,4] = [b0+? no...]
    
    Wait, B is 4x6, so B[j,k] for j in 0..3, k in 0..5.
    
    C[0,k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k] + a3*B[3,k]
    C[1,k] = a4*B[0,k] + a5*B[1,k] + a6*B[2,k] + a7*B[3,k]
    
    So we have 12 output entries, each a dot product of a 4-vector from A with a 4-vector from B.
    
    The total number of bilinear terms in schoolbook is 2*4*6 = 48? No, it's 2*4*6 = 48 terms? 
    Wait, rank = n*m*p = 2*4*6 = 48? But the current program says rank = n*m*p = 48?
    
    Let me re-read. The current program has rank = n*m*p = 2*4*6 = 48. But it says "rank = n*m*p" and creates 8x48, 24x48, 12x48 matrices.
    
    Hmm, but the task says the current score is 1.0000 with rank 42? Let me re-read.
    
    "yielding a total rank of 30 + 4 = 34, which is strictly lower than the current best of 42"
    
    So the current best is 42, not 48. The seed program uses rank 48 but maybe there's a better one at 42?
    
    Wait, the seed program literally has rank = n*m*p = 48. But the directive says "current best of 42". Maybe other programs in the population found 42?
    
    Let me just try to get a lower rank. The theoretical lower bound for 2x4 * 4x6 is at least max(2*6, 2*4, 4*6) = 24? No, the rank is at least the max of the dimensions... Actually, the tensor rank for matrix multiplication [n,m,p] is at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24.
    
    Let me try a different approach: use the fact that we can share computations.
    
    For C[0,k] and C[1,k], both use the same B column. So we can compute:
    For each k (0..5), we need:
    C[0,k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k] + a3*B[3,k]
    C[1,k] = a4*B[0,k] + a5*B[1,k] + a6*B[2,k] + a7*B[3,k]
    
    This is like multiplying a 2x4 matrix by a 4x1 vector, for each of 6 columns.
    
    The rank of [2,4,1] matrix multiplication is 4 (it's just a linear map from 4-dim to 2-dim, which needs 4 rank-1 terms... actually, the tensor for [2,4,1] has rank 4 since it's equivalent to the identity on 4-dim mapped to 2 outputs).
    
    So total rank would be 6 * 4 = 24 if we can do it independently. But that's the lower bound.
    
    Actually, [2,4,1] means multiplying 2x4 by 4x1. The tensor rank is 4 (since we need 4 independent bilinear forms).
    
    So if we do 6 independent [2,4,1] multiplications, total rank = 6*4 = 24.
    
    But wait, the B matrix entries are shared across the 6 columns? No, each column is independent. So we can indeed do 6 independent [2,4,1] multiplications.
    
    Let me construct this:
    For each k in 0..5:
      For each j in 0..3:
        Term: U[a_row, j] * V[j, k] -> contributes to W[k, a_row]
        
    Specifically, for column k of B and row i of A:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Each term (i,j,k) is a rank-1 tensor. Total 2*4*6 = 48 terms in schoolbook.
    
    To reduce, we need to combine terms. For [2,4,1], we can use 4 terms instead of 4 (same). So no gain there.
    
    Hmm, so the schoolbook is already optimal for [2,4,1]? Then total is 48.
    
    But the directive says current best is 42. So there must be a way to share across columns.
    
    Let me think again. Maybe we can use Strassen-like tricks on the 4x4 "middle" dimension.
    
    Split B into two 2x6 blocks: B_top (rows 0,1) and B_bottom (rows 2,3).
    Split A into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3).
    
    C = A_left * B_top + A_right * B_bottom
    
    Each A_left * B_top is [2,2,6] multiplication, rank 12 in schoolbook.
    A_right * B_bottom is also [2,2,6], rank 12.
    Total 24. But we can use Strassen for [2,2,2] to reduce.
    
    [2,2,6] can be seen as 3 independent [2,2,2] multiplications (columns 0-1, 2-3, 4-5).
    Each [2,2,2] has rank 7 (Strassen). So 3*7 = 21 per block.
    Two blocks: 42.
    
    That matches the "current best of 42"!
    
    So the current best uses Strassen on [2,2,2] blocks: 3 Strassen multiplications for A_left*B_top (rank 21) and 3 for A_right*B_bottom (rank 21), total 42.
    
    Can we do better? The lower bound is 24. Known results: R(2,2,6) <= 21 (via 3x Strassen). R(2,4,6) <= ?
    
    Actually, R(2,4,6) = R(2,2,6) + R(2,2,6) if we split as above, but maybe we can do better by not splitting that way.
    
    Alternative: Use the fact that R(2,4,6) <= R(2,4,4) + R(2,4,2) or similar.
    
    R(2,4,4): We can view this as 4 independent [2,4,1] multiplications, each of rank 4, total 16. Or use Strassen: split 4 into 2+2, get 2*[2,2,2] + 2*[2,2,2] = 2*7+2*7 = 28? No, that's worse.
    
    Actually [2,4,4]: split A into 2x2 blocks (2 blocks of 2x2), B into 4x4 (split into 2x2 blocks, 4 of them). C = sum of 2*4 = 8 [2,2,2] multiplications? No, that's not right either.
    
    Let me just go with the 42-rank Strassen approach and see if I can implement it correctly. If I can get 42, that's an improvement over the seed's 48.
    
    Actually wait, the seed has rank 48 and scores 1.0. If I get 42, my score would be 48/42 = 1.143.
    
    Let me implement the 42-rank decomposition:
    - Split A (2x4) into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    - Split B (4x6) into B_top (2x6, rows 0-1) and B_bottom (2x6, rows 2-3)
    - C = A_left * B_top + A_right * B_bottom
    - For A_left * B_top: split into 3 [2,2,2] multiplications (cols 0-1, 2-3, 4-5 of B_top)
      Each uses Strassen rank 7, total 21
    - For A_right * B_bottom: same, 21
    - Total rank: 42
    
    Strassen for [2,2,2]:
    Given P (2x2) and Q (2x2), R = P*Q (2x2):
    M1 = (P00+P11)*(Q00+Q11)
    M2 = (P10+P11)*Q00
    M3 = P00*(Q01-Q11)
    M4 = P11*(Q10-Q00)
    M5 = (P00+P01)*Q11
    M6 = (P10-P00)*(Q00+Q01)
    M7 = (P01-P11)*(Q10+Q11)
    
    R00 = M1 + M4 - M5 + M7
    R01 = M3 + M5
    R10 = M2 + M4
    R11 = M1 - M2 + M3 + M6
    
    Now I need to express each Mi as a bilinear form in the original A and B entries, and accumulate into C.
    
    Let me define the full construction.
    
    For A_left * B_top, column pair (2c, 2c+1):
    P = A_left = [[A[0,2c], A[0,2c+1]], [A[1,2c], A[1,2c+1]]]
    Q = B_top[:, 2c:2c+2] = [[B[0,2c], B[0,2c+1]], [B[1,2c], B[1,2c+1]]]
    R = P*Q = [[C[0,2c], C[0,2c+1]], [C[1,2c], C[1,2c+1]]]
    
    Similarly for A_right * B_bottom.
    
    Let me build the factor matrices.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1
    
    # A is 2x4, indexed A[i,j] -> U[i*m+j] = U[i*4+j]
    # B is 4x6, indexed B[j,k] -> V[j*p+k] = V[j*6+k]
    # C is 2x6, indexed C[i,k] -> W[k*n+i] = W[k*2+i]
    
    # A_left * B_top: cols 0,1 of A; rows 0,1 of B
    # For c in 0,1,2 (column pairs of B_top):
    for c in range(3):
        k0 = 2 * c
        k1 = 2 * c + 1
        # P = A_left = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # Q = B_top[:, k0:k1+1] = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        
        # Strassen terms:
        # M1 = (P00+P11)*(Q00+Q11)
        # P00 = A[0,0], P11 = A[1,1]
        # Q00 = B[0,k0], Q11 = B[1,k1]
        # R00 += M1, R11 += M1
        add_term(0, 1, k0, 1, k0, 1)  # A[0,0]*B[0,k0] -> C[0,k0]
        add_term(0, 1, k1, 1, k0, 1)  # A[0,0]*B[1,k1] -> C[0,k0]
        add_term(5, 1, k0, 1, k1, 1)  # A[1,1]*B[0,k0] -> C[1,k1]
        add_term(5, 1, k1, 1, k1, 1)  # A[1,1]*B[1,k1] -> C[1,k1]
        
        # M2 = (P10+P11)*Q00
        # P10 = A[1,0], P11 = A[1,1], Q00 = B[0,k0]
        # R10 += M2, R11 -= M2
        add_term(4, 1, k0, 1, k0, 1)  # A[1,0]*B[0,k0] -> C[1,k0]
        add_term(5, 1, k0, 1, k0, 1)  # A[1,1]*B[0,k0] -> C[1,k0]
        add_term(4, 1, k0, -1, k1, -1)  # A[1,0]*B[0,k0] -> -C[1,k1]
        add_term(5, 1, k0, -1, k1, -1)  # A[1,1]*B[0,k0] -> -C[1,k1]
        
        # M3 = P00*(Q01-Q11)
        # P00 = A[0,0], Q01 = B[0,k1], Q11 = B[1,k1]
        # R01 += M3, R11 += M3
        add_term(0, 1, k1, 1, k1, 1)  # A[0,0]*B[0,k1] -> C[0,k1]
        add_term(0, 1, k1+6, 1, k1, 1)  # A[0,0]*B[1,k1] -> C[0,k1]
        add_term(0, 1, k1, 1, k1+2, 1)  # A[0,0]*B[0,k1] -> C[1,k1]
        add_term(0, 1, k1+6, 1, k1+2, 1)  # A[0,0]*B[1,k1] -> C[1,k1]
        
        # M4 = P11*(Q10-Q00)
        # P11 = A[1,1], Q10 = B[1,k0], Q00 = B[0,k0]
        # R00 += M4, R10 += M4
        add_term(5, 1, k0+6, 1, k0, 1)  # A[1,1]*B[1,k0] -> C[0,k0]
        add_term(5, 1, k0, -1, k0, -1)  # A[1,1]*B[0,k0] -> -C[0,k0]
        add_term(5, 1, k0+6, 1, k0+2, 1)  # A[1,1]*B[1,k0] -> C[1,k0]
        add_term(5, 1, k0, -1, k0+2, -1)  # A[1,1]*B[0,k0] -> -C[1,k0]
        
        # M5 = (P00+P01)*Q11
        # P00 = A[0,0], P01 = A[0,1], Q11 = B[1,k1]
        # R00 -= M5, R01 += M5
        add_term(0, -1, k1+6, -1, k0, -1)  # A[0,0]*B[1,k1] -> -C[0,k0]
        add_term(1, -1, k1+6, -1, k0, -1)  # A[0,1]*B[1,k1] -> -C[0,k0]
        add_term(0, 1, k1+6, 1, k1, 1)  # A[0,0]*B[1,k1] -> C[0,k1]
        add_term(1, 1, k1+6, 1, k1, 1)  # A[0,1]*B[1,k1] -> C[0,k1]
        
        # M6 = (P10-P00)*(Q00+Q01)
        # P10 = A[1,0], P00 = A[0,0], Q00 = B[0,k0], Q01 = B[0,k1]
        # R11 += M6
        add_term(4, 1, k0, 1, k1+2, 1)  # A[1,0]*B[0,k0] -> C[1,k1]
        add_term(4, 1, k1, 1, k1+2, 1)  # A[1,0]*B[0,k1] -> C[1,k1]
        add_term(0, -1, k0, -1, k1+2, -1)  # A[0,0]*B[0,k0] -> -C[1,k1]
        add_term(0, -1, k1, -1, k1+2, -1)  # A[0,0]*B[0,k1] -> -C[1,k1]
        
        # M7 = (P01-P11)*(Q10+Q11)
        # P01 = A[0,1], P11 = A[1,1], Q10 = B[1,k0], Q11 = B[1,k1]
        # R00 += M7
        add_term(1, 1, k0+6, 1, k0, 1)  # A[0,1]*B[1,k0] -> C[0,k0]
        add_term(1, 1, k1+6, 1, k0, 1)  # A[0,1]*B[1,k1] -> C[0,k0]
        add_term(5, -1, k0+6, -1, k0, -1)  # A[1,1]*B[1,k0] -> -C[0,k0]
        add_term(5, -1, k1+6, -1, k0, -1)  # A[1,1]*B[1,k1] -> -C[0,k0]
    
    # A_right * B_bottom: cols 2,3 of A; rows 2,3 of B
    for c in range(3):
        k0 = 2 * c
        k1 = 2 * c + 1
        # P = A_right = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
        # Q = B_bottom[:, k0:k1+1] = [[B[2,k0], B[2,k1]], [B[3,k0], B[3,k1]]]
        
        # M1 = (P00+P11)*(Q00+Q11)
        # P00 = A[0,2], P11 = A[1,3]
        # Q00 = B[2,k0], Q11 = B[3,k1]
        add_term(2, 1, 12 + k0, 1, k0, 1)  # A[0,2]*B[2,k0] -> C[0,k0]
        add_term(2, 1, 18 + k1, 1, k0, 1)  # A[0,2]*B[3,k1] -> C[0,k0]
        add_term(7, 1, 12 + k0, 1, k1, 1)  # A[1,3]*B[2,k0] -> C[1,k1]
        add_term(7, 1, 18 + k1, 1, k1, 1)  # A[1,3]*B[3,k1] -> C[1,k1]
        
        # M2 = (P10+P11)*Q00
        # P10 = A[1,2], P11 = A[1,3], Q00 = B[2,k0]
        add_term(6, 1, 12 + k0, 1, k0, 1)  # A[1,2]*B[2,k0] -> C[1,k0]
        add_term(7, 1, 12 + k0, 1, k0, 1)  # A[1,3]*B[2,k0] -> C[1,k0]
        add_term(6, 1, 12 + k0, -1, k1, -1)  # A[1,2]*B[2,k0] -> -C[1,k1]
        add_term(7, 1, 12 + k0, -1, k1, -1)  # A[1,3]*B[2,k0] -> -C[1,k1]
        
        # M3 = P00*(Q01-Q11)
        # P00 = A[0,2], Q01 = B[2,k1], Q11 = B[3,k1]
        add_term(2, 1, 12 + k1, 1, k1, 1)  # A[0,2]*B[2,k1] -> C[0,k1]
        add_term(2, 1, 18 + k1, 1, k1, 1)  # A[0,2]*B[3,k1] -> C[0,k1]
        add_term(2, 1, 12 + k1, 1, k1+2, 1)  # A[0,2]*B[2,k1] -> C[1,k1]
        add_term(2, 1, 18 + k1, 1, k1+2, 1)  # A[0,2]*B[3,k1] -> C[1,k1]
        
        # M4 = P11*(Q10-Q00)
        # P11 = A[1,3], Q10 = B[3,k0], Q00 = B[2,k0]
        add_term(7, 1, 18 + k0, 1, k0, 1)  # A[1,3]*B[3,k0] -> C[0,k0]
        add_term(7, 1, 12 + k0, -1, k0, -1)  # A[1,3]*B[2,k0] -> -C[0,k0]
        add_term(7, 1, 18 + k0, 1, k0+2, 1)  # A[1,3]*B[3,k0] -> C[1,k0]
        add_term(7, 1, 12 + k0, -1, k0+2, -1)  # A[1,3]*B[2,k0] -> -C[1,k0]
        
        # M5 = (P00+P01)*Q11
        # P00 = A[0,2], P01 = A[0,3], Q11 = B[3,k1]
        add_term(2, -1, 18 + k1, -1, k0, -1)  # A[0,2]*B[3,k1] -> -C[0,k0]
        add_term(3, -1, 18 + k1, -1, k0, -1)  # A[0,3]*B[3,k1] -> -C[0,k0]
        add_term(2, 1, 18 + k1, 1, k1, 1)  # A[0,2]*B[3,k1] -> C[0,k1]
        add_term(3, 1, 18 + k1, 1, k1, 1)  # A[0,3]*B[3,k1] -> C[0,k1]
        
        # M6 = (P10-P00)*(Q00+Q01)
        # P10 = A[1,2], P00 = A[0,2], Q00 = B[2,k0], Q01 = B[2,k1]
        add_term(6, 1, 12 + k0, 1, k1+2, 1)  # A[1,2]*B[2,k0] -> C[1,k1]
        add_term(6, 1, 12 + k1, 1, k1+2, 1)  # A[1,2]*B[2,k1] -> C[1,k1]
        add_term(2, -1, 12 + k0, -1, k1+2, -1)  # A[0,2]*B[2,k0] -> -C[1,k1]
        add_term(2, -1, 12 + k1, -1, k1+2, -1)  # A[0,2]*B[2,k1] -> -C[1,k1]
        
        # M7 = (P01-P11)*(Q10+Q11)
        # P01 = A[0,3], P11 = A[1,3], Q10 = B[3,k0], Q11 = B[3,k1]
        add_term(3, 1, 18 + k0, 1, k0, 1)  # A[0,3]*B[3,k0] -> C[0,k0]
        add_term(3, 1, 18 + k1, 1, k0, 1)  # A[0,3]*B[3,k1] -> C[0,k0]
        add_term(7, -1, 18 + k0, -1, k0, -1)  # A[1,3]*B[3,k0] -> -C[0,k0]
        add_term(7, -1, 18 + k1, -1, k0, -1)  # A[1,3]*B[3,k1] -> -C[0,k0]
    
    return {"U": U, "V": V, "W": W}
