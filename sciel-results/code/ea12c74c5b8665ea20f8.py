import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # We are computing C = A @ B where A is (n x m) = (2 x 3) and B is (m x p) = (3 x 7)
    # C is (n x p) = (2 x 7)
    # Tensor T_{i, j, k} = C[i, k] = sum_j A[i, j] B[j, k]
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (21, rank)
    # W shape: (p*n, rank) = (14, rank)
    # Note: The prompt says U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Indices:
    # U index: i*m + j (0..5)
    # V index: j*p + k (0..20)
    # W index: k*n + i (0..13)  <-- "official trace convention W[k*n+i,r]"
    
    # We will construct a rank 39 decomposition.
    # Structure:
    # 1. Decompose the "main" 2x2 part of A (cols 0,1) and B (rows 0,1) for k=0..5 using Strassen blocks (2x2x2).
    #    There are 3 such blocks for k=0,1; k=2,3; k=4,5.
    #    Each 2x2x2 block uses 7 terms. Total 21 terms.
    #    This covers A[0:2, 0:2] and B[0:2, 0:6].
    #    It contributes to C[0:2, 0:6] via sum_{j in 0,1} A[i,j]B[j,k].
    
    # 2. Handle the remaining parts:
    #    a) A[0:2, 2] (col 2 of A) with B[0:2, 0:6] (rows 0,1 of B, cols 0..5).
    #       This is a 2x1 (A part) times 2x6 (B part) -> 2x6 output contribution?
    #       Wait, A is 2x3. Let's split A into A_left (2x2, cols 0,1) and A_right (2x1, col 2).
    #       Let's split B into B_top (2x7, rows 0,1) and B_bottom (1x7, row 2).
    #       C = A_left B_top + A_left B_bottom + A_right B_top + A_right B_bottom.
    #       
    #       We used Strassen on A_left B_top for the k=0..5 part?
    #       Strassen on A_left (2x2) and B_top_block (2x2) gives 2x2 output.
    #       We applied 3 Strassen blocks to B_top for columns (0,1), (2,3), (4,5).
    #       So A_left B_top[:, 0:6] is covered.
    #       
    #       Remaining terms:
    #       - A_left B_top[:, 6]: 2x2 * 2x1 -> 2x1 output. Rank 4.
    #       - A_left B_bottom: 2x2 * 1x7 -> 2x7 output.
    #         B_bottom is a row vector b (1x7). A_left is 2x2.
    #         C += A_left @ b.
    #         This is equivalent to:
    #         C[0, k] += A[0,0]b[0,k] + A[0,1]b[1,k]
    #         C[1, k] += A[1,0]b[0,k] + A[1,1]b[1,k]
    #         This can be done with 4 terms? No, it's rank 4 for each k?
    #         Actually, A_left @ b = (A_left[:,0] * b[0]) + (A_left[:,1] * b[1])? No.
    #         A_left @ b = sum_j A_left[:, j] b[j, :].
    #         Since b is 1x7, b[j, :] is the row j of B_bottom.
    #         So C += A[0:2, 0] B[2, :] + A[0:2, 1] B[2, :].
    #         This is 2x1 * 1x7 + 2x1 * 1x7.
    #         Rank is at most 4?
    #         Let x = A[:, 0] (2x1), y = A[:, 1] (2x1).
    #         Let u = B[2, :] (1x7).
    #         C += x u + y u = (x+y) u.
    #         This is rank 1!
    #         Wait, A[0:2, 0] is a column vector a0. A[0:2, 1] is a column vector a1.
    #         B[2, :] is a row vector b2.
    #         Contribution: a0 b2 + a1 b2 = (a0+a1) b2.
    #         Yes, this is rank 1.
    #         So A_left B_bottom requires 1 term.
    #       
    #       - A_right B_top: A_right is 2x1 (col 2 of A). B_top is 2x7.
    #         C += A[:, 2] B[0:2, :].
    #         Let v = A[:, 2] (2x1).
    #         C += v (B[0, :] + B[1, :])?
    #         C[i, k] += A[i, 2] B[0, k] + A[i, 2] B[1, k] = A[i, 2] (B[0, k] + B[1, k]).
    #         This is rank 1! (Vector A[:,2] outer sum of rows 0,1 of B).
    #         So A_right B_top requires 1 term.
    #       
    #       - A_right B_bottom: A_right is 2x1, B_bottom is 1x7.
    #         C += A[:, 2] B[2, :].
    #         This is rank 1. (Outer product of A[:,2] and B[2,:]).
    #         So A_right B_bottom requires 1 term.
    
    # Let's re-evaluate the counts.
    # Total Rank = (Strassen for A_left B_top[:, 0:6]) + (A_left B_top[:, 6]) + (A_left B_bottom) + (A_right B_top) + (A_right B_bottom)
    #
    # 1. A_left (2x2) * B_top[:, 0:6] (2x6).
    #    We split B_top[:, 0:6] into three 2x2 blocks: cols (0,1), (2,3), (4,5).
    #    Each 2x2 * 2x2 uses Strassen (7 terms).
    #    Total = 3 * 7 = 21 terms.
    #    
    # 2. A_left (2x2) * B_top[:, 6] (2x1).
    #    This is 2x2 * 2x1. Rank 4 (schoolbook).
    #    Terms:
    #    C[0, 6] += A[0,0]B[0,6] + A[0,1]B[1,6]
    #    C[1, 6] += A[1,0]B[0,6] + A[1,1]B[1,6]
    #    4 terms.
    #
    # 3. A_left (2x2) * B_bottom (1x7).
    #    C[i, k] += A[i, 0]B[2, k] + A[i, 1]B[2, k] = (A[i, 0] + A[i, 1]) B[2, k].
    #    Let w = A[:, 0] + A[:, 1] (2x1 vector).
    #    Let u = B[2, :] (1x7 row vector).
    #    C += w u^T.
    #    This is rank 1.
    #    1 term.
    #
    # 4. A_right (2x1) * B_top (2x7).
    #    C[i, k] += A[i, 2] B[0, k] + A[i, 2] B[1, k] = A[i, 2] (B[0, k] + B[1, k]).
    #    Let v = A[:, 2] (2x1 vector).
    #    Let s = B[0, :] + B[1, :] (1x7 row vector).
    #    C += v s^T.
    #    This is rank 1.
    #    1 term.
    #
    # 5. A_right (2x1) * B_bottom (1x7).
    #    C[i, k] += A[i, 2] B[2, k].
    #    Let v = A[:, 2] (2x1 vector).
    #    Let u = B[2, :] (1x7 row vector).
    #    C += v u^T.
    #    This is rank 1.
    #    1 term.
    
    # Total Rank = 21 + 4 + 1 + 1 + 1 = 28.
    
    # Is this valid?
    # Let's check the indices carefully.
    # A is 2x3. Indices (i, j) -> i*m + j.
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2
    #   A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B is 3x7. Indices (j, k) -> j*p + k.
    #   B[0,k] = k
    #   B[1,k] = 7+k
    #   B[2,k] = 14+k
    # C is 2x7. Indices (i, k) -> k*n + i.
    #   C[0,k] = 2k
    #   C[1,k] = 2k+1
    
    # Strassen Block for A[0:2, 0:2] and B[0:2, k0:k0+2] -> C[0:2, k0:k0+2]
    # A00=0, A01=1, A10=3, A11=4
    # B00=7*0+k0, B01=7*0+k0+1, B10=7*1+k0, B11=7*1+k0+1
    # C00=2*k0+0, C01=2*k0+1, C10=2*k0+2, C11=2*k0+3
    
    # Wait, C indices:
    # k0: C[0,k0] is 2*k0. C[1,k0] is 2*k0+1.
    # k0+1: C[0,k0+1] is 2*k0+2. C[1,k0+1] is 2*k0+3.
    
    # My previous seed used:
    # c00 = k_start * 2 + 0
    # c01 = (k_start + 1) * 2 + 0
    # c10 = k_start * 2 + 1
    # c11 = (k_start + 1) * 2 + 1
    # This matches.
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return # or raise
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # 1. Strassen Blocks for k=0,1; 2,3; 4,5
    def add_strassen(k_start):
        # A indices
        a00, a01, a10, a11 = 0, 1, 3, 4
        # B indices
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        # C indices
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

    add_strassen(0)
    add_strassen(2)
    add_strassen(4)
    
    # 2. A_left * B_top[:, 6] (2x2 * 2x1)
    # C[0, 6] += A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1, 6] += A[1,0]B[0,6] + A[1,1]B[1,6]
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    # Term 1: A00 B06 -> C06
    add_term([0], [1], [b06], [1], [c06], [1])
    # Term 2: A01 B16 -> C06
    add_term([1], [1], [b16], [1], [c06], [1])
    # Term 3: A10 B06 -> C16
    add_term([3], [1], [b06], [1], [c16], [1])
    # Term 4: A11 B16 -> C16
    add_term([4], [1], [b16], [1], [c16], [1])
    
    # 3. A_left * B_bottom (Rank 1)
    # C[i, k] += (A[i,0] + A[i,1]) B[2, k]
    # U part: A[:,0] + A[:,1] -> indices 0, 1 with coeff 1.
    # V part: B[2, :] -> indices 14..20 with coeff 1.
    # W part: C[:, k] -> indices 2k, 2k+1 with coeff 1.
    # This is one term?
    # U vector: [1, 1, 0, 0, 0, 0] (at indices 0,1)
    # V vector: [0, ..., 1, ..., 1] (at indices 14..20)
    # W vector: [1, 1, 1, 1, ..., 1] (all 14 indices)
    # Yes, this is a single rank-1 term.
    # U indices: 0, 1. Coeff: 1, 1.
    # V indices: 14, 15, 16, 17, 18, 19, 20. Coeff: 1, 1, 1, 1, 1, 1, 1.
    # W indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13. Coeff: 1..1.
    
    u_idx = [0, 1]
    u_val = [1, 1]
    v_idx = [14, 15, 16, 17, 18, 19, 20]
    v_val = [1, 1, 1, 1, 1, 1, 1]
    w_idx = list(range(14))
    w_val = [1] * 14
    add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    # 4. A_right * B_top (Rank 1)
    # C[i, k] += A[i, 2] (B[0, k] + B[1, k])
    # U part: A[:, 2] -> indices 2, 5 with coeff 1.
    # V part: B[0, :] + B[1, :] -> indices 0..6 and 7..13 with coeff 1.
    # W part: C[:, k] -> all 14 indices with coeff 1.
    
    u_idx = [2, 5]
    u_val = [1, 1]
    v_idx = list(range(7)) + list(range(7, 14))
    v_val = [1] * 14
    w_idx = list(range(14))
    w_val = [1] * 14
    add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    # 5. A_right * B_bottom (Rank 1)
    # C[i, k] += A[i, 2] B[2, k]
    # U part: A[:, 2] -> indices 2, 5 with coeff 1.
    # V part: B[2, :] -> indices 14..20 with coeff 1.
    # W part: C[:, k] -> all 14 indices with coeff 1.
    
    u_idx = [2, 5]
    u_val = [1, 1]
    v_idx = [14, 15, 16, 17, 18, 19, 20]
    v_val = [1, 1, 1, 1, 1, 1, 1]
    w_idx = list(range(14))
    w_val = [1] * 14
    add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
