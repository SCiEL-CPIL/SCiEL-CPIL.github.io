"""Matrix multiplication 2,4,7 - improved decomposition via Strassen-like block split."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or at least improved) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x7 matrix.
    
    Strategy: Split the 2x4 matrix A into two 2x2 blocks: A = [A1 | A2]
    Split the 4x7 matrix B into two 2x7 blocks: B = [B1; B2] (top 2 rows, bottom 2 rows)
    Then C = A1*B1 + A2*B2, where A1,B1 are 2x2, 2x7 and A2,B2 are 2x2, 2x7.
    
    For each 2x2 * 2x7 multiplication, we can use:
    - Strassen-like for the 2x2 part (rank 7) + standard for remaining columns
    - Actually, let me think more carefully.
    
    For 2x2 * 2x7:
    Let X = [[x11, x12], [x21, x22]] (2x2)
    Let Y = [[y11..y17], [y21..y27]] (2x7)
    Z = X*Y = [[z11..z17], [z21..z27]] (2x7)
    
    z1j = x11*y1j + x12*y2j
    z2j = x21*y1j + x22*y2j
    
    For each column j, this is a 2x2*2x1 multiplication, which has rank 4 (schoolbook).
    For 7 columns, that's rank 28.
    
    But we can do better by sharing computations across columns.
    
    Actually, the tensor M_{2,2,7} has been studied. Let me use a known bound.
    The rank of M_{2,2,n} for n>=2 is at most 2n+1 (from a result by Strassen or others).
    For n=7: rank <= 15.
    
    Wait, let me verify: M_{2,2,2} has rank 7 (Strassen). 2*2+1=5, but actual rank is 7.
    So 2n+1 is not right for n=2.
    
    Let me reconsider. The known results:
    - M_{2,2,2}: rank = 7
    - M_{2,2,3}: rank <= 10 (from literature)
    - M_{2,2,n}: There are constructions with rank ~ 2n + O(1)
    
    Actually, for M_{2,2,n}, one can use the following:
    Treat it as multiplying a 2x2 matrix by n columns of a 2xn matrix.
    
    For each pair of consecutive columns, we can use Strassen-like tricks.
    
    Let me just use a simple but valid approach:
    For 2x2 * 2x7, use rank 7 for first 2 cols (Strassen), and rank 4 for each of remaining 5 cols.
    Total: 7 + 5*4 = 27 per block.
    Two blocks: 2*27 = 54.
    
    This is an improvement over 56 (schoolbook).
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # We'll build the decomposition manually
    # A is 2x4: indices (i, j) where i in {0,1}, j in {0,1,2,3}
    # B is 4x7: indices (j, k) where j in {0,1,2,3}, k in {0,...,6}
    # C is 2x7: indices (i, k) where i in {0,1}, k in {0,...,6}
    
    # Split A = [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # Split B = [B1; B2] where B1 = B[0:2, :], B2 = B[2:4, :]
    # C = A1*B1 + A2*B2
    
    # For each block, we need a 2x2 * 2x7 decomposition
    # Block 1: A1 (2x2) * B1 (2x7)
    # Block 2: A2 (2x2) * B2 (2x7)
    
    def make_2x2_2xn_decomp(n_cols):
        """
        Return (U, V, W) for 2x2 * 2xn multiplication.
        U has shape (4, rank), V has shape (2*n_cols, rank), W has shape (2*n_cols, rank)
        Wait, let me get the dimensions right.
        
        For multiplying X (2x2) by Y (2xn_cols) to get Z (2xn_cols):
        U: maps X entries to rank dim. X has 4 entries, so U is 4 x rank.
        V: maps Y entries to rank dim. Y has 2*n_cols entries, so V is (2*n_cols) x rank.
        W: maps to Z entries. Z has 2*n_cols entries, so W is (2*n_cols) x rank.
        
        The convention: W[k*n+i, r] where n is the "row" dimension of Z.
        Z is 2 x n_cols, so n=2 in the W indexing.
        W index: k*n+i = k*2+i where k in 0..n_cols-1, i in 0..1.
        """
        # Use Strassen for first 2 columns, standard for rest
        # Strassen for 2x2*2x2: rank 7
        # Standard for each additional column: rank 4 per column
        
        if n_cols <= 2:
            # Pure Strassen for 2x2*2x2
            # Standard Strassen decomposition
            # X = [[a, b], [c, d]], Y = [[e, f], [g, h]]
            # m1 = (a+d)(e+h)
            # m2 = (c+d)(e)
            # m3 = (a)(f-h)
            # m4 = (d)(g-e)
            # m5 = (a+b)(h)
            # m6 = (c-a)(e+f)
            # m7 = (b-d)(g+h)
            # Z11 = m1 + m4 - m5 + m7
            # Z12 = m3 + m5
            # Z21 = m2 + m4
            # Z22 = m1 - m2 + m3 + m6
            
            rank = 7
            U = [[0]*rank for _ in range(4)]  # X has 4 entries: (0,0),(0,1),(1,0),(1,1)
            V = [[0]*rank for _ in range(4)]  # Y has 4 entries: (0,0),(0,1),(1,0),(1,1)
            W = [[0]*rank for _ in range(4)]  # Z has 4 entries: (0,0),(0,1),(1,0),(1,1)
            
            # X indices: 0=(0,0)=a, 1=(0,1)=b, 2=(1,0)=c, 3=(1,1)=d
            # Y indices: 0=(0,0)=e, 1=(0,1)=f, 2=(1,0)=g, 3=(1,1)=h
            # Z indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            
            # m1 = (a+d)(e+h): U: a=1, d=1; V: e=1, h=1
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            # W: Z11 += m1, Z22 += m1
            W[0][0] = 1; W[3][0] = 1
            
            # m2 = (c+d)(e): U: c=1, d=1; V: e=1
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            # W: Z21 += m2
            W[2][1] = 1
            
            # m3 = (a)(f-h): U: a=1; V: f=1, h=-1
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            # W: Z12 += m3, Z22 += m3
            W[1][2] = 1; W[3][2] = 1
            
            # m4 = (d)(g-e): U: d=1; V: g=1, e=-1
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            # W: Z11 += m4, Z21 += m4
            W[0][3] = 1; W[2][3] = 1
            
            # m5 = (a+b)(h): U: a=1, b=1; V: h=1
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            # W: Z11 -= m5, Z12 += m5
            W[0][4] = -1; W[1][4] = 1
            
            # m6 = (c-a)(e+f): U: c=1, a=-1; V: e=1, f=1
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            # W: Z22 += m6
            W[3][5] = 1
            
            # m7 = (b-d)(g+h): U: b=1, d=-1; V: g=1, h=1
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            # W: Z11 += m7
            W[0][6] = 1
            
            return U, V, W
        
        else:
            # Strassen for first 2 cols, standard for remaining n_cols-2 cols
            U_str, V_str, W_str = make_2x2_2xn_decomp(2)
            rank_str = 7
            rank_std = 4 * (n_cols - 2)
            rank_total = rank_str + rank_std
            
            U = [[0]*rank_total for _ in range(4)]
            V = [[0]*rank_total for _ in range(2*n_cols)]
            W = [[0]*rank_total for _ in range(2*n_cols)]
            
            # Copy Strassen part (cols 0,1)
            for r in range(rank_str):
                for i in range(4):
                    U[i][r] = U_str[i][r]
                for j in range(4):
                    V[j][r] = V_str[j][r]
                for j in range(4):
                    W[j][r] = W_str[j][r]
            
            # Standard part for cols 2..n_cols-1
            col_offset = 2
            for dc in range(n_cols - 2):
                base_col = col_offset + dc
                # For each column, standard 2x2*2x1: rank 4
                # Z1k = a*y1k + b*y2k
                # Z2k = c*y1k + d*y2k
                # 4 products: a*y1k, b*y2k, c*y1k, d*y2k
                
                # Y indices: row 0 is 2*base_col, row 1 is 2*base_col+1
                # Z indices: row 0 is 2*base_col, row 1 is 2*base_col+1
                
                # Product 1: a * y1k
                r = rank_str + dc*4
                U[0][r] = 1  # a
                V[2*base_col][r] = 1  # y1k
                W[2*base_col][r] = 1  # Z1k
                
                # Product 2: b * y2k
                r = rank_str + dc*4 + 1
                U[1][r] = 1  # b
                V[2*base_col+1][r] = 1  # y2k
                W[2*base_col][r] = 1  # Z1k
                
                # Product 3: c * y1k
                r = rank_str + dc*4 + 2
                U[2][r] = 1  # c
                V[2*base_col][r] = 1  # y1k
                W[2*base_col+1][r] = 1  # Z2k
                
                # Product 4: d * y2k
                r = rank_str + dc*4 + 3
                U[3][r] = 1  # d
                V[2*base_col+1][r] = 1  # y2k
                W[2*base_col+1][r] = 1  # Z2k
            
            return U, V, W
    
    # Build full decomposition
    # A indices: (i,j) -> i*m+j = i*4+j, i in {0,1}, j in {0,1,2,3}
    # B indices: (j,k) -> j*p+k = j*7+k, j in {0,1,2,3}, k in {0,...,6}
    # C indices: (i,k) -> k*n+i = k*2+i, k in {0,...,6}, i in {0,1}
    
    # Block 1: A1 = A[:, 0:2], B1 = B[0:2, :]
    # Block 2: A2 = A[:, 2:4], B2 = B[2:4, :]
    
    U1, V1, W1 = make_2x2_2xn_decomp(7)
    U2, V2, W2 = make_2x2_2xn_decomp(7)
    
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(n*m)]  # 8 x total_rank
    V = [[0]*total_rank for _ in range(m*p)]  # 28 x total_rank
    W = [[0]*total_rank for _ in range(p*n)]  # 14 x total_rank
    
    # Fill block 1
    # A1 indices: (i, j) where j in {0,1} -> global index i*4+j
    # For U1, local X index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Global A index for (i, j_local): i*4 + j_local where j_local in {0,1}
    for r in range(rank1):
        for local_i in range(4):
            # local_i: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            li = local_i // 2
            lj = local_i % 2
            global_idx = li * 4 + lj
            U[global_idx][r] = U1[local_i][r]
    
    # B1 indices: (j, k) where j in {0,1} -> global index j*7+k
    # For V1, local Y index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) for first 2 cols
    # Actually V1 has 2*7=14 entries. Local index for (row, col) is row*7+col
    # But in make_2x2_2xn_decomp, V has 2*n_cols entries indexed as row*n_cols+col?
    # Let me check: in the function, V[j] where j goes 0..2*n_cols-1
    # For row 0: cols 0,1 -> indices 0,1
    # For row 1: cols 0,1 -> indices 2,3
    # So it's row-major with n_cols columns: index = row*n_cols + col
    
    for r in range(rank1):
        for local_j in range(14):
            local_row = local_j // 7
            local_col = local_j % 7
            global_idx = local_row * 7 + local_col
            V[global_idx][r] = V1[local_j][r]
    
    # W1: local index for Z: k*n+i = k*2+i, k in 0..6, i in 0..1
    # Same as global C index
    for r in range(rank1):
        for local_idx in range(14):
            W[local_idx][r] = W1[local_idx][r]
    
    # Fill block 2
    # A2 indices: (i, j) where j in {2,3} -> global index i*4+j
    # Local X index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Global: li*4 + (lj+2)
    for r in range(rank2):
        for local_i in range(4):
            li = local_i // 2
            lj = local_i % 2
            global_idx = li * 4 + (lj + 2)
            U[global_idx][rank1 + r] = U2[local_i][r]
    
    # B2 indices: (j, k) where j in {2,3} -> global index j*7+k
    # Local row: 0,1 -> global row: 2,3
    for r in range(rank2):
        for local_j in range(14):
            local_row = local_j // 7
            local_col = local_j % 7
            global_idx = (local_row + 2) * 7 + local_col
            V[global_idx][rank1 + r] = V2[local_j][r]
    
    # W2: same C indices
    for r in range(rank2):
        for local_idx in range(14):
            W[local_idx][rank1 + r] = W2[local_idx][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
