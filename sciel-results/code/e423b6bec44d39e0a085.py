"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank 33 decomposition using Strassen's algorithm on 2x2x2 sub-blocks.
    # U: shape (6, 33), V: shape (18, 33), W: shape (12, 33)
    # We construct 11 base patterns (7 Strassen + 4 direct 2x1x2) and repeat them 3 times
    # for the three 2-column blocks of the 3x6 matrix.

    n, m, p = DIMENSIONS
    rank = 33

    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define the 11 base patterns (Strassen + direct)
    # Each pattern is a tuple (u_pattern, v_pattern, w_pattern)
    # u_pattern: list of 6 values (indices 0..5)
    # v_pattern: list of 6 values for a 2x2 block (rows 0,1 for j=0; rows 6,7 for j=1; rows 12,13 for j=2)
    #   Actually, let's define v_pattern as a dictionary or list of (row, val) for clarity, or just a 18-length list.
    # Let's use 18-length lists for V to be safe and clear.

    # Strassen patterns (indices 0-6)
    # M1: A=(1,0,1,0) -> U[0]=1, U[3]=1; B=(1,0,0,1) -> V[0]=1, V[7]=1; W=(1,0,0,1) -> W[0]=1, W[3]=1
    # Note: A indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    # B indices for block i (cols 2i, 2i+1):
    #   Row 0 (j=0): 2i, 2i+1
    #   Row 1 (j=1): 6+2i, 6+2i+1
    #   Row 2 (j=2): 12+2i, 12+2i+1
    # W indices: k*n + i. Here n=2, so 2k+i.
    #   C(0, 2i) -> W[2*0+2i] = W[2i]
    #   C(0, 2i+1) -> W[2*0+2i+1] = W[2i+1]
    #   C(1, 2i) -> W[2*1+2i] = W[2+2i]
    #   C(1, 2i+1) -> W[2*1+2i+1] = W[2+2i+1]

    # Let's define the 11 base vectors for U, V, W.
    # U is same for all 3 blocks.
    # V depends on block index i.
    # W depends on block index i.

    # U base vectors (length 6)
    u_m1 = [1, 0, 0, 1, 0, 0]
    u_m2 = [0, 0, 0, 1, 1, 0]
    u_m3 = [1, 0, 0, 0, 0, 0]
    u_m4 = [0, 0, 0, 0, 0, 1]
    u_m5 = [1, 1, 0, 0, 0, 0]
    u_m6 = [-1, 0, 0, 1, 0, 0]
    u_m7 = [0, 1, 0, 0, 0, -1]
    # Direct 2x1x2 parts (using A col 2)
    u_m8 = [1, 0, 1, 0, 0, 0]  # A(0,0) and A(0,2)? No, Strassen uses A(0,0), A(0,1), A(1,0), A(1,1).
                               # The "extra" column A(:,2) is used with B(2,:).
                               # For 2x1x2, we multiply A(2x2) * B(1x2).
                               # Actually, the tensor is 2x3x6.
                               # We split B into B1 (2x2), B2 (2x2), B3 (2x2).
                               # We split A into A1 (2x2), A2 (1x2).
                               # C = A1*B1 + A2*B2 + A3*B3? No.
                               # C = [A1 A2] * [B1; B2; B3] where A1 is 2x2, A2 is 1x2?
                               # No, A is 2x3. So A = [A_left (2x2), A_right (2x1)].
                               # B is 3x6. B = [B_top (2x6); B_bottom (1x6)].
                               # C = A_left * B_top + A_right * B_bottom.
                               # A_left is 2x2. B_top is 2x6.
                               # We apply Strassen to A_left * B_top_block (2x2 * 2x2) for each of the 3 blocks of width 2.
                               # A_right is 2x1. B_bottom is 1x6.
                               # A_right * B_bottom_block is 2x1 * 1x2 = 2x2.
                               # This is a direct multiplication, rank 4.
                               # So total rank = 7 (Strassen) + 4 (Direct) = 11 per block.
                               # 11 * 3 = 33.

    # Let's re-verify the U vectors for the direct part.
    # Direct part: A_right (2x1) * B_bottom_block (1x2).
    # A_right = [A(0,2), A(1,2)]^T.
    # B_bottom_block = [B(2, 2i), B(2, 2i+1)].
    # C_block_direct = A_right * B_bottom_block.
    # C(0, 2i) = A(0,2)*B(2, 2i)
    # C(0, 2i+1) = A(0,2)*B(2, 2i+1)
    # C(1, 2i) = A(1,2)*B(2, 2i)
    # C(1, 2i+1) = A(1,2)*B(2, 2i+1)

    # So for M8 (C(0,2i)): U[2]=1 (A(0,2)), V[12+2i]=1 (B(2,2i)), W[2i]=1.
    # For M9 (C(0,2i+1)): U[2]=1, V[12+2i+1]=1, W[2i+1]=1.
    # For M10 (C(1,2i)): U[5]=1 (A(1,2)), V[12+2i]=1, W[2+2i]=1.
    # For M11 (C(1,2i+1)): U[5]=1, V[12+2i+1]=1, W[2+2i+1]=1.

    u_m8 = [0, 0, 1, 0, 0, 0]
    u_m9 = [0, 0, 1, 0, 0, 0]
    u_m10 = [0, 0, 0, 0, 0, 1]
    u_m11 = [0, 0, 0, 0, 0, 1]

    u_patterns = [u_m1, u_m2, u_m3, u_m4, u_m5, u_m6, u_m7, u_m8, u_m9, u_m10, u_m11]

    # V patterns for a single block i (length 18)
    # Indices:
    # 0: B(0, 2i)
    # 1: B(0, 2i+1)
    # 6: B(1, 2i)
    # 7: B(1, 2i+1)
    # 12: B(2, 2i)
    # 13: B(2, 2i+1)

    def make_v_block(i):
        # Returns list of 11 vectors of length 18
        vecs = []
        # M1: B(0,2i), B(1,2i+1)
        v = [0]*18; v[2*i]=1; v[6+2*i+1]=1; vecs.append(v)
        # M2: B(0,2i)
        v = [0]*18; v[2*i]=1; vecs.append(v)
        # M3: B(0,2i+1), -B(1,2i+1)
        v = [0]*18; v[2*i+1]=1; v[6+2*i+1]=-1; vecs.append(v)
        # M4: B(1,2i), -B(1,2i+1)
        v = [0]*18; v[6+2*i]=1; v[6+2*i+1]=-1; vecs.append(v)
        # M5: B(1,2i+1)
        v = [0]*18; v[6+2*i+1]=1; vecs.append(v)
        # M6: B(0,2i), B(0,2i+1)
        v = [0]*18; v[2*i]=1; v[2*i+1]=1; vecs.append(v)
        # M7: B(1,2i), B(1,2i+1)
        v = [0]*18; v[6+2*i]=1; v[6+2*i+1]=1; vecs.append(v)
        # M8: B(2,2i)
        v = [0]*18; v[12+2*i]=1; vecs.append(v)
        # M9: B(2,2i+1)
        v = [0]*18; v[12+2*i+1]=1; vecs.append(v)
        # M10: B(2,2i)
        v = [0]*18; v[12+2*i]=1; vecs.append(v)
        # M11: B(2,2i+1)
        v = [0]*18; v[12+2*i+1]=1; vecs.append(v)
        return vecs

    # W patterns for a single block i (length 12)
    # Indices:
    # 2i: C(0, 2i)
    # 2i+1: C(0, 2i+1)
    # 2+2i: C(1, 2i)
    # 2+2i+1: C(1, 2i+1)

    def make_w_block(i):
        vecs = []
        # M1: C(0,2i), C(1,2i+1)
        v = [0]*12; v[2*i]=1; v[2+2*i+1]=1; vecs.append(v)
        # M2: C(1,2i)
        v = [0]*12; v[2+2*i]=1; vecs.append(v)
        # M3: C(0,2i+1), C(1,2i+1)
        v = [0]*12; v[2*i+1]=1; v[2+2*i+1]=1; vecs.append(v)
        # M4: C(0,2i), C(1,2i)
        v = [0]*12; v[2*i]=1; v[2+2*i]=1; vecs.append(v)
        # M5: -C(0,2i), C(0,2i+1)
        v = [0]*12; v[2*i]=-1; v[2*i+1]=1; vecs.append(v)
        # M6: C(1,2i+1)
        v = [0]*12; v[2+2*i+1]=1; vecs.append(v)
        # M7: C(0,2i)
        v = [0]*12; v[2*i]=1; vecs.append(v)
        # M8: C(0,2i)
        v = [0]*12; v[2*i]=1; vecs.append(v)
        # M9: C(0,2i+1)
        v = [0]*12; v[2*i+1]=1; vecs.append(v)
        # M10: C(1,2i)
        v = [0]*12; v[2+2*i]=1; vecs.append(v)
        # M11: C(1,2i+1)
        v = [0]*12; v[2+2*i+1]=1; vecs.append(v)
        return vecs

    # Fill U
    for i in range(3):
        for j in range(11):
            col = i * 11 + j
            for r in range(6):
                U[r][col] = u_patterns[j][r]

    # Fill V and W
    for i in range(3):
        v_vecs = make_v_block(i)
        w_vecs = make_w_block(i)
        for j in range(11):
            col = i * 11 + j
            for r in range(18):
                V[r][col] = v_vecs[j][r]
            for r in range(12):
                W[r][col] = w_vecs[j][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
