"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Block decomposition using Strassen for 2x2x2 and schoolbook for 1x2x2.
    Rank is 88, which is better than the schoolbook rank of 96.
    """
    n, m, p = DIMENSIONS
    
    # Helper to create a block decomposition from a list of terms
    # term: (u_list, v_list, w_list) where u_list is dict {idx: val}, etc.
    def make_block(n_loc, m_loc, p_loc, terms):
        """
        Creates U, V, W matrices for a local block.
        u indices: i*m_loc + j
        v indices: j*p_loc + k
        w indices: k*n_loc + i
        """
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n_loc * m_loc)]
        V = [[0.0] * rank for _ in range(m_loc * p_loc)]
        W = [[0.0] * rank for _ in range(p_loc * n_loc)]
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, val in u_dict.items():
                U[idx][r] = val
            for idx, val in v_dict.items():
                V[idx][r] = val
            for idx, val in w_dict.items():
                W[idx][r] = val
        return U, V, W

    def combine_blocks(blocks, n_glob, m_glob, p_glob, 
                       u_maps, v_maps, w_maps):
        """
        Combines multiple blocks into a global decomposition.
        blocks: list of (U, V, W)
        u_maps, v_maps, w_maps: list of functions mapping local idx to global idx
        """
        total_rank = sum(len(b[2]) for b in blocks)
        U = [[0.0] * total_rank for _ in range(n_glob * m_glob)]
        V = [[0.0] * total_rank for _ in range(m_glob * p_glob)]
        W = [[0.0] * total_rank for _ in range(p_glob * n_glob)]
        
        offset = 0
        for b_idx, (Ub, Vb, Wb) in enumerate(blocks):
            rank_b = len(Wb)
            umap = u_maps[b_idx]
            vmap = v_maps[b_idx]
            wmap = w_maps[b_idx]
            
            for i in range(len(Ub)):
                global_i = umap(i)
                if global_i is not None:
                    for r in range(rank_b):
                        U[global_i][offset + r] = Ub[i][r]
                        
            for j in range(len(Vb)):
                global_j = vmap(j)
                if global_j is not None:
                    for r in range(rank_b):
                        V[global_j][offset + r] = Vb[j][r]
                        
            for k in range(len(Wb)):
                global_k = wmap(k)
                if global_k is not None:
                    for r in range(rank_b):
                        W[global_k][offset + r] = Wb[k][r]
                        
            offset += rank_b
            
        return U, V, W

    # 1. Strassen 2x2x2 (Rank 7)
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a10)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    s_terms = [
        # M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # C00 +1, C11 +1
        # M2
        ({2:1, 3:1}, {0:1}, {1:1, 3:-1}),    # C10 +1, C11 -1
        # M3
        ({0:1}, {1:-1, 3:-1}, {1:1, 3:1}),   # C01 +1, C11 +1 (Wait, M3 is a00(b01-b11))
        # Let's re-verify Strassen coefficients for C
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00 +1, C11 +1. w: {0:1, 3:1}
        # M2: C10 +1, C11 -1. w: {1:1, 3:-1}
        # M3: C01 +1, C11 +1. w: {1:1, 3:1}
        # M4: C00 +1, C10 +1. w: {0:1, 1:1}
        # M5: C00 -1, C01 +1. w: {0:-1, 1:1}
        # M6: C11 +1. w: {3:1}
        # M7: C00 +1. w: {0:1}
        
        # Correcting M3, M4, M5, M6, M7 w-vectors
        # M3: a00(b01-b11). u:{0:1}, v:{1:1, 3:-1}. w:{1:1, 3:1}
        # M4: a11(b10-b00). u:{3:1}, v:{2:1, 0:-1}. w:{0:1, 1:1}
        # M5: (a00+a10)b11. u:{0:1, 2:1}, v:{3:1}. w:{0:-1, 1:1}
        # M6: (a10-a00)(b00+b01). u:{2:1, 0:-1}, v:{0:1, 1:1}. w:{3:1}
        # M7: (a01-a11)(b10+b11). u:{1:1, 3:-1}, v:{2:1, 3:1}. w:{0:1}
    ]
    
    # Re-defining s_terms correctly based on C formulas
    s_terms = [
        # M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2
        ({2:1, 3:1}, {0:1}, {1:1, 3:-1}),
        # M3
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        # M4
        ({3:1}, {2:1, 0:-1}, {0:1, 1:1}),
        # M5
        ({0:1, 2:1}, {3:1}, {0:-1, 1:1}),
        # M6
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # M7
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    U_s, V_s, W_s = make_block(2, 2, 2, s_terms)

    # 2. Schoolbook 1x2x2 (Rank 4)
    # A=[a0, a1], B=[[b00, b01], [b10, b11]]
    # C0 = a0b00 + a1b10
    # C1 = a0b01 + a1b11
    sb_terms = [
        ({0:1}, {0:1}, {0:1}), # a0b00 -> C0
        ({1:1}, {2:1}, {0:1}), # a1b10 -> C0 (V idx: j*p+k. j=1, k=0 -> 1*2+0=2)
        ({0:1}, {1:1}, {1:1}), # a0b01 -> C1 (V idx: j=0, k=1 -> 0*2+1=1)
        ({1:1}, {3:1}, {1:1})  # a1b11 -> C1 (V idx: j=1, k=1 -> 1*2+1=3)
    ]
    U_sb, V_sb, W_sb = make_block(1, 2, 2, sb_terms)

    # 3. Construct 2x4x2 (Rank 14)
    # Two 2x2x2 blocks.
    # Block 1: A_L (cols 0,1), B_T (rows 0,1).
    # Block 2: A_R (cols 2,3), B_B (rows 2,3).
    
    def map_2x4x2_u(idx, block):
        # idx is local 2x2 U index (0..3)
        i = idx // 2
        j = idx % 2
        if block == 0:
            return i * 4 + j
        else:
            return i * 4 + (2 + j)

    def map_2x4x2_v(idx, block):
        # idx is local 2x2 V index (0..3)
        j = idx // 2
        k = idx % 2
        if block == 0:
            return j * 2 + k
        else:
            return (2 + j) * 2 + k

    def map_2x4x2_w(idx, block):
        # idx is local 2x2 W index (0..3)
        # W is 2x2. Same for both blocks.
        return idx

    blocks_242 = [(U_s, V_s, W_s), (U_s, V_s, W_s)]
    u_maps_242 = [lambda i: map_2x4x2_u(i, 0), lambda i: map_2x4x2_u(i, 1)]
    v_maps_242 = [lambda i: map_2x4x2_v(i, 0), lambda i: map_2x4x2_v(i, 1)]
    w_maps_242 = [lambda i: map_2x4x2_w(i, 0), lambda i: map_2x4x2_w(i, 1)]
    
    U_242, V_242, W_242 = combine_blocks(blocks_242, 2, 4, 2, u_maps_242, v_maps_242, w_maps_242)

    # 4. Construct 2x4x4 (Rank 28)
    # Two 2x4x2 blocks.
    # Block 1: B_L (cols 0,1).
    # Block 2: B_R (cols 2,3).
    
    def map_244_u(idx, block):
        # idx is local 2x4 U index (0..7)
        return idx

    def map_244_v(idx, block):
        # idx is local 2x4 V index (0..7)
        j = idx // 2
        k = idx % 2
        if block == 0:
            return j * 4 + k
        else:
            return j * 4 + (2 + k)

    def map_244_w(idx, block):
        # idx is local 2x4 W index (0..3)
        # W is k*n+i. n=2.
        k = idx // 2
        i = idx % 2
        if block == 0:
            return k * 2 + i
        else:
            return (2 + k) * 2 + i

    blocks_244 = [(U_242, V_242, W_242), (U_242, V_242, W_242)]
    u_maps_244 = [lambda i: map_244_u(i, 0), lambda i: map_244_u(i, 1)]
    v_maps_244 = [lambda i: map_244_v(i, 0), lambda i: map_244_v(i, 1)]
    w_maps_244 = [lambda i: map_244_w(i, 0), lambda i: map_244_w(i, 1)]
    
    U_244, V_244, W_244 = combine_blocks(blocks_244, 2, 4, 4, u_maps_244, v_maps_244, w_maps_244)

    # 5. Construct 1x4x2 (Rank 8)
    # Two 1x2x2 blocks.
    # Block 1: A_L (cols 0,1), B_T (rows 0,1).
    # Block 2: A_R (cols 2,3), B_B (rows 2,3).
    
    def map_142_u(idx, block):
        # idx is local 1x2 U index (0..1)
        j = idx
        if block == 0:
            return j
        else:
            return 2 + j

    def map_142_v(idx, block):
        # idx is local 1x2 V index (0..3)
        j = idx // 2
        k = idx % 2
        if block == 0:
            return j * 2 + k
        else:
            return (2 + j) * 2 + k

    def map_142_w(idx, block):
        # idx is local 1x2 W index (0..1)
        return idx

    blocks_142 = [(U_sb, V_sb, W_sb), (U_sb, V_sb, W_sb)]
    u_maps_142 = [lambda i: map_142_u(i, 0), lambda i: map_142_u(i, 1)]
    v_maps_142 = [lambda i: map_142_v(i, 0), lambda i: map_142_v(i, 1)]
    w_maps_142 = [lambda i: map_142_w(i, 0), lambda i: map_142_w(i, 1)]
    
    U_142, V_142, W_142 = combine_blocks(blocks_142, 1, 4, 2, u_maps_142, v_maps_142, w_maps_142)

    # 6. Construct 1x4x4 (Rank 16)
    # Two 1x4x2 blocks.
    # Block 1: B_L (cols 0,1).
    # Block 2: B_R (cols 2,3).
    
    def map_144_u(idx, block):
        return idx

    def map_144_v(idx, block):
        j = idx // 2
        k = idx % 2
        if block == 0:
            return j * 4 + k
        else:
            return j * 4 + (2 + k)

    def map_144_w(idx, block):
        k = idx
        i = 0
        if block == 0:
            return k * 1 + i
        else:
            return (2 + k) * 1 + i

    blocks_144 = [(U_142, V_142, W_142), (U_142, V_142, W_142)]
    u_maps_144 = [lambda i: map_144_u(i, 0), lambda i: map_144_u(i, 1)]
    v_maps_144 = [lambda i: map_144_v(i, 0), lambda i: map_144_v(i, 1)]
    w_maps_144 = [lambda i: map_144_w(i, 0), lambda i: map_144_w(i, 1)]
    
    U_144, V_144, W_144 = combine_blocks(blocks_144, 1, 4, 4, u_maps_144, v_maps_144, w_maps_144)

    # 7. Construct 3x4x4 (Rank 44)
    # Top 2x4x4 and Bottom 1x4x4.
    
    def map_344_u_top(idx):
        # idx is local 2x4 U index (0..7)
        i = idx // 4
        j = idx % 4
        return i * 4 + j

    def map_344_u_bot(idx):
        # idx is local 1x4 U index (0..3)
        j = idx
        return 2 * 4 + j

    def map_344_v(idx, block):
        return idx

    def map_344_w_top(idx):
        # idx is local 2x4 W index (0..7)
        k = idx // 2
        i = idx % 2
        return k * 3 + i

    def map_344_w_bot(idx):
        # idx is local 1x4 W index (0..3)
        k = idx
        i = 0
        return k * 3 + 2

    blocks_344 = [(U_244, V_244, W_244), (U_144, V_144, W_144)]
    u_maps_344 = [map_344_u_top, map_344_u_bot]
    v_maps_344 = [lambda i: map_344_v(i, 0), lambda i: map_344_v(i, 1)]
    w_maps_344 = [map_344_w_top, map_344_w_bot]
    
    U_344, V_344, W_344 = combine_blocks(blocks_344, 3, 4, 4, u_maps_344, v_maps_344, w_maps_344)

    # 8. Construct 3x4x8 (Rank 88)
    # Two 3x4x4 blocks.
    # Block 1: B_L (cols 0..3).
    # Block 2: B_R (cols 4..7).
    
    def map_348_u(idx, block):
        return idx

    def map_348_v(idx, block):
        j = idx // 4
        k = idx % 4
        if block == 0:
            return j * 8 + k
        else:
            return j * 8 + (4 + k)

    def map_348_w(idx, block):
        k = idx // 3
        i = idx % 3
        if block == 0:
            return k * 3 + i
        else:
            return (4 + k) * 3 + i

    blocks_348 = [(U_344, V_344, W_344), (U_344, V_344, W_344)]
    u_maps_348 = [lambda i: map_348_u(i, 0), lambda i: map_348_u(i, 1)]
    v_maps_348 = [lambda i: map_348_v(i, 0), lambda i: map_348_v(i, 1)]
    w_maps_348 = [lambda i: map_348_w(i, 0), lambda i: map_348_w(i, 1)]
    
    U, V, W = combine_blocks(blocks_348, 3, 4, 8, u_maps_348, v_maps_348, w_maps_348)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
