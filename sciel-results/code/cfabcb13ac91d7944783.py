import time
import numpy as np

_DIMS = (3, 4, 6)
_DEADLINE = time.time() + 110

def construct():
    """
    Construct a rank-66 exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    The decomposition is built by splitting the inner dimension m=4 into 1+3,
    using rank-18 for M(3,1,6) and rank-48 for M(3,3,6) via three 2x2x2 Strassen blocks.
    """
    n, m, p = _DIMS
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_sp, v_sp, w_sp):
        """u_sp, v_sp, w_sp are lists of (index, coeff)"""
        nonlocal col
        if col >= rank:
            raise RuntimeError("rank exceeded")
        for idx, val in u_sp:
            U[idx, col] += val
        for idx, val in v_sp:
            V[idx, col] += val
        for idx, val in w_sp:
            W[idx, col] += val
        col += 1

    # --- Block 1: M(3,1,6) with j=0 ---
    # Rank 18: each (i,k) pair is a rank-1 term
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            add_term([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])

    # --- Block 2: M(3,3,6) with j in 1..3 ---
    # Split p=6 into three blocks of 2: k in [0,1], [2,3], [4,5]
    # For each block, use M(3,3,2). Split i=3 into 1+2.
    # M(1,3,2): rank 6 (each (j,k) pair)
    # M(2,3,2): split j=3 into 1+2 -> M(2,1,2) rank 4 + M(2,2,2) rank 7 = 11
    # Total per k-block: 6 + 11 = 17, times 3 = 51? That's too much.
    # Better: M(3,3,2) directly.
    # M(3,3,2): split i=3 into 1+2, j=3 into 1+2
    # M(1,1,2): 2
    # M(1,2,2): 4
    # M(2,1,2): 4
    # M(2,2,2): 7 (Strassen)
    # Total: 2+4+4+7 = 17 per k-block, 3 blocks = 51. Plus 18 = 69. Too much.
    
    # Alternative: use M(3,3,2) with a direct rank-12 construction?
    # Actually M(3,3,2) has rank 12 (schoolbook) but can be done in 12.
    # Wait, 3*3*2 = 18 schoolbook. But we can do better.
    # For M(a,b,2), the rank is at most 2*a*b but can be less.
    # For M(3,3,2), we can split into M(3,1,2)+M(3,2,2) = 6 + 12 = 18? No.
    # M(3,1,2) = 6, M(3,2,2): split i=3 into 1+2: M(1,2,2)=4, M(2,2,2)=7, total 11. So 6+11=17.
    
    # Let me use a different split for the whole M(3,3,6).
    # Split p=6 into 3+3.
    # M(3,3,3) = 23 (Landsberg). But I don't have the exact terms.
    # Use M(3,3,3) = M(3,1,3) + M(3,2,3) = 9 + (M(3,1,3)+M(3,2,3))
    # M(3,1,3) = 9
    # M(3,2,3): split i=3 into 1+2: M(1,2,3)=6, M(2,2,3): split p=3 into 1+2: M(2,2,1)=4, M(2,2,2)=7, total 11. So 6+11=17.
    # M(3,3,3) = 9 + 17 = 26. Two of them = 52. Plus 18 = 70. Worse.
    
    # Let me stick with the 66-rank construction that was in the original.
    # The original used: for each k-block of 2, add_1_4_2 (rank 8) + add_2_4_2 (rank 14) = 22 per block, 3 blocks = 66.
    # Wait, that's 66 total, not 18+66. Let me re-examine.
    
    # The original 66-rank construction:
    # For block_k in range(3): k_off = block_k * 2
    #   add_1_4_2(0, 0, k_off): 4*2 = 8 rank-1 terms
    #   add_2_4_2(1, 0, k_off): 2 * 7 = 14 rank-1 terms
    # Total: 3 * (8 + 14) = 66.
    # This covers i=0 (1x4x2 per block) and i=1,2 (2x4x2 per block via two 2x2x2).
    # So it's a direct M(3,4,6) decomposition, not split by m.
    
    # Let me implement this correctly.
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_1_4_2(i_off, j_off, k_off):
        """Add M(1,4,2) block: i fixed, j in [j_off, j_off+3], k in [k_off, k_off+1]"""
        nonlocal col
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1

    def add_2_2_2(i_off, j_off, k_off):
        """Add M(2,2,2) block using 7 rank-1 terms (Strassen-like)."""
        nonlocal col
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # 7 terms for M(2,2,2)
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            ([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1]),
            ([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1]),
            ([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1]),
            ([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1]),
            ([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1]),
            ([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1]),
            ([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1]),
        ]
        
        for u_idx_list, u_vals, v_idx_list, v_vals, w_idx_list, w_vals in terms:
            for idx, val in zip(u_idx_list, u_vals):
                U[idx, col] += val
            for idx, val in zip(v_idx_list, v_vals):
                V[idx, col] += val
            for idx, val in zip(w_idx_list, w_vals):
                W[idx, col] += val
            col += 1

    def add_2_4_2(i_off, j_off, k_off):
        """Add M(2,4,2) block by splitting j into two M(2,2,2)."""
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    # Main loop: 3 k-blocks of size 2
    for block_k in range(3):
        k_off = block_k * 2
        # i=0: M(1,4,2)
        add_1_4_2(0, 0, k_off)
        # i=1,2: M(2,4,2)
        add_2_4_2(1, 0, k_off)

    # Verify rank
    if col != rank:
        raise RuntimeError(f"Expected rank {rank}, got {col}")

    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
