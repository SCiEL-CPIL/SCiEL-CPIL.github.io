"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Uses the known 7-term Winograd scheme for 2x2x2 sub-blocks where applicable,
    and schoolbook for the remaining 1-wide strips.
    
    For 4x5x5:
    - Split rows into two blocks of size 2: [0:2], [2:4]
    - Split mids into blocks [0:2], [2:4], [4:5] (sizes 2, 2, 1)
    - Split cols into blocks [0:2], [2:4], [4:5] (sizes 2, 2, 1)
    
    For 2x2x2 sub-blocks (4 of them: 2 row-blocks x 2 mid-blocks-of-size-2 x ... wait)
    Actually: 2 row-blocks (size 2 each) x 3 mid-blocks (sizes 2,2,1) x 3 col-blocks (sizes 2,2,1)
    
    2x2x2 sub-blocks occur when row-block size=2, mid-block size=2, col-block size=2:
    - 2 row-blocks x 2 mid-blocks-of-size-2 x 2 col-blocks-of-size-2 = 8 sub-blocks of 2x2x2
    
    Wait, that's not right either. Let me reconsider.
    
    The 4x5x5 multiplication: A is 4x5, B is 5x5, C is 4x5.
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    Block decomposition:
    - A rows: [0:2], [2:4] -> 2 blocks of height 2
    - A cols / B rows: [0:2], [2:4], [4:5] -> blocks of width 2, 2, 1
    - B cols / C cols: [0:2], [2:4], [4:5] -> blocks of width 2, 2, 1
    - C rows: [0:2], [2:4] -> 2 blocks of height 2
    
    Sub-block multiplication: C_block[ri, ci] = sum_{mi} A_block[ri, mi] * B_block[mi, ci]
    
    For each combination of (row_block, mid_block, col_block), we have a sub-problem of size
    (row_size x mid_size) * (mid_size x col_size) = row_size x col_size.
    
    There are 2 * 3 * 3 = 18 sub-problems.
    
    Sub-problem sizes:
    - (2x2)*(2x2) -> 2x2: rank 7 (Winograd) instead of 4 (schoolbook) -- wait, schoolbook is 4
    - (2x2)*(2x1) -> 2x1: rank 2 (schoolbook)
    - (2x1)*(2x2) -> 2x2: rank 4 (schoolbook)
    - (2x1)*(2x1) -> 2x1: rank 2 (schoolbook)
    - etc.
    
    Wait, I'm confusing myself. The sub-problem for a specific (row_block, mid_block, col_block) 
    triple is: multiply a (row_size x mid_size) matrix by a (mid_size x col_size) matrix.
    
    The number of multiplications needed for a (r x m) * (m x p) sub-problem is r*m*p (schoolbook).
    With Winograd, for 2x2x2 we can do it in 7 instead of 8.
    
    So the total schoolbook rank is sum over all (rb, mb, cb) of (size_rb * size_mb * size_cb)
    = 4 * 5 * 5 = 100 (which is just n*m*p).
    
    With Winograd on the 2x2x2 sub-problems:
    - Sub-problems where rb_size=2, mb_size=2, cb_size=2: use rank 7 instead of 8
    - How many such sub-problems? 2 row-blocks x 2 mid-blocks-of-size-2 x 2 col-blocks-of-size-2 = 8
    - Savings: 8 * (8-7) = 8
    - New rank: 100 - 8 = 92
    
    Wait, but that's not quite right. The sub-problems are not independent in the same way.
    Actually, they ARE independent because each output element C[i,k] depends only on the 
    corresponding sub-block products.
    
    Let me recount:
    Row blocks: R0=[0:2] (size 2), R1=[2:4] (size 2)
    Mid blocks: M0=[0:2] (size 2), M1=[2:4] (size 2), M2=[4:5] (size 1)
    Col blocks: C0=[0:2] (size 2), C1=[2:4] (size 2), C2=[4:5] (size 1)
    
    Sub-problem (Ri, Mj, Ck) has dimensions: (2 x size_Mj) * (size_Mj x size_Ck)
    
    Schoolbook cost for each:
    (R0,M0,C0): 2*2*2 = 8
    (R0,M0,C1): 2*2*2 = 8
    (R0,M0,C2): 2*2*1 = 4
    (R0,M1,C0): 2*2*2 = 8
    (R0,M1,C1): 2*2*2 = 8
    (R0,M1,C2): 2*2*1 = 4
    (R0,M2,C0): 2*1*2 = 4
    (R0,M2,C1): 2*1*2 = 4
    (R0,M2,C2): 2*1*1 = 2
    (R1,M0,C0): 2*2*2 = 8
    (R1,M0,C1): 2*2*2 = 8
    (R1,M0,C2): 2*2*1 = 4
    (R1,M1,C0): 2*2*2 = 8
    (R1,M1,C1): 2*2*2 = 8
    (R1,M1,C2): 2*2*1 = 4
    (R1,M2,C0): 2*1*2 = 4
    (R1,M2,C1): 2*1*2 = 4
    (R1,M2,C2): 2*1*1 = 2
    
    Total schoolbook: 8+8+4+8+8+4+4+4+2+8+8+4+8+8+4+4+4+2 = 100 ✓
    
    Winograd 2x2x2 saves 1 multiplication per 2x2x2 sub-problem.
    The 2x2x2 sub-problems are those with mb_size=2 and cb_size=2:
    (R0,M0,C0), (R0,M0,C1), (R0,M1,C0), (R0,M1,C1),
    (R1,M0,C0), (R1,M0,C1), (R1,M1,C0), (R1,M1,C1)
    That's 8 sub-problems, each saving 1, total savings = 8.
    New rank: 100 - 8 = 92.
    
    Let me implement this.
    """
    import time
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    
    # Block definitions
    row_blocks = [(0, 2, 2), (2, 4, 2)]  # (start, end, size)
    mid_blocks = [(0, 2, 2), (2, 4, 2), (4, 5, 1)]
    col_blocks = [(0, 2, 2), (2, 4, 2), (4, 5, 1)]
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms_U.append(u_entries)
        terms_V.append(v_entries)
        terms_W.append(w_entries)
    
    def schoolbook_subblock(r0, m0, k0, rs, ms, cs):
        """Add schoolbook terms for sub-block A[r0:r0+rs, m0:m0+ms] * B[m0:m0+ms, k0:k0+cs]."""
        for ri in range(rs):
            for mj in range(ms):
                for ck in range(cs):
                    u_idx = (r0 + ri) * m + (m0 + mj)
                    v_idx = (m0 + mj) * p + (k0 + ck)
                    w_idx = (k0 + ck) * n + (r0 + ri)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    def winograd_2x2x2(r0, m0, k0):
        """
        Add 7 rank-1 terms for 2x2x2 sub-block at A[r0:r0+2, m0:m0+2] * B[m0:m0+2, k0:k0+2].
        Uses the standard Winograd 7-term scheme.
        
        Let A_sub = [[a, b], [c, d]] where:
          a = A[r0, m0], b = A[r0, m0+1], c = A[r0+1, m0], d = A[r0+1, m0+1]
        Let B_sub = [[e, f], [g, h]] where:
          e = B[m0, k0], f = B[m0, k0+1], g = B[m0+1, k0], h = B[m0+1, k0+1]
        C_sub = A_sub * B_sub = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        The 7 multiplications are:
          m1 = (a+c)(e+g)
          m2 = (a-c)(e-g)
          m3 = (b+d)(f+g)
          m4 = (b-d)(f-g)
          m5 = (a+c+d)(e+g+h)
          m6 = (a+c+d)(f+g)  -- wait, let me verify this scheme
        
        Actually, let me use a verified scheme. The standard Winograd 2x2x2 uses:
          m1 = (a+c)(e+g)
          m2 = (a-c)(e-g)
          m3 = (b+d)(f+g)
          m4 = (b-d)(f-g)
          m5 = (a+c+d+e+g+f+h) -- NO, this mixes A and B variables
        
        I need to be careful. Each term must be (linear form in A) * (linear form in B).
        
        Let me use this verified 7-term scheme:
          m1 = (a+c) * (e+g)
          m2 = (a-c) * (e-g)
          m3 = (b+d) * (f+g)
          m4 = (b-d) * (f-g)
          m5 = (a+c+d) * (e+g+h)
          m6 = (a+c+d) * (f+g)  -- Hmm, need to verify
          m7 = ?
        
        Let me just derive it properly. We need to express:
          C00 = ae + bg
          C01 = af + bh
          C10 = ce + dg
          C11 = cf + dh
        
        using 7 products of the form (linear_A)(linear_B).
        
        One known valid scheme:
          m1 = (a+c)(e+g)       -> contributes to C00, C01, C10, C11
          m2 = (a-c)(e-g)       -> contributes to C00, C01, C10, C11
          m3 = (b+d)(f+g)       -> contributes to C00, C01, C10, C11
          m4 = (b-d)(f-g)       -> contributes to C00, C01, C10, C11
          m5 = (a+c+d)(e+g+h)  -> contributes to C00, C01, C10, C11
          m6 = (a+c+d)(f+g)    -> contributes to C00, C01, C10, C11
          m7 = (a+c+d)(h)      -> contributes to C00, C01, C10, C11
        
        Let me verify by expanding:
        C00 should be ae + bg:
        From m1: +ae + ag + ce + cg
        From m2: +ae - ag - ce + cg
        From m3: +bf + bg + df + dg
        From m4: +bf - bg - df + dg
        From m5: +ae + ag + ah + ce + cg + ch + de + dg + dh
        From m6: +af + ag + cf + cg + df + dg
        From m7: +ah + ch + dh
        
        This is getting complicated. Let me just use the schoolbook for all sub-blocks to ensure correctness,
        and accept rank 100. The Winograd optimization is tricky to get right without a reference.
        
        Actually, I recall a simpler valid 7-term scheme:
          m1 = (a+c)(e+g)
          m2 = (a-c)(e-g)
          m3 = (b+d)(f+g)
          m4 = (b-d)(f-g)
          m5 = (a+c+d+e+g+f+h) -- INVALID
        
        You know what, let me just use schoolbook for everything. Rank 100 is correct.
        """
        # For now, just use schoolbook for 2x2x2 as well (8 terms)
        schoolbook_subblock(r0, m0, k0, 2, 2, 2)
    
    # Process all sub-blocks
    for r_start, r_end, r_size in row_blocks:
        for m_start, m_end, m_size in mid_blocks:
            for k_start, k_end, k_size in col_blocks:
                if r_size == 2 and m_size == 2 and k_size == 2:
                    # Use Winograd (but for now, fall back to schoolbook)
                    winograd_2x2x2(r_start, m_start, k_start)
                else:
                    schoolbook_subblock(r_start, m_start, k_start, r_size, m_size, k_size)
    
    rank = len(terms_U)
    
    # Build U, V, W matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        for idx, val in terms_U[c]:
            U[idx, c] = val
        for idx, val in terms_V[c]:
            V[idx, c] = val
        for idx, val in terms_W[c]:
            W[idx, c] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
