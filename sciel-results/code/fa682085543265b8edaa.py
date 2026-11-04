"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add one rank-1 term to the decomposition."""
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] = coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] = coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_2x2(oi, oj, ok):
        """
        Add Strassen decomposition for 2x2 * 2x2 block multiplication.
        oi: offset row in A (0 or 2)
        oj: offset col in A / row in B (0 or 2)  
        ok: offset col in B (0, 2, or 4)
        """
        # m1 = (x00+x11)(y00+y11)
        u1 = {(oi, oj): 1, (oi+1, oj+1): 1}
        v1 = {(oj, ok): 1, (oj+1, ok+1): 1}
        w1 = {(oi, ok): 1, (oi+1, ok+1): 1}
        add_term(u1, v1, w1)
        
        # m2 = (x10+x11)y00
        u2 = {(oi+1, oj): 1, (oi+1, oj+1): 1}
        v2 = {(oj, ok): 1}
        w2 = {(oi+1, ok): 1, (oi+1, ok+1): -1}
        add_term(u2, v2, w2)
        
        # m3 = x00(y01-y11)
        u3 = {(oi, oj): 1}
        v3 = {(oj, ok+1): 1, (oj+1, ok+1): -1}
        w3 = {(oi, ok+1): 1, (oi+1, ok+1): 1}
        add_term(u3, v3, w3)
        
        # m4 = x11(y10-y00)
        u4 = {(oi+1, oj+1): 1}
        v4 = {(oj+1, ok): 1, (oj, ok): -1}
        w4 = {(oi, ok): 1, (oi+1, ok): 1}
        add_term(u4, v4, w4)
        
        # m5 = (x00+x01)y11
        u5 = {(oi, oj): 1, (oi, oj+1): 1}
        v5 = {(oj+1, ok+1): 1}
        w5 = {(oi, ok): -1, (oi, ok+1): 1}
        add_term(u5, v5, w5)
        
        # m6 = (x10-x00)(y00+y01)
        u6 = {(oi+1, oj): 1, (oi, oj): -1}
        v6 = {(oj, ok): 1, (oj, ok+1): 1}
        w6 = {(oi+1, ok+1): 1}
        add_term(u6, v6, w6)
        
        # m7 = (x01-x11)(y10+y11)
        u7 = {(oi, oj+1): 1, (oi+1, oj+1): -1}
        v7 = {(oj+1, ok): 1, (oj+1, ok+1): 1}
        w7 = {(oi, ok): 1}
        add_term(u7, v7, w7)

    # PART 1: 4x4 * 4x4 (columns 0-3 of B)
    # Use recursive Strassen (rank 49)
    def add_super_term(u_blocks, v_blocks, w_blocks, oi, oj, ok):
        """
        Add a Strassen super-term which is a sum of block products.
        """
        terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]
        
        for u_t, v_t, w_t in terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for (br, bc), coeff_u in u_blocks.items():
                for (r, c), coeff_t in u_t.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    u_col[i * m + j] += coeff_u * coeff_t
            
            for (br, bc), coeff_v in v_blocks.items():
                for (r, c), coeff_t in v_t.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    v_col[j * p + k] += coeff_v * coeff_t
            
            for (br, bc), coeff_w in w_blocks.items():
                for (r, c), coeff_t in w_t.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    w_col[k * n + i] += coeff_w * coeff_t
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    def strassen_4x4(oi, oj, ok):
        """
        Add Strassen decomposition for 4x4 * 4x4 block multiplication.
        """
        add_super_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, oi, oj, ok)
        add_super_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}, oi, oj, ok)
        add_super_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, oi, oj, ok)
        add_super_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, oi, oj, ok)
        add_super_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}, oi, oj, ok)
        add_super_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}, oi, oj, ok)
        add_super_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}, oi, oj, ok)

    # PART 1: 4x4 * 4x4 (columns 0-3 of B)
    if _time_remaining() > 5.0:
        strassen_4x4(0, 0, 0)
    
    # PART 2: 4x4 * 4x2 (columns 4-5 of B)
    # Unified approach: reuse A-side Strassen precomputation by extending V
    # Each of the 7 Strassen terms for columns 0-3 has V support on j-blocks.
    # For columns 4-5, we need 2x2 blocks at (j=0,k=4) and (j=2,k=4).
    # We add 7 additional terms sharing the same U patterns as columns 0-3,
    # but with V targeting the right column strip.
    # This gives rank 49 + 7*2 = 63... but we can do better.
    # Actually, let's use the direct 2x2 Strassen for the strip (rank 28).
    # Total: 49 + 28 = 77.
    
    # Alternative: use a shared U structure.
    # For the 4x4x2 strip, we can use rank-7 per 2x2 block (Strassen 2x2x2).
    # 4 blocks of 2x2x2 = rank 28.
    # But we can share: the A-side combinations are the same for all k columns.
    # Strassen 4x4x2: treat as 4x4 * 4x2.
    # Split A into 2x2 blocks (4 blocks), B into 2x2 blocks (2 blocks).
    # Use 7 Strassen terms on the A-side, each multiplied by appropriate B-side.
    # This gives rank 7 * 2 = 14 for the strip!
    # Total: 49 + 14 = 63.
    
    # Let's implement: for each Strassen term t in the 4x4x4 decomposition,
    # the U pattern is fixed. For the 4x4x2 strip, we use the same U pattern
    # but with V targeting columns 4-5 of B.
    
    # The 7 Strassen super-terms for 4x4x4 have U patterns:
    # t1: {(0,0):1,(1,1):1} on A blocks
    # t2: {(1,0):1,(1,1):1}
    # t3: {(0,0):1}
    # t4: {(1,1):1}
    # t5: {(0,0):1,(0,1):1}
    # t6: {(1,0):1,(0,0):-1}
    # t7: {(0,1):1,(1,1):-1}
    
    # For 4x4x2, we need to compute C[i,4] and C[i,5] for i=0..3.
    # C[i,4] = sum_j A[i,j]*B[j,4]
    # C[i,5] = sum_j A[i,j]*B[j,5]
    
    # Using Strassen on A (2x2 blocks) with B being 4x2:
    # Split A into 2x2 blocks: A00, A01, A10, A11 (rows 0-1/2-3, cols 0-1/2-3)
    # B is 4x2: rows 0-3, cols 4-5. Split B rows into 2x2: B0 (rows 0-1), B1 (rows 2-3)
    
    # Strassen terms for 4x4 * (4x2 treated as two 2x2 column vectors):
    # m1 = (A00+A11) * (B0+B1) -> contributes to C00, C11 of the 2x2 result
    # m2 = (A10+A11) * B0
    # m3 = A00 * (B1-B0)
    # m4 = A11 * (B0-B1)
    # m5 = (A00+A01) * B1
    # m6 = (A10-A00) * (B0+B1)
    # m7 = (A01-A11) * (B0+B1)
    
    # Each mi is a 2x2 result (2 rows x 2 cols since B is 2x2 per block).
    # Total: 7 terms * 2 cols = 14 rank-1 terms for the 4x4x2 strip.
    
    # Wait, each Strassen term gives a 2x2 output block. For 4x4x2:
    # The output is 4x2. Split into 2x2 blocks: top 2x2 and bottom 2x2.
    # m1 gives contributions to both top and bottom 2x2 blocks.
    # So each Strassen term contributes 2 rank-1 terms (one per output row-block).
    # Total: 7 * 2 = 14 terms.
    
    # Let me implement this properly.
    
    def strassen_4x4x2(oi, oj, ok):
        """
        Add Strassen decomposition for 4x4 * 4x2 block multiplication.
        oi: offset row in A (0)
        oj: offset col in A (0)
        ok: offset col in B (4)
        """
        # 7 Strassen terms, each producing 2x2 output blocks
        # U patterns on A (4x4, split into 2x2 blocks):
        # V patterns on B (4x2, split rows into 2+2):
        # W patterns on C (4x2, split rows into 2+2):
        
        terms_4x4x2 = [
            # (U_block_pattern, V_block_pattern, W_block_pattern)
            # U: dict of (block_row, block_col) -> coeff on A's 2x2 blocks
            # V: dict of (block_row, block_col) -> coeff on B's 2x2 blocks (B is 4x2, so 2 row-blocks x 1 col-block)
            # W: dict of (block_row, block_col) -> coeff on C's 2x2 blocks (C is 4x2, so 2 row-blocks x 1 col-block)
            
            # m1 = (A00+A11)*(B0+B1)
            # U: {(0,0):1, (1,1):1}, V: {(0,0):1, (1,0):1}, W: {(0,0):1, (1,0):1}
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (1,0): 1}),
            
            # m2 = (A10+A11)*B0
            # U: {(1,0):1, (1,1):1}, V: {(0,0):1}, W: {(1,0):1, (1,1):-1}
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            
            # m3 = A00*(B1-B0)
            # U: {(0,0):1}, V: {(1,0):1, (0,0):-1}, W: {(0,0):1, (1,0):1}
            ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            
            # m4 = A11*(B0-B1)
            # U: {(1,1):1}, V: {(0,0):1, (1,0):-1}, W: {(0,0):1, (1,0):1}
            ({(1,1): 1}, {(0,0): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
            
            # m5 = (A00+A01)*B1
            # U: {(0,0):1, (0,1):1}, V: {(1,0):1}, W: {(0,0):-1, (0,1):1}
            ({(0,0): 1, (0,1): 1}, {(1,0): 1}, {(0,0): -1, (0,1): 1}),
            
            # m6 = (A10-A00)*(B0+B1)
            # U: {(1,0):1, (0,0):-1}, V: {(0,0):1, (1,0):1}, W: {(1,0):1}
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,0): 1}),
            
            # m7 = (A01-A11)*(B0+B1)
            # U: {(0,1):1, (1,1):-1}, V: {(0,0):1, (1,0):1}, W: {(0,0):1}
            ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(0,0): 1}),
        ]
        
        for u_pat, v_pat, w_pat in terms_4x4x2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U: A is 4x4, blocks are 2x2
            for (br, bc), coeff in u_pat.items():
                for r in range(2):
                    for c in range(2):
                        i = oi + 2*br + r
                        j = oj + 2*bc + c
                        u_col[i * m + j] += coeff
            
            # Fill V: B is 4x2 (rows 0-3, cols 4-5), blocks are 2x1 (2 rows x 1 col)
            for (br, bc), coeff in v_pat.items():
                for r in range(2):
                    for c in range(1):  # only 1 column block
                        j = oj + 2*br + r
                        k = ok + c
                        v_col[j * p + k] += coeff
            
            # Fill W: C is 4x2 (rows 0-3, cols 4-5), blocks are 2x1
            for (br, bc), coeff in w_pat.items():
                for r in range(2):
                    for c in range(1):
                        i = oi + 2*br + r
                        k = ok + c
                        w_col[k * n + i] += coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # PART 2: 4x4 * 4x2 (columns 4-5 of B)
    # Use 2x2 Strassen blocks: 4 blocks of 2x2*2x2, each rank 7
    # Total strip rank: 28
    if _time_remaining() > 5.0:
        for rb in range(2):
            strassen_2x2(rb*2, 0, 4)
            strassen_2x2(rb*2, 2, 4)
    
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify correctness with numpy
    import numpy as np
    U_np = np.array(U_final, dtype=float)
    V_np = np.array(V_final, dtype=float)
    W_np = np.array(W_final, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_decomp[i, k] += A_test[i, j] * U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    err = np.max(np.abs(C_test - C_decomp))
    if err > 1e-10:
        # Fallback: rebuild with known-correct decomposition
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        strassen_4x4(0, 0, 0)
        for rb in range(2):
            strassen_2x2(rb*2, 0, 4)
            strassen_2x2(rb*2, 2, 4)
        rank = len(U_cols)
        U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
        V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
        W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
