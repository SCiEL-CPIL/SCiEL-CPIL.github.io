def _add_strassen_444(U, V, W, col, a_base, b_base, c_base):
    """Add rank-49 recursive Strassen for 4x4x4."""
    # 7 super-Strassen terms: (A_comb, B_comb, C_comb)
    # Each comb is a list of (block, coeff)
    # Block ordering: 0=TL(00), 1=TR(01), 2=BL(10), 3=BR(11)
    
    # Using the same structure as _STRASSEN_222 but at the block level
    # m1: (A00+A11)(B00+B11) -> C00+C11
    # m2: (A10+A11)B00 -> C10
    # m3: A00(B01-B11) -> C01
    # m4: A11(B10-B01) -> C00+C10  [from the existing code's variant]
    # m5: (A00+A10)B11 -> -C00+C10  [wait, W=[-1,1,0,0] means -C00+C10]
    # m6: (-A00+A10)(B00+B01) -> C11
    # m7: (A01-A11)(B10+B11) -> C00
    
    # Let me just use the exact same patterns from _STRASSEN_222 at the block level
    super_terms = [
        # (U_blocks, V_blocks, W_blocks) - each is 4-element list [TL,TR,BL,BR]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    for (u_comb, v_comb, w_comb) in super_terms:
        # For each of the 7 inner Strassen rank-1 terms
        for (u_pat, v_pat, w_pat) in _STRASSEN_222:
            for s in range(4):
                sr = s // 2
                sc = s % 2
                
                # U contributions
                for blk in range(4):
                    if u_comb[blk] != 0 and u_pat[s] != 0:
                        br = blk // 2
                        bc = blk % 2
                        u_idx = a_base + (br * 2 + sr) * 4 + (bc * 2 + sc)
                        U[u_idx][col] += u_comb[blk] * u_pat[s]
                
                # V contributions
                for blk in range(4):
                    if v_comb[blk] != 0 and v_pat[s] != 0:
                        br = blk // 2
                        bc = blk % 2
                        v_idx = b_base + (br * 2 + sr) * 4 + (bc * 2 + sc)
                        V[v_idx][col] += v_comb[blk] * v_pat[s]
                
                # W contributions
                for blk in range(4):
                    if w_comb[blk] != 0 and w_pat[s] != 0:
                        br = blk // 2
                        bc = blk % 2
                        w_idx = c_base + (br * 2 + sr) * 4 + (bc * 2 + sc)
                        W[w_idx][col] += w_comb[blk] * w_pat[s]
                
                col += 1
    
    return col
