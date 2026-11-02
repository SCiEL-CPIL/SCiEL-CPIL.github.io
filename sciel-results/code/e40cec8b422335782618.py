"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def strassen_2x2x2(U_off, V_off, W_off):
        """U_off=(a,b,c,d) rows, V_off=(e,f,g,h) rows, W_off=(C11,C12,C21,C22) rows"""
        a_r, b_r, c_r, d_r = U_off
        e_r, f_r, g_r, h_r = V_off
        C11_r, C12_r, C21_r, C22_r = W_off
        return [
            ([(a_r,1),(d_r,1)], [(e_r,1),(h_r,1)], [(C11_r,1),(C22_r,1)]),
            ([(c_r,1),(d_r,1)], [(e_r,1)], [(C12_r,1),(C21_r,-1)]),
            ([(a_r,1)], [(f_r,1),(h_r,-1)], [(C11_r,1),(C22_r,1)]),
            ([(d_r,1)], [(g_r,1),(e_r,-1)], [(C12_r,1),(C21_r,-1)]),
            ([(a_r,1),(b_r,1)], [(h_r,1)], [(C11_r,-1),(C12_r,1),(C21_r,1)]),
            ([(c_r,1),(b_r,-1)], [(f_r,1)], [(C11_r,1),(C21_r,-1)]),
            ([(a_r,1),(c_r,-1)], [(g_r,1)], [(C11_r,1),(C22_r,-1)]),
        ]
    
    def schoolbook_2x2x1(U_off, V_off, W_off):
        a_r, b_r, c_r, d_r = U_off
        e_r, f_r = V_off
        C1_r, C2_r = W_off
        return [
            ([(a_r,1)], [(e_r,1)], [(C1_r,1)]),
            ([(b_r,1)], [(f_r,1)], [(C1_r,1)]),
            ([(c_r,1)], [(e_r,1)], [(C2_r,1)]),
            ([(d_r,1)], [(f_r,1)], [(C2_r,1)]),
        ]
    
    all_terms = []
    for a_rp in range(2):
        for a_cp in range(2):
            for b_rp in range(2):
                a_r = 4*(2*a_rp) + (2*a_cp)
                b_r = 4*(2*a_rp) + (2*a_cp+1)
                c_r = 4*(2*a_rp+1) + (2*a_cp)
                d_r = 4*(2*a_rp+1) + (2*a_cp+1)
                U_off = (a_r, b_r, c_r, d_r)
                
                for col_range in [(0,4), (4,8), (8,9)]:
                    k0, k1 = col_range
                    nc = k1 - k0
                    if nc == 4:
                        for sub in range(2):
                            kk0 = k0 + 2*sub
                            kk1 = k0 + 2*sub + 1
                            e_r = (2*b_rp)*9 + kk0
                            f_r = (2*b_rp)*9 + kk1
                            g_r = (2*b_rp+1)*9 + kk0
                            h_r = (2*b_rp+1)*9 + kk1
                            V_off = (e_r, f_r, g_r, h_r)
                            C11_r = kk0*9 + (2*a_rp)
                            C12_r = kk1*9 + (2*a_rp)
                            C21_r = kk0*9 + (2*a_rp+1)
                            C22_r = kk1*9 + (2*a_rp+1)
                            W_off = (C11_r, C12_r, C21_r, C22_r)
                            all_terms.extend(strassen_2x2x2(U_off, V_off, W_off))
                    else:  # nc == 1
                        e_r = (2*b_rp)*9 + k0
                        f_r = (2*b_rp+1)*9 + k0
                        V_off = (e_r, f_r)
                        C1_r = k0*9 + (2*a_rp)
                        C2_r = k0*9 + (2*a_rp+1)
                        W_off = (C1_r, C2_r)
                        all_terms.extend(schoolbook_2x2x1(U_off, V_off, W_off))
    
    rank = len(all_terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for col, (ue, ve, we) in enumerate(all_terms):
        for r, c in ue: U[r][col] = c
        for r, c in ve: V[r][col] = c
        for r, c in we: W[r][col] = c
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
