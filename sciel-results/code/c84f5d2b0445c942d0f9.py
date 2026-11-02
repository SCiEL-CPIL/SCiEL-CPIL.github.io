"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 38 decomposition
    # 2x4x3 part (rank 24) + 2x4x2 part (rank 14)
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec, u_offset, v_offset, w_offset):
        # u_vec: length n*m (8)
        # v_vec: length m*p (28)
        # w_vec: length p*n (14)
        # We pass local vectors and offsets? 
        # Easier to pass full-size vectors with zeros.
        
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # --- Part 1: 2x4x3 ---
    # A is 2x4. B is 4x3. C is 2x3.
    # Split A into AL (cols 0-1), AR (cols 2-3)
    # Split B into BL (rows 0-1), BR (rows 2-3)
    # C = AL*BL + AR*BR
    
    # Helper for 2x2x3 schoolbook
    def gen_223():
        terms = []
        # A: a b / c d
        # B: e f g / h i j
        # C: ae+bh, af+bi, ag+bj / ce+dh, cf+di, cg+dj
        
        # 1. ae
        terms.append(( [1,0,0,0], [1,0,0,0,0,0], [1,0,0,0,0,0] )) # C00
        # 2. bh
        terms.append(( [0,1,0,0], [0,0,0,1,0,0], [1,0,0,0,0,0] )) # C00
        # 3. af
        terms.append(( [1,0,0,0], [0,1,0,0,0,0], [0,1,0,0,0,0] )) # C01
        # 4. bi
        terms.append(( [0,1,0,0], [0,0,1,0,0,0], [0,1,0,0,0,0] )) # C01
        # 5. ag
        terms.append(( [1,0,0,0], [0,0,0,0,1,0], [0,0,1,0,0,0] )) # C02
        # 6. bj
        terms.append(( [0,1,0,0], [0,0,0,0,0,1], [0,0,1,0,0,0] )) # C02
        # 7. ce
        terms.append(( [0,0,1,0], [1,0,0,0,0,0], [0,0,0,1,0,0] )) # C10
        # 8. dh
        terms.append(( [0,0,0,1], [0,0,0,1,0,0], [0,0,0,1,0,0] )) # C10
        # 9. cf
        terms.append(( [0,0,1,0], [0,1,0,0,0,0], [0,0,0,0,1,0] )) # C11
        # 10. di
        terms.append(( [0,0,0,1], [0,0,1,0,0,0], [0,0,0,0,1,0] )) # C11
        # 11. cg
        terms.append(( [0,0,1,0], [0,0,0,0,1,0], [0,0,0,0,0,1] )) # C12
        # 12. dj
        terms.append(( [0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1] )) # C12
        return terms

    # Helper for 2x2x2 Strassen
    def gen_222():
        terms = []
        # A: a b / c d
        # B: e f / g h
        # C: p q / r s
        
        # m1 = (a+d)(e+h) -> p+=1, s+=1
        terms.append(( [1,0,0,1], [1,0,0,1], [1,0,0,1] ))
        # m2 = (c+d)e -> r+=1, s-=1
        terms.append(( [0,0,1,1], [1,0,0,0], [0,0,1,-1] ))
        # m3 = a(f-h) -> q+=1, s+=1
        terms.append(( [1,0,0,0], [0,1,0,-1], [0,1,0,1] ))
        # m4 = d(g-e) -> p+=1, r+=1
        terms.append(( [0,0,0,1], [0,0,1,-1], [1,0,1,0] ))
        # m5 = (a+b)h -> p-=1, q+=1
        terms.append(( [1,1,0,0], [0,0,0,1], [-1,1,0,0] ))
        # m6 = (c-a)(e+f) -> s+=1
        terms.append(( [-1,0,1,0], [1,1,0,0], [0,0,0,1] ))
        # m7 = (b-d)(g+h) -> p+=1
        terms.append(( [0,1,-1,0], [0,0,1,1], [1,0,0,0] ))
        return terms

    # Add 2x4x3 terms
    # AL: A cols 0,1. BL: B rows 0,1. C cols 0,1,2.
    for u, v, w in gen_223():
        # Map u (len 4) to U (len 8): indices 0,1,2,3
        U_full = [0]*8
        U_full[0] = u[0]
        U_full[1] = u[1]
        U_full[2] = u[2]
        U_full[3] = u[3]
        
        # Map v (len 6) to V (len 28): B rows 0,1. B is 4x3.
        # V index: j*p + k. j in 0,1. k in 0,1,2.
        # v[0]=e (j=0,k=0), v[1]=f (j=0,k=1), v[2]=g (j=0,k=2)
        # v[3]=h (j=1,k=0), v[4]=i (j=1,k=1), v[5]=j (j=1,k=2)
        V_full = [0]*28
        for jj in range(2):
            for kk in range(3):
                idx_v = jj*3 + kk
                idx_V = jj*7 + kk
                V_full[idx_V] = v[idx_v]
                
        # Map w (len 6) to W (len 14): C is 2x3.
        # W index: k*n + i. k in 0,1,2. i in 0,1.
        # w[0]=C00 (i=0,k=0), w[1]=C01 (i=0,k=1), w[2]=C02 (i=0,k=2)
        # w[3]=C10 (i=1,k=0), w[4]=C11 (i=1,k=1), w[5]=C12 (i=1,k=2)
        W_full = [0]*14
        for kk in range(3):
            for ii in range(2):
                idx_w = kk*2 + ii
                idx_W = kk*2 + ii # n=2
                W_full[idx_W] = w[idx_w]
                
        add_term(U_full, V_full, W_full, 0, 0, 0)

    # AR: A cols 2,3. BR: B rows 2,3. C cols 0,1,2.
    for u, v, w in gen_223():
        U_full = [0]*8
        U_full[4] = u[0]
        U_full[5] = u[1]
        U_full[6] = u[2]
        U_full[7] = u[3]
        
        V_full = [0]*28
        for jj in range(2):
            for kk in range(3):
                idx_v = jj*3 + kk
                idx_V = (jj+2)*7 + kk
                V_full[idx_V] = v[idx_v]
                
        W_full = [0]*14
        for kk in range(3):
            for ii in range(2):
                idx_w = kk*2 + ii
                idx_W = kk*2 + ii
                W_full[idx_W] = w[idx_w]
                
        add_term(U_full, V_full, W_full, 0, 0, 0)

    # --- Part 2: 2x4x2 ---
    # A is 2x4. B is 4x2. C is 2x2.
    # B cols 3,4,5,6? No, B is 4x7.
    # Part 1 used B cols 0,1,2.
    # Part 2 uses B cols 3,4. (Width 2)
    # So B_sub is rows 0-3, cols 3-4.
    
    # AL: A cols 0,1. BL: B rows 0,1, cols 3,4.
    for u, v, w in gen_222():
        U_full = [0]*8
        U_full[0] = u[0]
        U_full[1] = u[1]
        U_full[2] = u[2]
        U_full[3] = u[3]
        
        # v is 2x2. B_sub is 2x2 (rows 0,1, cols 3,4).
        # V index: j*p + k. j in 0,1. k in 3,4.
        V_full = [0]*28
        for jj in range(2):
            for kk in range(2):
                idx_v = jj*2 + kk
                idx_V = jj*7 + (kk+3)
                V_full[idx_V] = v[idx_v]
                
        # w is 2x2. C is 2x2. C cols 0,1.
        # W index: k*n + i. k in 0,1. i in 0,1.
        W_full = [0]*14
        for kk in range(2):
            for ii in range(2):
                idx_w = kk*2 + ii
                idx_W = kk*2 + ii
                W_full[idx_W] = w[idx_w]
                
        add_term(U_full, V_full, W_full, 0, 0, 0)

    # AR: A cols 2,3. BR: B rows 2,3, cols 3,4.
    for u, v, w in gen_222():
        U_full = [0]*8
        U_full[4] = u[0]
        U_full[5] = u[1]
        U_full[6] = u[2]
        U_full[7] = u[3]
        
        V_full = [0]*28
        for jj in range(2):
            for kk in range(2):
                idx_v = jj*2 + kk
                idx_V = (jj+2)*7 + (kk+3)
                V_full[idx_V] = v[idx_v]
                
        W_full = [0]*14
        for kk in range(2):
            for ii in range(2):
                idx_w = kk*2 + ii
                idx_W = kk*2 + ii
                W_full[idx_W] = w[idx_w]
                
        add_term(U_full, V_full, W_full, 0, 0, 0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
