"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Block partitions
    n_blocks = [2, 1]  # n=3
    m_blocks = [2, 2, 1]  # m=5
    p_blocks = [2, 2, 3]  # p=7
    
    # Compute offsets
    n_offsets = [0]
    for s in n_blocks[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks[:-1]:
        p_offsets.append(p_offsets[-1] + s)
    
    # Determine which blocks are 2x2x2 (Strassen-applicable)
    strassen_blocks = []
    for ii in range(len(n_blocks)):
        for jj in range(len(m_blocks)):
            for kk in range(len(p_blocks)):
                if n_blocks[ii] == 2 and m_blocks[jj] == 2 and p_blocks[kk] == 2:
                    strassen_blocks.append((ii, jj, kk))
    
    # Build U, V, W
    # Start with all zeros, we'll fill in
    # First, determine total rank
    rank = 0
    for ii in range(len(n_blocks)):
        for jj in range(len(m_blocks)):
            for kk in range(len(p_blocks)):
                if (ii, jj, kk) in strassen_blocks:
                    rank += 7
                else:
                    rank += n_blocks[ii] * m_blocks[jj] * p_blocks[kk]
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for ii in range(len(n_blocks)):
        for jj in range(len(m_blocks)):
            for kk in range(len(p_blocks)):
                if (ii, jj, kk) in strassen_blocks:
                    # Strassen decomposition for 2x2x2
                    ro = n_offsets[ii]  # row offset in A
                    co = m_offsets[jj]  # col offset in A / row offset in B
                    oo = p_offsets[kk]  # col offset in B / col offset in C
                    
                    # A sub-block: rows ro,ro+1; cols co,co+1
                    # B sub-block: rows co,co+1; cols oo,oo+1
                    # C sub-block: rows ro,ro+1; cols oo,oo+1
                    
                    # Strassen 7 terms
                    # Each term: (alpha · A_sub) * (beta · B_sub)
                    # A entries indexed by (r,c) in {0,1}x{0,1}
                    # B entries indexed by (r,c) in {0,1}x{0,1}
                    # C entries indexed by (r,c) in {0,1}x{0,1}
                    
                    # M1: (a+d)(e+h)
                    # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
                    a_alpha = [(0,0,1.0), (1,1,1.0)]  # (row, col, coeff)
                    b_alpha = [(0,0,1.0), (1,1,1.0)]
                    c_gamma = [(0,0,1.0), (1,1,1.0)]  # C[0,0]+=M1, C[1,1]+=M1
                    
                    # M2: (c+d)e
                    # c=A[1,0], d=A[1,1], e=B[0,0]
                    a_alpha = [(1,0,1.0), (1,1,1.0)]
                    b_alpha = [(0,0,1.0)]
                    c_gamma = [(1,0,1.0)]  # C[1,0]+=M2
                    
                    # M3: a(g-h)
                    # a=A[0,0], g=B[1,0], h=B[1,1]
                    a_alpha = [(0,0,1.0)]
                    b_alpha = [(1,0,1.0), (1,1,-1.0)]
                    c_gamma = [(0,1,1.0), (1,1,1.0)]  # C[0,1]+=M3, C[1,1]+=M3
                    
                    # M4: d(f-e)
                    # d=A[1,1], f=B[0,1], e=B[0,0]
                    a_alpha = [(1,1,1.0)]
                    b_alpha = [(0,1,1.0), (0,0,-1.0)]
                    c_gamma = [(0,0,1.0), (1,0,1.0)]  # C[0,0]+=M4, C[1,0]+=M4
                    
                    # M5: (a+b)h
                    # a=A[0,0], b=A[0,1], h=B[1,1]
                    a_alpha = [(0,0,1.0), (0,1,1.0)]
                    b_alpha = [(1,1,1.0)]
                    c_gamma = [(0,0,-1.0), (0,1,1.0)]  # C[0,0]-=M5, C[0,1]+=M5
                    
                    # M6: (c-a)(e+f)
                    # c=A[1,0], a=A[0,0], e=B[0,0], f=B[0,1]
                    a_alpha = [(1,0,1.0), (0,0,-1.0)]
                    b_alpha = [(0,0,1.0), (0,1,1.0)]
                    c_gamma = [(1,1,1.0)]  # C[1,1]+=M6
                    
                    # M7: (b-d)(g+h)
                    # b=A[0,1], d=A[1,1], g=B[1,0], h=B[1,1]
                    a_alpha = [(0,1,1.0), (1,1,-1.0)]
                    b_alpha = [(1,0,1.0), (1,1,1.0)]
                    c_gamma = [(0,0,1.0)]  # C[0,0]+=M7
                    
                    strassen_terms = [
                        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
                        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0)]),
                        ([(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
                        ([(1,1,1.0)], [(0,1,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
                        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
                        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
                        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
                    ]
                    
                    for a_parts, b_parts, c_parts in strassen_terms:
                        # Add to U
                        for (ar, ac, coeff) in a_parts:
                            row = (ro + ar) * m + (co + ac)
                            U[row][col] += coeff
                        # Add to V
                        for (br, bc, coeff) in b_parts:
                            row = (co + br) * p + (oo + bc)
                            V[row][col] += coeff
                        # Add to W
                        for (cr, cc, coeff) in c_parts:
                            # W index: k*n + i where k is col in C, i is row in C
                            row = (oo + cc) * n + (ro + cr)
                            W[row][col] += coeff
                        col += 1
                else:
                    # Schoolbook for this block
                    ns = n_blocks[ii]
                    ms = m_blocks[jj]
                    ps = p_blocks[kk]
                    ro = n_offsets[ii]
                    co = m_offsets[jj]
                    oo = p_offsets[kk]
                    for ar in range(ns):
                        for ac in range(ms):
                            for bc in range(ps):
                                # U: A[(ro+ar), (co+ac)]
                                U[(ro+ar)*m + (co+ac)][col] = 1.0
                                # V: B[(co+ac), (oo+bc)]
                                V[(co+ac)*p + (oo+bc)][col] = 1.0
                                # W: C[(ro+ar), (oo+bc)] → W[(oo+bc)*n + (ro+ar)]
                                W[(oo+bc)*n + (ro+ar)][col] = 1.0
                                col += 1
    
    # Convert to integer/half-integer if possible
    # All entries are integers or half-integers already
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
