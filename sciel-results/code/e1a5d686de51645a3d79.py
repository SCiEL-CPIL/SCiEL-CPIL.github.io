"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Build the full tensor T[i,j,k] = 1 for all i<m... 
    # Actually T[i,j,k] corresponds to C[i,k] += A[i,j]*B[j,k]
    # So the tensor has entries T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n + i, c] is the output coefficient
    
    # Start with schoolbook rank = n*m*p = 100
    # Try to reduce using Strassen on 2x2x2 sub-blocks
    
    # Strategy: partition indices and apply Strassen where possible
    # For a 2x2x2 block: indices i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    # Strassen uses 7 terms instead of 8
    
    # Let's identify all 2x2x2 sub-blocks and replace their 8 schoolbook terms with 7 Strassen terms
    # But we need to be careful about overlapping blocks
    
    # Approach: Greedily find non-overlapping 2x2x2 blocks and apply Strassen
    # n=4, m=5, p=5
    # Possible 2x2x2 blocks: i in {0,2} (start), j in {0,2,3} (start), k in {0,2,3} (start)
    # i starts: 0, 2 (since n=4, can start at 0 or 2)
    # j starts: 0, 2, 3 (since m=5, can start at 0,2,3)
    # k starts: 0, 2, 3 (since p=5, can start at 0,2,3)
    
    # Total possible blocks: 2 * 3 * 3 = 18
    # Each block saves 1 rank (8 -> 7)
    # But blocks can't overlap in the same (i,j,k) position
    
    # Let's find a set of non-overlapping 2x2x2 blocks
    # Block (i0,j0,k0) covers i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    
    # Greedy: try to pack as many non-overlapping blocks as possible
    used = set()  # set of (i,j,k) tuples that are used
    
    def block_positions(i0, j0, k0):
        return [(i0+di, j0+dj, k0+dk) for di in range(2) for dj in range(2) for dk in range(2)]
    
    # Generate all candidate blocks
    candidates = []
    for i0 in range(0, n-1):
        for j0 in range(0, m-1):
            for k0 in range(0, p-1):
                candidates.append((i0, j0, k0))
    
    # Greedily select non-overlapping blocks
    selected_blocks = []
    for (i0, j0, k0) in candidates:
        positions = block_positions(i0, j0, k0)
        if all(pos not in used for pos in positions):
            selected_blocks.append((i0, j0, k0))
            used.update(positions)
    
    # Now build the decomposition
    # Start with schoolbook, then for each selected block, replace 8 terms with 7 Strassen terms
    
    # Schoolbook terms: for each (i,j,k), one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # Strassen for block (i0,j0,k0): 
    #   Let a00=A[i0,j0], a01=A[i0,j0+1], a10=A[i0+1,j0], a11=A[i0+1,j0+1]
    #   Let b00=B[j0,k0], b01=B[j0,k0+1], b10=B[j0+1,k0], b11=B[j0+1,k0+1]
    #   c00 = (a00+a11)*(b00+b11) - (a10+a11)*(b10+b11) + (a00+a10)*(b00+b10) - a00*b00
    #   Wait, standard Strassen:
    #   p1 = (a00+a11)*(b00+b11)
    #   p2 = (a10+a11)*b00
    #   p3 = a00*(b01-b11)
    #   p4 = a11*(b10-b00)
    #   p5 = (a00+a01)*(b11+b10)  -- wait, need to be careful with indices
    #   
    #   Standard 2x2 Strassen:
    #   C = A*B where A,B are 2x2
    #   p1 = (A00+A11)*(B00+B11)
    #   p2 = (A10+A11)*B00
    #   p3 = A00*(B01-B11)
    #   p4 = A11*(B10-B00)
    #   p5 = (A00+A01)*B11
    #   p6 = (A10-A00)*(B00+B01)
    #   p7 = (A01-A11)*(B10+B11)
    #   C00 = p1 - p4 + p6 - p2
    #   C01 = p3 + p5
    #   C10 = p2 + p4
    #   C11 = p1 - p3 + p5 - p7
    #
    #   Each pk is a product of two linear combinations.
    #   In tensor decomposition, each pk corresponds to one rank-1 term.
    #   The linear combinations define the U, V, W vectors.
    
    # Let me build this properly
    # For each Strassen term pk in block (i0,j0,k0):
    #   U vector: has entries at positions corresponding to A's linear combination
    #   V vector: has entries at positions corresponding to B's linear combination  
    #   W vector: has entries at positions corresponding to C's linear combination
    
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For Strassen block (i0, j0, k0):
    # A indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
    #   U positions: i0*m+j0, i0*m+(j0+1), (i0+1)*m+j0, (i0+1)*m+(j0+1)
    # B indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
    #   V positions: j0*p+k0, j0*p+(k0+1), (j0+1)*p+k0, (j0+1)*p+(k0+1)
    # C indices: (k0,i0), (k0,i0+1), (k0+1,i0), (k0+1,i0+1)
    #   W positions: k0*n+i0, k0*n+(i0+1), (k0+1)*n+i0, (k0+1)*n+(i0+1)
    
    # Strassen terms (each is a product of a linear combo of A entries and a linear combo of B entries,
    # contributing to a linear combo of C entries):
    
    # p1 = (A00+A11)*(B00+B11) -> contributes to C00(+1), C11(+1)
    # p2 = (A10+A11)*B00 -> contributes to C00(-1), C10(+1)
    # p3 = A00*(B01-B11) -> contributes to C01(+1), C11(-1)
    # p4 = A11*(B10-B00) -> contributes to C00(-1), C10(+1)
    # p5 = (A00+A01)*B11 -> contributes to C01(+1), C11(+1)
    # p6 = (A10-A00)*(B00+B01) -> contributes to C00(+1), C10(-1)  -- wait let me recheck
    # p7 = (A01-A11)*(B10+B11) -> contributes to C11(-1), C10(-1)  -- need to verify
    
    # Let me re-derive Strassen carefully:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C00 = p1 - p4 + p6 - p2
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p3 + p5 - p7
    
    # So for block (i0,j0,k0), mapping:
    # a = A[i0,j0], b = A[i0,j0+1], c = A[i0+1,j0], d = A[i0+1,j0+1]
    # e = B[j0,k0], f = B[j0,k0+1], g = B[j0+1,k0], h = B[j0+1,k0+1]
    # C[i0,k0] = C00, C[i0,k0+1] = C01, C[i0+1,k0] = C10, C[i0+1,k0+1] = C11
    
    # U positions for a,b,c,d: i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
    # V positions for e,f,g,h: j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
    # W positions for C00,C01,C10,C11: k0*n+i0, k0*n+i0+1, (k0+1)*n+i0, (k0+1)*n+i0+1
    
    # p1: U: a(+1), d(+1); V: e(+1), h(+1); W: C00(+1), C11(+1)
    # p2: U: c(+1), d(+1); V: e(+1); W: C00(-1), C10(+1)
    # p3: U: a(+1); V: f(+1), h(-1); W: C01(+1), C11(-1)
    # p4: U: d(+1); V: g(+1), e(-1); W: C00(-1), C10(+1)
    # p5: U: a(+1), b(+1); V: h(+1); W: C01(+1), C11(+1)
    # p6: U: c(+1), a(-1); V: e(+1), f(+1); W: C00(+1), C10(-1)
    # p7: U: b(+1), d(-1); V: g(+1), h(+1); W: C11(-1), C10(-1)
    
    # Wait, let me recheck p6 and p7 contributions:
    # C00 = p1 - p4 + p6 - p2: p6 contributes +1 to C00
    # C10 = p2 + p4: p6 does NOT contribute to C10
    # Hmm, I think I made an error. Let me recheck.
    # C00 = p1 - p4 + p6 - p2
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p3 + p5 - p7
    
    # So p6 only appears in C00. p7 only appears in C11.
    # p6: W: C00(+1)
    # p7: W: C11(-1)
    
    # Let me redo:
    # p1: U:{a:+1, d:+1}, V:{e:+1, h:+1}, W:{C00:+1, C11:+1}
    # p2: U:{c:+1, d:+1}, V:{e:+1}, W:{C00:-1, C10:+1}
    # p3: U:{a:+1}, V:{f:+1, h:-1}, W:{C01:+1, C11:-1}
    # p4: U:{d:+1}, V:{g:+1, e:-1}, W:{C00:-1, C10:+1}
    # p5: U:{a:+1, b:+1}, V:{h:+1}, W:{C01:+1, C11:+1}
    # p6: U:{c:+1, a:-1}, V:{e:+1, f:+1}, W:{C00:+1}
    # p7: U:{b:+1, d:-1}, V:{g:+1, h:+1}, W:{C11:-1}
    
    # Now let's build the decomposition
    rank = n * m * p  # start with schoolbook
    # We'll build U, V, W as lists of lists
    
    # First, mark which (i,j,k) positions are covered by Strassen blocks
    strassen_covered = set()
    for (i0, j0, k0) in selected_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    strassen_covered.add((i0+di, j0+dj, k0+dk))
    
    # Schoolbook terms for non-covered positions
    U = []
    V = []
    W = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    u_vec = [0] * (n*m)
                    v_vec = [0] * (m*p)
                    w_vec = [0] * (p*n)
                    u_vec[i*m+j] = 1
                    v_vec[j*p+k] = 1
                    w_vec[k*n+i] = 1
                    U.append(u_vec)
                    V.append(v_vec)
                    W.append(w_vec)
    
    # Strassen terms for each selected block
    for (i0, j0, k0) in selected_blocks:
        # Define U, V, W positions
        ua = i0*m + j0
        ub = i0*m + (j0+1)
        uc = (i0+1)*m + j0
        ud = (i0+1)*m + (j0+1)
        ve = j0*p + k0
        vf = j0*p + (k0+1)
        vg = (j0+1)*p + k0
        vh = (j0+1)*p + (k0+1)
        wC00 = k0*n + i0
        wC01 = k0*n + (i0+1)
        wC10 = (k0+1)*n + i0
        wC11 = (k0+1)*n + (i0+1)
        
        # p1
        u1 = [0]*(n*m); u1[ua]=1; u1[ud]=1
        v1 = [0]*(m*p); v1[ve]=1; v1[vh]=1
        w1 = [0]*(p*n); w1[wC00]=1; w1[wC11]=1
        U.append(u1); V.append(v1); W.append(w1)
        
        # p2
        u2 = [0]*(n*m); u2[uc]=1; u2[ud]=1
        v2 = [0]*(m*p); v2[ve]=1
        w2 = [0]*(p*n); w2[wC00]=-1; w2[wC10]=1
        U.append(u2); V.append(v2); W.append(w2)
        
        # p3
        u3 = [0]*(n*m); u3[ua]=1
        v3 = [0]*(m*p); v3[vf]=1; v3[vh]=-1
        w3 = [0]*(p*n); w3[wC01]=1; w3[wC11]=-1
        U.append(u3); V.append(v3); W.append(w3)
        
        # p4
        u4 = [0]*(n*m); u4[ud]=1
        v4 = [0]*(m*p); v4[vg]=1; v4[ve]=-1
        w4 = [0]*(p*n); w4[wC00]=-1; w4[wC10]=1
        U.append(u4); V.append(v4); W.append(w4)
        
        # p5
        u5 = [0]*(n*m); u5[ua]=1; u5[ub]=1
        v5 = [0]*(m*p); v5[vh]=1
        w5 = [0]*(p*n); w5[wC01]=1; w5[wC11]=1
        U.append(u5); V.append(v5); W.append(w5)
        
        # p6
        u6 = [0]*(n*m); u6[uc]=1; u6[ua]=-1
        v6 = [0]*(m*p); v6[ve]=1; v6[vf]=1
        w6 = [0]*(p*n); w6[wC00]=1
        U.append(u6); V.append(v6); W.append(w6)
        
        # p7
        u7 = [0]*(n*m); u7[ub]=1; u7[ud]=-1
        v7 = [0]*(m*p); v7[vg]=1; v7[vh]=1
        w7 = [0]*(p*n); w7[wC11]=-1
        U.append(u7); V.append(v7); W.append(w7)
    
    final_rank = len(U)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
