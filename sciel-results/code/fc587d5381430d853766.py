"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Partition:
    # A rows: [0,1] (top), [2] (bottom)
    # A cols / B rows: [0,1] (left), [2,3] (right)
    # B cols / C cols: [0,1] (left), [2,3] (mid), [4,5,6] (right)
    # C rows: [0,1] (top), [2] (bottom)
    
    # Block dimensions:
    # A[0:2, 0:2] (2x2), A[0:2, 2:4] (2x2)
    # A[2:3, 0:2] (1x2), A[2:3, 2:4] (1x2)
    # B[0:2, 0:2] (2x2), B[0:2, 2:4] (2x2), B[0:2, 4:7] (2x3)
    # B[2:4, 0:2] (2x2), B[2:4, 2:4] (2x2), B[2:4, 4:7] (2x3)
    # C[0:2, 0:2] (2x2), C[0:2, 2:4] (2x2), C[0:2, 4:7] (2x3)
    # C[2:3, 0:2] (1x2), C[2:3, 2:4] (1x2), C[2:3, 4:7] (1x3)
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]  (2x2*2x2 + 2x2*2x2)
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]  (2x2*2x2 + 2x2*2x2)
    # C[0:2, 4:7] = A[0:2,0:2]*B[0:2,4:7] + A[0:2,2:4]*B[2:4,4:7]  (2x2*2x3 + 2x2*2x3)
    # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]  (1x2*2x2 + 1x2*2x2)
    # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]  (1x2*2x2 + 1x2*2x2)
    # C[2:3, 4:7] = A[2:3,0:2]*B[0:2,4:7] + A[2:3,2:4]*B[2:4,4:7]  (1x2*2x3 + 1x2*2x3)
    
    # For 2x2*2x2 blocks, use Strassen (rank 7 each).
    # For 2x2*2x3 blocks, schoolbook (rank 12 each).
    # For 1x2*2x2 blocks, schoolbook (rank 4 each).
    # For 1x2*2x3 blocks, schoolbook (rank 6 each).
    
    # Count:
    # 2x2*2x2 blocks: 4 (C00, C01, C20, C21 each have 2 such terms) = 4*2 = 8 blocks? No.
    # C[0:2,0:2] has 2 terms of 2x2*2x2
    # C[0:2,2:4] has 2 terms of 2x2*2x2
    # C[2:3,0:2] has 2 terms of 1x2*2x2
    # C[2:3,2:4] has 2 terms of 1x2*2x2
    # C[0:2,4:7] has 2 terms of 2x2*2x3
    # C[2:3,4:7] has 2 terms of 1x2*2x3
    
    # Strassen for 2x2*2x2: 4 blocks * 7 = 28
    # Schoolbook for 2x2*2x3: 2 blocks * 12 = 24
    # Schoolbook for 1x2*2x2: 4 blocks * 4 = 16
    # Schoolbook for 1x2*2x3: 2 blocks * 6 = 12
    # Total: 28 + 24 + 16 + 12 = 80
    
    # Great, rank 80!
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add a single rank-1 term."""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[u_idx] = u_val
        v[v_idx] = v_val
        w[w_idx] = w_val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def add_schoolbook_block(ar, ac, br, bc):
        """
        Add schoolbook decomposition for A[ar, ac] * B[br, bc] -> C[ar, bc]
        ar: (r_start, r_end) for A rows and C rows
        ac: (c_start, c_end) for A cols and B rows
        bc: (c_start, c_end) for B cols and C cols
        """
        for i in range(ar[0], ar[1]):
            for j in range(ac[0], ac[1]):
                for k in range(bc[0], bc[1]):
                    add_rank1(i * m + j, 1, j * p + k, 1, k * n + i, 1)
    
    def add_strassen_block(ar, ac, br, bc):
        """
        Add Strassen decomposition for A[ar, ac] * B[br, bc] -> C[ar, bc]
        where all blocks are 2x2.
        ar = (r0, r1), ac = (c0, c1), br = (r0, r1), bc = (c0, c1)
        with r1-r0 = 2, c1-c0 = 2.
        """
        r0, r1 = ar
        c0, c1 = ac
        # A indices: A[r0,c0], A[r0,c0+1], A[r0+1,c0], A[r0+1,c0+1]
        # B indices: B[c0,c0], B[c0,c0+1], B[c0+1,c0], B[c0+1,c0+1]
        # C indices: C[r0,c0], C[r0,c0+1], C[r0+1,c0], C[r0+1,c0+1]
        
        a00 = r0 * m + c0
        a01 = r0 * m + c0 + 1
        a10 = (r0+1) * m + c0
        a11 = (r0+1) * m + c0 + 1
        
        b00 = c0 * p + c0
        b01 = c0 * p + c0 + 1
        b10 = (c0+1) * p + c0
        b11 = (c0+1) * p + c0 + 1
        
        c00 = c0 * n + r0
        c01 = c0 * n + r0 + 1  # Wait, W[k*n+i], so C[r,c] -> W[c*n+r]
        c10 = (c0+1) * n + r0
        c11 = (c0+1) * n + r0 + 1
        
        # Strassen:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A11) * (B00 + B11) ... wait, that's M1 again.
        
        # Let me use the standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a+b)(h)
        # M4 = (d-c)(g+h)
        # M5 = (a-c)(e+f)
        # M6 = b(f-h)
        # M7 = c(g-e)
        #
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M6 - M1 + M4
        # C10 = M2 + M4 - M5 + M7
        # C11 = M1 - M3 + M2 - M6
        
        # But I need to express these as linear combinations of A and B elements.
        # Each M is a product of a linear combo of A's and a linear combo of B's.
        # So M = (sum_ia A_ia) * (sum_jb B_jb) = sum_ia sum_jb A_ia * B_jb
        # This is a rank-(number of terms) term, but actually it's a single "product" 
        # in the bilinear sense, which corresponds to rank 1 in the tensor decomposition 
        # only if it's a single A element times a single B element.
        
        # Wait, I'm confusing myself. In the tensor decomposition for matrix multiplication,
        # each rank-1 term is of the form U_vec * V_vec * W_vec where U_vec picks out 
        # specific (i,j) pairs, V_vec picks out specific (j,k) pairs, and W_vec picks out 
        # specific (k,i) pairs.
        
        # For Strassen, each M is a product of a linear combination of A entries and a 
        # linear combination of B entries. This corresponds to a sum of rank-1 terms.
        # For example, M1 = (A00 + A11)(B00 + B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # That's 4 rank-1 terms, not 1.
        
        # So Strassen doesn't directly give a rank reduction in the tensor decomposition 
        # sense unless we can combine the W vectors.
        
        # Actually, the key insight is that in Strassen, the same M is used to compute 
        # multiple C entries. So we can factor it out.
        
        # M1 contributes to C00, C01, C10, C11 with coefficients +1, -1, +1, -1 respectively.
        # So we can have a single "term" that is M1, and then the W vector has entries at 
        # positions corresponding to C00, C01, C10, C11 with values +1, -1, +1, -1.
        
        # But M1 itself is (A00+A11)(B00+B11), which is a product of two linear combos.
        # In the tensor decomposition, this would be represented as:
        # U vector: has 1 at positions for A00 and A11 (i.e., (r0,c0) and (r0+1,c0+1))
        # V vector: has 1 at positions for B00 and B11 (i.e., (c0,c0) and (c0+1,c0+1))
        # W vector: has coefficients for the C entries.
        
        # Wait, that's not quite right either. The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For a rank-1 term, U[i,j,r] is nonzero for specific (i,j), V[j,k,r] for specific (j,k), 
        # and W[k,i,r] for specific (k,i).
        
        # If U has 1 at (r0,c0) and (r0+1,c0+1), and V has 1 at (c0,c0) and (c0+1,c0+1),
        # then the product U[i,j,r]*V[j,k,r] is nonzero when (i,j) is in {(r0,c0), (r0+1,c0+1)} 
        # AND (j,k) is in {(c0,c0), (c0+1,c0+1)}.
        # This gives terms for (i,j,k) where i=r0, j=c0, k=c0; i=r0, j=c0+1, k=c0+1; 
        # i=r0+1, j=c0, k=c0; i=r0+1, j=c0+1, k=c0+1.
        # But we need i=r0, j=c0, k=c0+1 and i=r0+1, j=c0, k=c0+1 etc. for the cross terms.
        
        # So a single rank-1 term can't capture (A00+A11)(B00+B11) because it would 
        # include unwanted cross terms.
        
        # I think I've been overthinking this. Let me just verify: for the standard 
        # matrix multiplication tensor, the rank is at least n*m*p for schoolbook, and 
        # Strassen shows it's at most 7 for 2x2*2x2. The way Strassen works is that 
        # it uses 7 bilinear forms (products of linear combos) to compute all 8 outputs.
        # Each bilinear form corresponds to one "rank" in the decomposition.
        
        # So for a 2x2*2x2 block, we can use 7 rank-1 terms in the global decomposition.
        
        # Let me just carefully write out the 7 terms for Strassen.
        
        # For A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
        # M1 = (a+d)(e+h) = ae + ah + de + dh
        # M2 = (c+d)(e) = ce + de
        # M3 = (a+b)(h) = ah + bh
        # M4 = (d-c)(g+h) = dg + dh - cg - ch
        # M5 = (a-c)(e+f) = ae + af - ce - cf
        # M6 = b(f-h) = bf - bh
        # M7 = (a-c)(e+f) ... wait, M5 and M7 are the same?
        
        # Let me look up the correct Strassen formulas.
        # Standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a+b)(h)
        # M4 = (d-c)(g+h)
        # M5 = (a-c)(e+f)
        # M6 = b(f-h)
        # M7 = c(g-e)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M6 - M1 + M4
        # C10 = M2 + M4 - M5 + M7
        # C11 = M1 - M3 + M2 - M6
        
        # Let me verify C00:
        # M1 - M4 + M5 - M7
        # = (ae+ah+de+dh) - (dg+dh-cg-ch) + (ae+af-ce-cf) - (cg-ce)
        # = ae+ah+de+dh - dg - dh + cg + ch + ae+af-ce-cf - cg + ce
        # = 2ae + ah + de - dg + ch + af - ce - cf + ce
        # = 2ae + ah + de - dg + ch + af - cf
        # That's not equal to ae + bg. So my formula is wrong.
        
        # Let me use the correct Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a+b)(h)
        # M4 = (d-c)(g+h)
        # M5 = (a-c)(e+f)
        # M6 = b(f-h)
        # M7 = c(g-e)
        
        # C00 = M1 - M4 + M5 - M7
        # Let me expand each:
        # M1 = ae + ah + de + dh
        # M4 = dg + dh - cg - ch
        # M5 = ae + af - ce - cf
        # M7 = cg - ce
        # M1 - M4 + M5 - M7 = (ae+ah+de+dh) - (dg+dh-cg-ch) + (ae+af-ce-cf) - (cg-ce)
        # = ae + ah + de + dh - dg - dh + cg + ch + ae + af - ce - cf - cg + ce
        # = 2ae + ah + de - dg + ch + af - cf
        # Still not right.
        
        # I think the issue is that I'm not using the right formula. Let me just use a 
        # known correct Strassen implementation and verify it numerically.
        
        # For now, let me just return the schoolbook rank 84 and make sure it's correct.
        
    # Just return schoolbook for now
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
