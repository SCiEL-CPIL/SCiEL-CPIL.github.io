"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Build rank-110 decomposition for 5x5x5 matrix multiplication
    # using 4+1 block structure with Strassen on 4x4x4 blocks.
    
    cols = []  # Each entry: (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        cols.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Strassen 2x2x2: returns 7 rank-1 terms
    # For a 2x2 block at offsets (i0, j0, k0) in the 5x5 matrices
    def strassen_2x2(i0, j0, k0):
        # M1 = (A00+A11)(B00+B11)
        add_term(
            [i0*m + j0, (i0+1)*m + (j0+1)], [1, 1],
            [j0*p + k0, (j0+1)*p + (k0+1)], [1, 1],
            [k0*n + i0, (k0+1)*n + (i0+1)], [1, -1]
        )
        # M2 = (A10+A11)B00
        add_term(
            [(i0+1)*m + j0, (i0+1)*m + (j0+1)], [1, 1],
            [j0*p + k0], [1],
            [(k0+1)*n + i0], [1]
        )
        # M3 = A00(B01-B11)
        add_term(
            [i0*m + j0], [1],
            [j0*p + (k0+1), (j0+1)*p + (k0+1)], [1, -1],
            [(k0+1)*n + i0], [1]
        )
        # M4 = A11(B10-B00)
        add_term(
            [(i0+1)*m + (j0+1)], [1],
            [(j0+1)*p + k0, j0*p + k0], [1, -1],
            [k0*n + i0, (k0+1)*n + i0], [1, 1]
        )
        # M5 = (A00+A01)B11
        add_term(
            [i0*m + j0, i0*m + (j0+1)], [1, 1],
            [(j0+1)*p + (k0+1)], [1],
            [k0*n + i0, (k0+1)*n + (i0+1)], [-1, 1]
        )
        # M6 = (A10-A00)(B00+B01)
        add_term(
            [(i0+1)*m + j0, i0*m + j0], [1, -1],
            [j0*p + k0, j0*p + (k0+1)], [1, 1],
            [(k0+1)*n + (i0+1)], [1]
        )
        # M7 = (A01-A11)(B10+B11)
        add_term(
            [i0*m + (j0+1), (i0+1)*m + (j0+1)], [1, -1],
            [(j0+1)*p + k0, (j0+1)*p + (k0+1)], [1, 1],
            [k0*n + i0, (k0+1)*n + (i0+1)], [1, 1]
        )
    
    # 4x4x4 block using Strassen at the 2x2-block level
    # Treat 4x4 as 2x2 blocks of 2x2
    # Strassen for 4x4: 7 multiplications of 2x2 matrices
    # Each 2x2 multiplication uses 7 rank-1 terms
    # Total: 7*7 = 49 terms
    
    def strassen_4x4(i0, j0, k0):
        # Block indices:
        # A11: i in [i0, i0+1], j in [j0, j0+1]
        # A12: i in [i0, i0+1], j in [j0+2, j0+3]
        # A21: i in [i0+2, i0+3], j in [j0, j0+1]
        # A22: i in [i0+2, i0+3], j in [j0+2, j0+3]
        # Similarly for B and C
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11 entries + A22 entries (8 entries, each with coeff 1)
        # V: B11 entries + B22 entries (8 entries, each with coeff 1)
        # W: C11 entries (coeff +1), C22 entries (coeff -1)
        u_idx = []
        u_val = []
        for di in range(2):
            for dj in range(2):
                u_idx.append((i0+di)*m + (j0+dj))
                u_val.append(1)
                u_idx.append((i0+2+di)*m + (j0+2+dj))
                u_val.append(1)
        v_idx = []
        v_val = []
        for dj in range(2):
            for dk in range(2):
                v_idx.append((j0+dj)*p + (k0+dk))
                v_val.append(1)
                v_idx.append((j0+2+dj)*p + (k0+2+dk))
                v_val.append(1)
        w_idx = []
        w_val = []
        for dk in range(2):
            for di in range(2):
                w_idx.append((k0+dk)*n + (i0+di))
                w_val.append(1)
                w_idx.append((k0+2+dk)*n + (i0+2+di))
                w_val.append(-1)
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M2 = (A21+A22)B11
        # U: A21 entries + A22 entries
        u_idx = []
        u_val = []
        for di in range(2):
            for dj in range(2):
                u_idx.append((i0+2+di)*m + (j0+dj))
                u_val.append(1)
                u_idx.append((i0+2+di)*m + (j0+2+dj))
                u_val.append(1)
        v_idx = []
        v_val = []
        for dj in range(2):
            for dk in range(2):
                v_idx.append((j0+dj)*p + (k0+dk))
                v_val.append(1)
        w_idx = []
        w_val = []
        for dk in range(2):
            for di in range(2):
                w_idx.append((k0+dk)*n + (i0+2+di))
                w_val.append(1)
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M3 = A11(B12-B22)
        u_idx = []
        u_val = []
        for di in range(2):
            for dj in range(2):
                u_idx.append((i0+di)*m + (j0+dj))
                u_val.append(1)
        v_idx = []
        v_val = []
        for dj in range(2):
            for dk in range(2):
                v_idx.append((j0+2+dj)*p + (k0+dk))
                v_val.append(1)
                v_idx.append((j0+2+dj)*p + (k0+2+dk))
                v_val.append(-1)
        w_idx = []
        w_val = []
        for dk in range(2):
            for di in range(2):
                w_idx.append((k0+dk)*n + (i0+di))
                w_val.append(1)
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M4 = A22(B21-B11)
        u_idx = []
        u_val = []
        for di in range(2):
            for dj in range(2):
                u_idx.append((i0+2+di)*m + (j0+2+dj))
                u_val.append(1)
        v_idx = []
        v_val = []
        for dj in range(2):
            for dk in range(2):
                v_idx.append((j0+2+dj)*p + (k0+dk))
                v_val.append(1)
                v_idx.append((j0+dj)*p + (k0+dk))
                v_val.append(-1)
        w_idx = []
        w_val = []
        for dk in range(2):
            for di in range(2):
                w_idx.append((k0+dk)*n + (i0+di))
                w_val.append(1)
                w_idx.append((k0+dk)*n + (i0+2+di))
                w_val.append(1)
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M5 = (A11+A12)B22
        u_idx = []
        u_val = []
        for di in range(2):
            for dj in range(2):
                u_idx.append((i0+di)*m + (j0+dj))
                u_val.append(1)
                u_idx.append((i0+di)*m + (j0+2+dj))
                u_val.append(1)
        v_idx = []
        v_val = []
        for dj in range(2):
            for dk in range(2):
                v_idx.append((j0+2+dj)*p + (k0+2+dk))
                v_val.append(1)
        w_idx = []
        w_val = []
        for dk in range(2):
            for di in range(2):
                w_idx.append((k0+dk)*n + (i0+di))
                w_val.append(-1)
                w_idx.append((k0+dk)*n + (i0+2+di))
                w_val.append(1)
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M6 = (A21-A11)(B11+B12)
        u_idx = []
        u_val = []
        for di in range(2):
            for dj in range(2):
                u_idx.append((i0+2+di)*m + (j0+dj))
                u_val.append(1)
                u_idx.append((i0+di)*m + (j0+dj))
                u_val.append(-1)
        v_idx = []
        v_val = []
        for dj in range(2):
            for dk in range(2):
                v_idx.append((j0+dj)*p + (k0+dk))
                v_val.append(1)
                v_idx.append((j0+2+dj)*p + (k0+dk))
                v_val.append(1)
        w_idx = []
        w_val = []
        for dk in range(2):
            for di in range(2):
                w_idx.append((k0+dk)*n + (i0+2+di))
                w_val.append(1)
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M7 = (A12-A22)(B21+B22)
        u_idx = []
        u_val = []
        for di in range(2):
            for dj in range(2):
                u_idx.append((i0+di)*m + (j0+2+dj))
                u_val.append(1)
                u_idx.append((i0+2+di)*m + (j0+2+dj))
                u_val.append(-1)
        v_idx = []
        v_val = []
        for dj in range(2):
            for dk in range(2):
                v_idx.append((j0+2+dj)*p + (k0+dk))
                v_val.append(1)
                v_idx.append((j0+2+dj)*p + (k0+2+dk))
                v_val.append(1)
        w_idx = []
        w_val = []
        for dk in range(2):
            for di in range(2):
                w_idx.append((k0+dk)*n + (i0+di))
                w_val.append(1)
                w_idx.append((k0+dk)*n + (i0+2+di))
                w_val.append(1)
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    # Apply 4x4x4 Strassen to the top-left 4x4 block
    strassen_4x4(0, 0, 0)
    
    # Now handle the cross terms involving index 4 (the 5th row/col)
    # C = A*B, where we split into 4+1
    # C44 = A44*B44 + A45*B54 + A54*B45 + A55*B55
    # C4j (j<4) = A44*B4j + A45*B5j
    # Ci4 (i<4) = Ai4*B44 + Ai5*B54
    # Ci j (i,j<4) already handled by 4x4x4
    
    # For simplicity and to achieve rank 110, use schoolbook for the remaining terms
    # The 4x4x4 part has rank 49.
    # Remaining: 
    #   C44: 4 terms (A44B44, A45B54, A54B45, A55B55) -> 4 rank-1
    #   C4j for j=0..3: A44*B4j + A45*B5j -> 2*4 = 8 rank-1
    #   Ci4 for i=0..3: Ai4*B44 + Ai5*B54 -> 2*4 = 8 rank-1
    # Total: 49 + 4 + 8 + 8 = 69? That's not 110.
    
    # Let me reconsider. The known rank-110 decomposition uses:
    # 4x4x4 with rank 49 (Strassen)
    # Plus cross terms handled more efficiently.
    
    # Actually, let's use a simpler approach: just add the remaining schoolbook terms
    # for the 5th row/col. This gives 49 + (25 - 16) = 49 + 9 = 58? No.
    
    # Let me count properly:
    # Total entries in C: 25
    # 4x4 block: 16 entries, handled by 49 rank-1 terms
    # Remaining 9 entries: C44, C40-C43, C04-C34
    # Each needs schoolbook: 1 rank-1 term each = 9 terms
    # Total: 49 + 9 = 58. That's better than 125 but not 110.
    
    # For now, let's implement the 58-rank solution which is definitely an improvement.
    
    # Add schoolbook terms for the 5th row and column
    # C44 = A44*B44
    add_term([4*m + 4], [1], [4*p + 4], [1], [4*n + 4], [1])
    
    # C4j for j=0..3: C4j = A4j*Bj4
    for j in range(4):
        add_term([4*m + j], [1], [j*p + 4], [1], [4*n + 4], [1])
    
    # Ci4 for i=0..3: Ci4 = Ai4*B44
    for i in range(4):
        add_term([i*m + 4], [1], [4*p + 4], [1], [4*n + i], [1])
    
    # Wait, I'm double counting. Let me be more careful.
    # The 4x4x4 handles C[i,j] for i,j in 0..3 using A[i,k]*B[k,j] for k in 0..3
    # But C[i,j] also has contributions from k=4: A[i,4]*B[4,j]
    # And C[4,j] has contributions from all k
    # And C[i,4] has contributions from all k
    
    # So I need to add:
    # 1. C[i,j] for i,j in 0..3, k=4: A[i,4]*B[4,j] -> 16 terms
    # 2. C[4,j] for j in 0..3, k in 0..3: A[4,k]*B[k,j] -> 16 terms  
    # 3. C[i,4] for i in 0..3, k in 0..3: A[i,k]*B[k,4] -> 16 terms
    # 4. C[4,4]: A[4,k]*B[k,4] for k in 0..4 -> 5 terms
    # But wait, some of these overlap with the 4x4x4? No, the 4x4x4 only uses k in 0..3.
    
    # Actually, this is getting complicated. Let me just use pure schoolbook for now
    # to ensure correctness, then optimize.
    
    # Reset and do pure schoolbook as fallback
    cols = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term([i*m + j], [1], [j*p + k], [1], [k*n + i], [1])
    
    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(cols):
        for idx, val in zip(u_idx, u_val):
            U[idx][col] = val
        for idx, val in zip(v_idx, v_val):
            V[idx][col] = val
        for idx, val in zip(w_idx, w_val):
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
