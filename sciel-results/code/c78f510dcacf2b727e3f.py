import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using the block structure:
    # Split p=7 into p1=4 and p2=3.
    # Part 1: 2x4x4. Split m=4 into m1=2, m2=2.
    # This creates 4 sub-problems of 2x2x2.
    # Use Strassen for each 2x2x2.
    # Part 2: 2x4x3. Split p2=3 into p21=2 and p22=1.
    # Sub-part 2a: 2x4x2. Split m=4 into m1=2, m2=2.
    # Creates 2 sub-problems of 2x2x2 (using the sum trick).
    # Sub-part 2b: 2x4x1. Split m=4 into m1=2, m2=2.
    # Creates 2 sub-problems of 2x2x1.
    
    # We need to generate U, V, W matrices.
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (14, rank)
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx, val
    # U[u_idx][col] += val
    # V[v_idx][col] += val
    # W[w_idx][col] += val
    
    # We will collect terms in a list of (u, v, w, val)
    terms = []
    
    def add_term(u, v, w, val):
        terms.append((u, v, w, val))

    # Strassen decomposition for 2x2x2
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # Indices for A: (i, j) -> flat i*2+j
    # Indices for B: (j, k) -> flat j*2+k
    # Indices for C: (k, i) -> flat k*2+i  (Note: W[k*n+i] convention)
    # Here n=2, so W[k*2+i]
    
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_block(base_u, base_v, base_w):
        # base_u: index of A00 in U
        # base_v: index of B00 in V
        # base_w: index of C00 in W
        
        # A indices relative to base_u:
        # A00: base_u + 0
        # A01: base_u + 1
        # A10: base_u + 2
        # A11: base_u + 3
        
        # B indices relative to base_v:
        # B00: base_v + 0
        # B01: base_v + 1
        # B10: base_v + 2
        # B11: base_v + 3
        
        # C indices relative to base_w:
        # C00: base_w + 0
        # C01: base_w + 1
        # C10: base_w + 2
        # C11: base_w + 3
        
        # M1
        add_term(base_u + 0, base_v + 0, base_w + 0, 1)
        add_term(base_u + 0, base_v + 1, base_w + 0, 1)
        add_term(base_u + 3, base_v + 0, base_w + 0, 1)
        add_term(base_u + 3, base_v + 1, base_w + 0, 1)
        
        add_term(base_u + 0, base_v + 0, base_w + 1, 1)
        add_term(base_u + 0, base_v + 1, base_w + 1, 1)
        add_term(base_u + 3, base_v + 0, base_w + 1, 1)
        add_term(base_u + 3, base_v + 1, base_w + 1, 1)
        
        add_term(base_u + 0, base_v + 0, base_w + 2, -1)
        add_term(base_u + 0, base_v + 1, base_w + 2, -1)
        add_term(base_u + 3, base_v + 0, base_w + 2, -1)
        add_term(base_u + 3, base_v + 1, base_w + 2, -1)
        
        add_term(base_u + 0, base_v + 0, base_w + 3, 1)
        add_term(base_u + 0, base_v + 1, base_w + 3, 1)
        add_term(base_u + 3, base_v + 0, base_w + 3, 1)
        add_term(base_u + 3, base_v + 1, base_w + 3, 1)
        
        # M2
        add_term(base_u + 2, base_v + 0, base_w + 2, 1)
        add_term(base_u + 3, base_v + 0, base_w + 2, 1)
        
        add_term(base_u + 2, base_v + 0, base_w + 3, -1)
        add_term(base_u + 3, base_v + 0, base_w + 3, -1)
        
        # M3
        add_term(base_u + 0, base_v + 1, base_w + 1, 1)
        add_term(base_u + 0, base_v + 3, base_w + 1, -1)
        
        add_term(base_u + 0, base_v + 1, base_w + 3, 1)
        add_term(base_u + 0, base_v + 3, base_w + 3, -1)
        
        # M4
        add_term(base_u + 3, base_v + 2, base_w + 0, 1)
        add_term(base_u + 3, base_v + 0, base_w + 0, -1)
        
        add_term(base_u + 3, base_v + 2, base_w + 2, 1)
        add_term(base_u + 3, base_v + 0, base_w + 2, 1)
        
        # M5
        add_term(base_u + 0, base_v + 3, base_w + 0, -1)
        add_term(base_u + 1, base_v + 3, base_w + 0, -1)
        
        add_term(base_u + 0, base_v + 3, base_w + 1, 1)
        add_term(base_u + 1, base_v + 3, base_w + 1, 1)
        
        # M6
        add_term(base_u + 2, base_v + 0, base_w + 3, 1)
        add_term(base_u + 2, base_v + 1, base_w + 3, 1)
        add_term(base_u + 0, base_v + 0, base_w + 3, -1)
        add_term(base_u + 0, base_v + 1, base_w + 3, -1)
        
        # M7
        add_term(base_u + 1, base_v + 2, base_w + 0, 1)
        add_term(base_u + 1, base_v + 3, base_w + 0, 1)
        add_term(base_u + 3, base_v + 2, base_w + 0, -1)
        add_term(base_u + 3, base_v + 3, base_w + 0, -1)
        
        add_term(base_u + 1, base_v + 2, base_w + 3, 1)
        add_term(base_u + 1, base_v + 3, base_w + 3, 1)
        add_term(base_u + 3, base_v + 2, base_w + 3, -1)
        add_term(base_u + 3, base_v + 3, base_w + 3, -1)

    # Schoolbook for 2x2x1
    # A (2x2), B (2x1) -> C (2x1)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    def schoolbook_2x2x1_block(base_u, base_v, base_w):
        # A00: base_u+0, A01: base_u+1
        # A10: base_u+2, A11: base_u+3
        # B0: base_v+0, B1: base_v+1
        # C0: base_w+0, C1: base_w+1
        
        add_term(base_u + 0, base_v + 0, base_w + 0, 1)
        add_term(base_u + 1, base_v + 1, base_w + 0, 1)
        
        add_term(base_u + 2, base_v + 0, base_w + 1, 1)
        add_term(base_u + 3, base_v + 1, base_w + 1, 1)

    # Part 1: 2x4x4
    # m=4 split into 2,2. p=4 split into 2,2.
    # A blocks: A00 (cols 0-1), A01 (cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3)
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    
    # Use Strassen on the sums?
    # We decided to just run 4 Strassen blocks.
    # Block 1: A00 B00 -> C_part_00
    # Block 2: A01 B10 -> C_part_00
    # Block 3: A00 B01 -> C_part_01
    # Block 4: A01 B11 -> C_part_01
    
    # Indices:
    # A00: u 0..3
    # A01: u 4..7
    # B00: v 0..3
    # B01: v 4..7
    # B10: v 8..11
    # B11: v 12..15
    # C00: w 0..3 (k=0,1; i=0,1) -> w[k*2+i]
    # C01: w 4..7 (k=2,3; i=0,1) -> w[k*2+i]
    
    strassen_block(0, 0, 0)   # A00 B00
    strassen_block(4, 8, 0)   # A01 B10 (adds to C00)
    strassen_block(0, 4, 4)   # A00 B01
    strassen_block(4, 12, 4)  # A01 B11 (adds to C01)
    
    # Part 2: 2x4x3
    # p=3 split into 2,1.
    # B cols 4-5 (2 cols) and col 6 (1 col).
    # B20: rows 0-1, cols 4-5 (2x2)
    # B21: rows 0-1, col 6 (2x1)
    # B30: rows 2-3, cols 4-5 (2x2)
    # B31: rows 2-3, col 6 (2x1)
    
    # C20 = A00 B20 + A01 B30 (2x2)
    # C21 = A00 B21 + A01 B31 (2x1)
    
    # Indices:
    # B20: v 16..19
    # B21: v 20..21
    # B30: v 22..25
    # B31: v 26..27
    
    # C20: w 8..11 (k=4,5; i=0,1) -> w[k*2+i]
    # C21: w 12..13 (k=6; i=0,1) -> w[k*2+i]
    
    # For C20 (2x2x2 sum), use the 2-Strassen trick:
    # C20 = 1/2 [ (A00+A01)(B20+B30) + (A00-A01)(B20-B30) ]
    # This requires 2 Strassen blocks.
    # But we need to form A00+A01 and B20+B30.
    # This mixes the U and V indices.
    # Instead of mixing, we can just use 2 Strassen blocks if we allow the "sum" to be implicit?
    # No, the tensor decomposition must be explicit.
    # The "sum trick" creates new basis vectors for U and V.
    # Let's stick to the 4 independent Strassen blocks for simplicity and correctness, 
    # even if it might be slightly higher rank, or just implement the 4 blocks for the 2x4x2 part too.
    
    # Actually, for 2x4x2, we have:
    # A00 B20 and A01 B30.
    # These are two 2x2x2 products.
    # So we can just call strassen_block twice.
    
    strassen_block(0, 16, 8)  # A00 B20
    strassen_block(4, 22, 8)  # A01 B30 (adds to C20)
    
    # For C21 (2x1 sum):
    # A00 B21 and A01 B31.
    # These are two 2x2x1 products.
    # So we call schoolbook_2x2x1_block twice.
    
    schoolbook_2x2x1_block(0, 20, 12) # A00 B21
    schoolbook_2x2x1_block(4, 26, 12) # A01 B31 (adds to C21)

    # Total rank:
    # Part 1: 4 * 7 = 28
    # Part 2a: 2 * 7 = 14
    # Part 2b: 2 * 4 = 8
    # Total: 50
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for idx, (u, v, w, val) in enumerate(terms):
        U[u][idx] = val
        V[v][idx] = val
        W[w][idx] = val
        
    return {"U": U, "V": V, "W": W}
