import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    def add_strassen_222(u_base, v_base, w_base):
        # u_base: index of A[0,0] in U (i*m + j)
        # v_base: index of B[0,0] in V (j*p + k)
        # w_base: index of C[0,0] in W (k*n + i)
        # A is 2x2, B is 2x2, C is 2x2
        # A indices: u_base, u_base+1, u_base+m, u_base+m+1
        # B indices: v_base, v_base+1, v_base+p, v_base+p+1
        # C indices: w_base, w_base+1, w_base+n, w_base+n+1
        
        # M1 = (A00+A11)(B00+B11)
        add_term(u_base, v_base, w_base, 1) # A00 B00
        add_term(u_base, v_base+p, w_base, 1) # A00 B11 -- Wait, Strassen terms are linear in A and B separately?
        # No, the decomposition is a sum of products of linear forms.
        # The term is (A00+A11) * (B00+B11).
        # This expands to 4 products.
        # But in the tensor decomposition, each "column" corresponds to one product of a U-vector, V-vector, W-vector.
        # So we add 4 terms for M1?
        # No, the rank is 7.
        # The 7 terms are the products of the linear forms.
        # Term 1: (A00+A11) * (B00+B11).
        # U part: A00 and A11 are 1.
        # V part: B00 and B11 are 1.
        # W part: C00, C11 get contributions.
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So W[0,0] gets +1, W[1,1] gets +1.
        
        # Let's implement the 7 terms explicitly.
        
        # T1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[u_base][col] += 1
        U[u_base+m][col] += 1
        V[v_base][col] += 1
        V[v_base+p][col] += 1
        W[w_base][col] += 1
        W[w_base+n+1][col] += 1
        col += 1

        # T2: (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1
        U[u_base+m][col] += 1
        U[u_base+m+1][col] += 1
        V[v_base][col] += 1
        W[w_base+n][col] += 1
        col += 1

        # T3: A00(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1
        U[u_base][col] += 1
        V[v_base+1][col] += 1
        V[v_base+p][col] += -1
        W[w_base+1][col] += 1
        col += 1

        # T4: A11(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00=1, C10=1
        U[u_base+m+1][col] += 1
        V[v_base+p][col] += 1
        V[v_base][col] += -1
        W[w_base][col] += 1
        W[w_base+n][col] += 1
        col += 1

        # T5: (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=-1, C01=1
        U[u_base][col] += 1
        U[u_base+1][col] += 1
        V[v_base+p][col] += 1
        W[w_base][col] += -1
        W[w_base+1][col] += 1
        col += 1

        # T6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        U[u_base+m][col] += 1
        U[u_base][col] += -1
        V[v_base][col] += 1
        V[v_base+1][col] += 1
        W[w_base+n+1][col] += 1
        col += 1

        # T7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1, C11=1
        U[u_base+1][col] += 1
        U[u_base+m+1][col] += -1
        V[v_base+p][col] += 1
        V[v_base+p+1][col] += 1
        W[w_base][col] += 1
        W[w_base+n+1][col] += 1
        col += 1

    def add_1x2x2(u_base, v_base, w_base):
        # A is 1x2, B is 2x2, C is 1x2
        # A: u_base, u_base+1
        # B: v_base, v_base+1, v_base+p, v_base+p+1
        # C: w_base, w_base+1
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        add_term(u_base, v_base, w_base, 1)
        add_term(u_base+1, v_base+p, w_base, 1)
        add_term(u_base, v_base+1, w_base+1, 1)
        add_term(u_base+1, v_base+p+1, w_base+1, 1)

    def add_1x4x4(u_base, v_base, w_base):
        # A is 1x4, B is 4x4, C is 1x4
        # Split A into A1 (1x2), A2 (1x2)
        # Split B into B1 (2x2), B2 (2x2)
        # C = A1 B1 + A2 B2
        # A1: u_base, u_base+1
        # A2: u_base+2, u_base+3
        # B1: v_base ... v_base+p+1
        # B2: v_base+p ... v_base+2p+1
        # C1: w_base, w_base+1
        # C2: w_base+2, w_base+3
        
        add_1x2x2(u_base, v_base, w_base)
        add_1x2x2(u_base+2, v_base+p, w_base+2)

    def add_2x4x4(u_base, v_base, w_base):
        # A is 2x4, B is 4x4, C is 2x4
        # Split A into A1 (2x2), A2 (2x2)
        # Split B into B1 (2x2), B2 (2x2)
        # C = A1 B1 + A2 B2
        # A1: u_base ... u_base+m+1
        # A2: u_base+m ... u_base+2m+1
        # B1: v_base ... v_base+p+1
        # B2: v_base+p ... v_base+2p+1
        # C1: w_base ... w_base+n+1
        # C2: w_base+n ... w_base+2n+1
        
        add_strassen_222(u_base, v_base, w_base)
        add_strassen_222(u_base+m, v_base+p, w_base+n)

    # Total rank 44
    # Left half (k=0..3)
    # A is 3x4.
    # A1 (1x4) at u_base=0
    # A2 (2x4) at u_base=m (4)
    # B_L (4x4) at v_base=0
    # C_L (3x4) at w_base=0
    
    # 1x4x4 part
    add_1x4x4(0, 0, 0) # Rank 8
    
    # 2x4x4 part
    add_2x4x4(m, 0, n) # Rank 14. A starts at row 1. C starts at row 1.
    
    # Right half (k=4..7)
    # B_R (4x4) at v_base=p (4)
    # C_R (3x4) at w_base=p (4)
    
    # 1x4x4 part
    add_1x4x4(0, p, p) # Rank 8
    
    # 2x4x4 part
    add_2x4x4(m, p, p+n) # Rank 14

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
