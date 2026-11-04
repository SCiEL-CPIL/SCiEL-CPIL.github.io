import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen constants for 2x2x2
    # M1 = (A00+A11)(B10+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = -M1 + M3 + M5 + M7
    # C10 = M2 - M4 + M6
    # C11 = -M2 + M1 + M4 + M6
    
    # Block 1: k in {0, 1}
    # j in {0, 1}, i in {0, 1}
    # u_idx = i*3 + j
    # v_idx = j*7 + k
    # w_idx = k*2 + i
    
    def add_strassen_block(k_start):
        nonlocal col
        # M1
        add_term(0, 7*0 + k_start+1, (k_start+1)*2 + 0, 1, 1, 1) # A00 B11 C01? No.
        # Let's map carefully.
        # M1 = (A00+A11)(B10+B11)
        # Terms in U: A00 (i=0,j=0), A11 (i=1,j=1)
        # Terms in V: B10 (j=1, k=k_start), B11 (j=1, k=k_start+1)
        # Wait, B indices are j, k.
        # B10 means j=1, k=k_start.
        # B11 means j=1, k=k_start+1.
        # This is getting messy to hardcode. I will generate the 7 terms explicitly.
        
        # Reset col logic? No, add_term increments col.
        # I need to define the 7 multiplications.
        
        # M1: (A00+A11) * (B10+B11)
        # U: A00 (0,0), A11 (1,1)
        # V: B10 (1, k_s), B11 (1, k_s+1)
        # This is not a single term in U/V/W. It's a sum.
        # The decomposition is a sum of rank-1 tensors.
        # Strassen's algorithm expresses the 8 products as 7.
        # The 7 "multiplications" are:
        # m1 = (A00+A11) * (B10+B11)
        # m2 = (A10+A11) * B00
        # m3 = A00 * (B10-B11)
        # m4 = A11 * (B01-B00)
        # m5 = (A00+A01) * B11
        # m6 = (A10-A00) * (B00+B10)
        # m7 = (A01-A11) * (B11+B01)
        
        # Each mi is a product of a linear combination of A's and a linear combination of B's.
        # So each mi corresponds to ONE rank-1 tensor in the decomposition?
        # No. A rank-1 tensor is $u \otimes v \otimes w$.
        # $u$ is a vector in A-space, $v$ in B-space, $w$ in C-space.
        # If $m1 = (A00+A11)(B10+B11)$, this contributes to multiple $C$ entries.
        # Specifically, $m1$ is used in $C00, C01, C11$.
        # So $m1$ is NOT a rank-1 tensor. It is a "multiplication" that feeds into multiple outputs.
        # The tensor decomposition is $T = \sum_{c=1}^r u_c \otimes v_c \otimes w_c$.
        # In Strassen, we define 7 intermediate variables $m_1..m_7$.
        # Each $m_k$ is a product of a linear form in A and a linear form in B.
        # $m_k = L_A^{(k)}(A) L_B^{(k)}(B)$.
        # Then $C$ is a linear combination of $m_k$'s.
        # $C_{i,j} = \sum_k w_{i,j}^{(k)} m_k$.
        # This means the tensor is $\sum_k (L_A^{(k)} \otimes L_B^{(k)} \otimes w^{(k)})$.
        # Where $w^{(k)}$ is the vector of coefficients for $C$.
        # So each $m_k$ corresponds to ONE rank-1 term in the decomposition!
        # The vector $u_k$ has the coefficients of $L_A^{(k)}$.
        # The vector $v_k$ has the coefficients of $L_B^{(k)}$.
        # The vector $w_k$ has the coefficients of $C$ in terms of $m_k$.
        
        # So for each Strassen block, we add 7 terms.
        
        # Let's define the 7 terms for a block starting at k_start.
        # A indices: i=0,1; j=0,1.
        # B indices: j=0,1; k=k_start, k_start+1.
        # C indices: i=0,1; k=k_start, k_start+1.
        
        # Term 1: m1 = (A00+A11)(B10+B11)
        # u: A00=1, A11=1
        # v: B10=1, B11=1 (Note: B10 is j=1, k=k_s; B11 is j=1, k=k_s+1)
        # w: C00=1, C01=-1, C10=0, C11=1 (from C00=M1..., C01=-M1..., C11=M1...)
        # Wait, C10 does not use M1.
        
        # Let's write the code to add these 7 terms.
        
        # 1. m1
        add_term(0*3+0, 1*7+k_start, (k_start)*2+0, 1, 1, 1) # A00 B10 C00
        # This is wrong. add_term adds a SINGLE rank 1 tensor.
        # The rank 1 tensor for m1 is:
        # u vector: [1, 0, 0, 0, 1, 0] (A00, A01, A10, A11...)
        # v vector: [0, 0, 0, 0, 1, 1, 0...] (B10, B11)
        # w vector: [1, -1, 0, 1, ...] (C00, C01, C10, C11)
        
        # My add_term function assumes a single non-zero entry in U, V, W.
        # I need a function to add a general rank-1 tensor with sparse vectors.
        
    # Redefine add_term to accept indices and values for the sparse vectors.
    
    def add_rank1_term(u_sparsity, v_sparsity, w_sparsity):
        nonlocal col
        for idx, val in u_sparsity.items():
            U[idx][col] = val
        for idx, val in v_sparsity.items():
            V[idx][col] = val
        for idx, val in w_sparsity.items():
            W[idx][col] = val
        col += 1

    # Strassen Block Helper
    def add_strassen_block(k_s):
        # A indices: i=0,1; j=0,1. u_idx = i*3+j
        # B indices: j=0,1; k=k_s, k_s+1. v_idx = j*7+k
        # C indices: i=0,1; k=k_s, k_s+1. w_idx = k*2+i
        
        # m1 = (A00+A11)(B10+B11)
        # u: {0:1, 4:1} (A00=0*3+0, A11=1*3+1)
        # v: {1*7+k_s:1, 1*7+k_s+1:1} (B10, B11)
        # w: C00=1, C01=-1, C11=1. (C00=k_s*2+0, C01=k_s+1*2+0, C11=k_s+1*2+1)
        add_rank1_term({0:1, 4:1}, {7+k_s:1, 7+k_s+1:1}, {k_s*2:1, (k_s+1)*2:-1, (k_s+1)*2+1:1})
        
        # m2 = (A10+A11)B00
        # u: {3:1, 4:1} (A10=1*3+0, A11=1*3+1)
        # v: {0*7+k_s:1} (B00)
        # w: C10=1, C11=-1.
        add_rank1_term({3:1, 4:1}, {k_s:1}, {k_s*2+1:1, (k_s+1)*2+1:-1})
        
        # m3 = A00(B10-B11)
        # u: {0:1}
        # v: {7+k_s:1, 7+k_s+1:-1}
        # w: C01=1.
        add_rank1_term({0:1}, {7+k_s:1, 7+k_s+1:-1}, {(k_s+1)*2:1})
        
        # m4 = A11(B01-B00)
        # u: {4:1}
        # v: {0*7+k_s+1:1, 0*7+k_s:-1}
        # w: C00=1, C10=-1, C11=1.
        add_rank1_term({4:1}, {k_s+1:1, k_s:-1}, {k_s*2:1, k_s*2+1:-1, (k_s+1)*2+1:1})
        
        # m5 = (A00+A01)B11
        # u: {0:1, 1:1}
        # v: {7+k_s+1:1}
        # w: C00=-1, C01=1.
        add_rank1_term({0:1, 1:1}, {7+k_s+1:1}, {k_s*2:-1, (k_s+1)*2:1})
        
        # m6 = (A10-A00)(B00+B10)
        # u: {3:1, 0:-1}
        # v: {k_s:1, 7+k_s:1}
        # w: C10=1.
        add_rank1_term({3:1, 0:-1}, {k_s:1, 7+k_s:1}, {k_s*2+1:1})
        
        # m7 = (A01-A11)(B11+B01)
        # u: {1:1, 4:-1}
        # v: {7+k_s+1:1, k_s+1:1}
        # w: C00=1, C01=1.
        add_rank1_term({1:1, 4:-1}, {7+k_s+1:1, k_s+1:1}, {k_s*2:1, (k_s+1)*2:1})

    # Add 3 Strassen blocks
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Schoolbook 2x2x1 block for k=6
    # A_L (2x2) * B_T4 (2x1)
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    # 4 terms.
    # 1. A00 B06 -> C06
    add_rank1_term({0:1}, {0*7+6:1}, {6*2+0:1})
    # 2. A01 B16 -> C06
    add_rank1_term({1:1}, {1*7+6:1}, {6*2+0:1})
    # 3. A10 B06 -> C16
    add_rank1_term({3:1}, {0*7+6:1}, {6*2+1:1})
    # 4. A11 B16 -> C16
    add_rank1_term({4:1}, {1*7+6:1}, {6*2+1:1})
    
    # Schoolbook 2x1x7 block for A_R (j=2) * B_B (j=2)
    # C_i,k += A_i,2 B_2,k
    # 14 terms.
    for i in range(2):
        for k in range(7):
            u_idx = i*3 + 2
            v_idx = 2*7 + k
            w_idx = k*2 + i
            add_rank1_term({u_idx:1}, {v_idx:1}, {w_idx:1})
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
