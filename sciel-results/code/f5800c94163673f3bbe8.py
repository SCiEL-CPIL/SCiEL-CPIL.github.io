import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 71 decomposition
    # We will build U, V, W with a dynamic list of columns
    # to easily manage the rank.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # u_idx: i*m + j
        # v_idx: j*p + k
        # w_idx: k*n + i
        U_cols.append([0] * 0) # placeholder, we build columns
        # Actually, it's easier to store terms as (u, v, w, coeff) and then assemble
        pass

    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))

    # Partition indices
    # A: i in 0..2, j in 0..4
    # B: j in 0..4, k in 0..4
    # C: i in 0..2, k in 0..4
    
    # Block 1: A_1 (j=0,1) * B_11 (j=0,1; k=0,1)
    # A_1 rows: 0,1,2. Cols 0,1.
    # B_11 rows: 0,1. Cols 0,1.
    # Output C rows 0,1,2. Cols 0,1.
    
    # Strassen for A_1[0:2, 0:2] * B_11[0:2, 0:2] -> C[0:2, 0:2]
    # A_1[0:2, 0:2] is rows 0,1 of A, cols 0,1.
    # Indices:
    # a = A[0,0] -> u=0*5+0=0
    # b = A[0,1] -> u=0*5+1=1
    # c = A[1,0] -> u=1*5+0=5
    # d = A[1,1] -> u=1*5+1=6
    
    # e = B[0,0] -> v=0*5+0=0
    # f = B[0,1] -> v=0*5+1=1
    # g = B[1,0] -> v=1*5+0=5
    # h = B[1,1] -> v=1*5+1=6
    
    # W indices for C[0:2, 0:2]:
    # C[0,0] -> w=0*3+0=0
    # C[0,1] -> w=1*3+0=3
    # C[1,0] -> w=0*3+1=1
    # C[1,1] -> w=1*3+1=4
    
    def strassen_2x2(a_u, b_u, c_u, d_u, e_v, f_v, g_v, h_v, c00_w, c01_w, c10_w, c11_w):
        # M1 = (a+d)(e+h)
        # U: a, d. V: e, h. W: all 4? No, M1 contributes to all 4.
        # We need to add terms for each M_k.
        
        # M1
        # (a+d)(e+h) = ae + ah + de + dh
        # This is not a single rank 1 term.
        # Strassen computes 7 intermediate products.
        # Each product is a scalar.
        # But we need to express the scalar product as a rank 1 tensor term?
        # No, the tensor decomposition is a sum of rank 1 tensors.
        # Strassen's algorithm computes the product using 7 multiplications of scalars.
        # Each scalar multiplication $x \cdot y$ corresponds to a term in the tensor?
        # No. The tensor is $T = \sum C_{ij} A_{ik} B_{kj}$.
        # Strassen rewrites this as $\sum_{l=1}^7 M_l \cdot (\text{linear combo of A}) \cdot (\text{linear combo of B})$.
        # Wait, the standard Strassen decomposition is:
        # $C_{ij} = \sum_{l=1}^7 S_{ij,l} M_l$ where $M_l = (\sum S_{l,i} A_{i}) (\sum S_{l,j} B_{j})$?
        # No.
        # $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$
        # This $M_1$ is a scalar.
        # In the tensor decomposition, we need to express $C$ as $\sum_r U_r \otimes V_r \otimes W_r$.
        # Strassen's 7 steps correspond to 7 rank-1 terms?
        # Let's check.
        # $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
        # This is NOT a rank 1 term in the tensor $A \otimes B \to C$.
        # A rank 1 term is $u \otimes v \otimes w$ where $u, v, w$ are vectors.
        # The term $(A_{11}+A_{22})(B_{11}+B_{22})$ is a product of two linear forms.
        # It contributes to $C_{11}, C_{12}, C_{21}, C_{22}$.
        # Specifically, $C_{11} += M_1$, $C_{12} += M_1$, etc? No.
        # $C_{11} = M_1 + M_4 - M_5 + M_7$.
        # So $M_1$ appears in $C_{11}$ with coeff 1, $C_{12}$ with coeff 1? No.
        # Let's look at the formula again.
        # $C_{11} = M_1 + M_4 - M_5 + M_7$
        # $C_{12} = M_3 + M_5$
        # $C_{21} = M_2 + M_4$
        # $C_{22} = M_1 - M_2 + M_3 + M_6$
        
        # Each $M_l$ is a scalar product of two linear combinations of entries of A and B.
        # $M_l = (\sum_{i,j} \alpha_{ij} A_{ij}) (\sum_{k,l} \beta_{kl} B_{kl})$.
        # This can be written as a sum of rank 1 tensors?
        # $(\sum \alpha A) (\sum \beta B) = \sum_{i,j,k,l} \alpha_{ij} \beta_{kl} A_{ij} B_{kl}$.
        # This is a sum of rank 1 terms $A_{ij} \otimes B_{kl} \otimes (\text{coeff for } C)$.
        # But Strassen reduces the number of *multiplications*, not necessarily the rank of the tensor?
        # Actually, the rank of the tensor is the minimum number of rank 1 terms.
        # Strassen's algorithm shows that the rank is at most 7 for $2 \times 2 \times 2$.
        # How?
        # It expresses the tensor as $\sum_{l=1}^7 T_l$ where $T_l$ is a rank 1 tensor?
        # No. $M_l$ is a scalar. The contribution to $C_{ij}$ is $c_{ij,l} M_l$.
        # So $C_{ij} = \sum_l c_{ij,l} (\sum \alpha_{ij} A) (\sum \beta_{kl} B)$.
        # This is NOT a sum of 7 rank 1 tensors.
        # A rank 1 tensor is $u \otimes v \otimes w$.
        # The term $(\sum \alpha A) (\sum \beta B) \cdot w$ is a rank 1 tensor IF the linear forms are rank 1?
        # No. $(\sum \alpha A)$ is a vector in the dual space?
        # Let's be precise.
        # The tensor $T$ maps $A, B$ to $C$.
        # $T = \sum_{r} u_r \otimes v_r \otimes w_r$.
        # $C = \sum_r (u_r \cdot A) (v_r \cdot B) w_r$.
        # Strassen's $M_l = (\alpha_l \cdot A) (\beta_l \cdot B)$.
        # Then $C_{ij} = \sum_l c_{ij,l} M_l$.
        # This means $C = \sum_l ( (\sum_{ij} c_{ij,l} w_{ij}) \otimes \alpha_l \otimes \beta_l )$?
        # Yes!
        # For each $l$, we have a term $u_l \otimes v_l \otimes w_l$ where:
        # $u_l = \alpha_l$ (vector of coefficients for A)
        # $v_l = \beta_l$ (vector of coefficients for B)
        # $w_l$ is the vector of coefficients for C, where $w_{ij} = c_{ij,l}$.
        # So each Strassen step $M_l$ corresponds to ONE rank 1 tensor term!
        
        # So we just need to define $\alpha_l, \beta_l, w_l$ for each of the 7 steps.
        
        # M1: (A11+A22)(B11+B22)
        # alpha: A11=1, A22=1.
        # beta: B11=1, B22=1.
        # w: C11=1, C12=1? No.
        # C11 += 1*M1. C12 += 0*M1?
        # Formula:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1:
        # w11 = 1, w12 = 0, w21 = 0, w22 = 1.
        
        # M2: (A21+A22)B11
        # alpha: A21=1, A22=1.
        # beta: B11=1.
        # w: C21 += 1*M2. w21=1. Others 0.
        
        # M3: A11(B12-B22)
        # alpha: A11=1.
        # beta: B12=1, B22=-1.
        # w: C12 += 1*M3. w12=1.
        
        # M4: A22(B21-B11)
        # alpha: A22=1.
        # beta: B21=1, B11=-1.
        # w: C11 += 1*M4, C21 += 1*M4. w11=1, w21=1.
        
        # M5: (A11+A12)B22
        # alpha: A11=1, A12=1.
        # beta: B22=1.
        # w: C11 += -1*M5, C12 += 1*M5. w11=-1, w12=1.
        
        # M6: (A21-A11)(B11+B12)
        # alpha: A21=1, A11=-1.
        # beta: B11=1, B12=1.
        # w: C22 += 1*M6. w22=1.
        
        # M7: (A12-A22)(B21+B22)
        # alpha: A12=1, A22=-1.
        # beta: B21=1, B22=1.
        # w: C11 += 1*M7. w11=1.
        
        # We need to map these local indices to global U, V, W.
        pass

    # This is getting complex to hardcode.
    # I will write a general function to add a rank 1 term defined by sparse vectors.
    
    def add_rank1_term(u_sparse, v_sparse, w_sparse):
        # u_sparse: dict {global_u_idx: coeff}
        # v_sparse: dict {global_v_idx: coeff}
        # w_sparse: dict {global_w_idx: coeff}
        for u_idx, u_coeff in u_sparse.items():
            for v_idx, v_coeff in v_sparse.items():
                for w_idx, w_coeff in w_sparse.items():
                    # The term is u_coeff * v_coeff * w_coeff * (e_u \otimes e_v \otimes e_w)
                    # We add this to the matrix.
                    terms.append((u_idx, v_idx, w_idx, u_coeff * v_coeff * w_coeff))

    # Helper to get global indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen for 2x2x2
    # Local A indices: (0,0), (0,1), (1,0), (1,1)
    # Local B indices: (0,0), (0,1), (1,0), (1,1)
    # Local C indices: (0,0), (0,1), (1,0), (1,1)
    
    def add_strassen_block(a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset):
        # a_i_offset: row offset for A
        # a_j_offset: col offset for A
        # b_j_offset: row offset for B
        # b_k_offset: col offset for B
        # c_i_offset: row offset for C
        # c_k_offset: col offset for C
        
        # M1
        u = {u_idx(a_i_offset+0, a_j_offset+0): 1, u_idx(a_i_offset+1, a_j_offset+1): 1}
        v = {v_idx(b_j_offset+0, b_k_offset+0): 1, v_idx(b_j_offset+1, b_k_offset+1): 1}
        w = {w_idx(c_k_offset+0, c_i_offset+0): 1, w_idx(c_k_offset+1, c_i_offset+1): 1}
        add_rank1_term(u, v, w)
        
        # M2
        u = {u_idx(a_i_offset+1, a_j_offset+0): 1, u_idx(a_i_offset+1, a_j_offset+1): 1}
        v = {v_idx(b_j_offset+0, b_k_offset+0): 1}
        w = {w_idx(c_k_offset+0, c_i_offset+1): 1}
        add_rank1_term(u, v, w)
        
        # M3
        u = {u_idx(a_i_offset+0, a_j_offset+0): 1}
        v = {v_idx(b_j_offset+0, b_k_offset+1): 1, v_idx(b_j_offset+1, b_k_offset+1): -1}
        w = {w_idx(c_k_offset+1, c_i_offset+0): 1}
        add_rank1_term(u, v, w)
        
        # M4
        u = {u_idx(a_i_offset+1, a_j_offset+1): 1}
        v = {v_idx(b_j_offset+1, b_k_offset+0): 1, v_idx(b_j_offset+0, b_k_offset+0): -1}
        w = {w_idx(c_k_offset+0, c_i_offset+0): 1, w_idx(c_k_offset+0, c_i_offset+1): 1}
        add_rank1_term(u, v, w)
        
        # M5
        u = {u_idx(a_i_offset+0, a_j_offset+0): 1, u_idx(a_i_offset+0, a_j_offset+1): 1}
        v = {v_idx(b_j_offset+1, b_k_offset+1): 1}
        w = {w_idx(c_k_offset+0, c_i_offset+0): -1, w_idx(c_k_offset+1, c_i_offset+0): 1}
        add_rank1_term(u, v, w)
        
        # M6
        u = {u_idx(a_i_offset+1, a_j_offset+0): 1, u_idx(a_i_offset+0, a_j_offset+0): -1}
        v = {v_idx(b_j_offset+0, b_k_offset+0): 1, v_idx(b_j_offset+0, b_k_offset+1): 1}
        w = {w_idx(c_k_offset+1, c_i_offset+1): 1}
        add_rank1_term(u, v, w)
        
        # M7
        u = {u_idx(a_i_offset+0, a_j_offset+1): 1, u_idx(a_i_offset+1, a_j_offset+1): -1}
        v = {v_idx(b_j_offset+1, b_k_offset+0): 1, v_idx(b_j_offset+1, b_k_offset+1): 1}
        w = {w_idx(c_k_offset+0, c_i_offset+0): 1}
        add_rank1_term(u, v, w)

    def add_schoolbook_block(a_i_start, a_i_end, a_j_start, a_j_end, b_j_start, b_j_end, b_k_start, b_k_end, c_i_start, c_i_end, c_k_start, c_k_end):
        for i in range(a_i_start, a_i_end):
            for j in range(a_j_start, a_j_end):
                for k in range(b_k_start, b_k_end):
                    u = {u_idx(i, j): 1}
                    v = {v_idx(j, k): 1}
                    w = {w_idx(k, i): 1}
                    add_rank1_term(u, v, w)

    # Block 1: A_1 B_11
    # A_1: i=0..2, j=0..1
    # B_11: j=0..1, k=0..1
    # C: i=0..2, k=0..1
    # Strassen for i=0,1; j=0,1; k=0,1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # Schoolbook for i=2; j=0,1; k=0,1
    add_schoolbook_block(2, 3, 0, 2, 0, 2, 0, 2, 2, 3, 0, 2)

    # Block 2: A_1 B_12
    # A_1: i=0..2, j=0..1
    # B_12: j=0..1, k=2..3
    # C: i=0..2, k=2..3
    add_strassen_block(0, 0, 0, 2, 0, 2)
    add_schoolbook_block(2, 3, 0, 2, 0, 2, 2, 4, 2, 3, 2, 4)

    # Block 3: A_1 B_13
    # A_1: i=0..2, j=0..1
    # B_13: j=0..1, k=4
    # C: i=0..2, k=4
    add_schoolbook_block(0, 3, 0, 2, 0, 2, 4, 5, 0, 3, 4, 5)

    # Block 4: A_2 B_21
    # A_2: i=0..2, j=2..3
    # B_21: j=2..3, k=0..1
    # C: i=0..2, k=0..1
    add_strassen_block(0, 2, 2, 0, 0, 0)
    add_schoolbook_block(2, 3, 2, 4, 2, 4, 0, 2, 2, 3, 0, 2)

    # Block 5: A_2 B_22
    # A_2: i=0..2, j=2..3
    # B_22: j=2..3, k=2..3
    # C: i=0..2, k=2..3
    add_strassen_block(0, 2, 2, 2, 0, 2)
    add_schoolbook_block(2, 3, 2, 4, 2, 4, 2, 4, 2, 3, 2, 4)

    # Block 6: A_2 B_23
    # A_2: i=0..2, j=2..3
    # B_23: j=2..3, k=4
    # C: i=0..2, k=4
    add_schoolbook_block(0, 3, 2, 4, 2, 4, 4, 5, 0, 3, 4, 5)

    # Block 7: A_3 B_31
    # A_3: i=0..2, j=4
    # B_31: j=4, k=0..1
    # C: i=0..2, k=0..1
    add_schoolbook_block(0, 3, 4, 5, 4, 5, 0, 2, 0, 3, 0, 2)

    # Block 8: A_3 B_32
    # A_3: i=0..2, j=4
    # B_32: j=4, k=2..3
    # C: i=0..2, k=2..3
    add_schoolbook_block(0, 3, 4, 5, 4, 5, 2, 4, 0, 3, 2, 4)

    # Block 9: A_3 B_33
    # A_3: i=0..2, j=4
    # B_33: j=4, k=4
    # C: i=0..2, k=4
    add_schoolbook_block(0, 3, 4, 5, 4, 5, 4, 5, 0, 3, 4, 5)

    # Assemble U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx][r] = coeff
        V[v_idx][r] = coeff
        W[w_idx][r] = coeff
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
