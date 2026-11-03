"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_val, v_val, w_val, u_idx, v_idx, w_idx):
        # u_idx: index in U (0..11)
        # v_idx: index in V (0..27)
        # w_idx: index in W (0..20)
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        u_col[u_idx] = u_val
        v_col[v_idx] = v_val
        w_col[w_idx] = w_val
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def add_strassen_2x2x2(i_start, j_start, k_start):
        # A indices: i in {i_start, i_start+1}, j in {j_start, j_start+1}
        # B indices: j in {j_start, j_start+1}, k in {k_start, k_start+1}
        # C indices: i in {i_start, i_start+1}, k in {k_start, k_start+1}
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00: i_start, j_start
        # A11: i_start+1, j_start+1
        # B00: j_start, k_start
        # B11: j_start+1, k_start+1
        
        # We need to express linear combinations in U, V, W.
        # This is not a single scalar multiplication, but a product of sums.
        # The tensor decomposition requires that each term is a product of one element from A, one from B, contributing to one element of C.
        # Strassen works by changing the basis.
        # The "multiplications" are:
        # M1 = (A00 + A11) (B00 + B11)
        # This is not a single tensor term. It corresponds to 4 terms if expanded, but Strassen combines them.
        # However, the tensor rank definition allows linear combinations in the factors U, V, W.
        # So, for the term corresponding to M1:
        # U vector has 1 at A00 and 1 at A11.
        # V vector has 1 at B00 and 1 at B11.
        # W vector has 1 at C00 and 1 at C11? No.
        
        # Let's recall how Strassen maps to tensor factors.
        # The product $C = AB$ is linear in A and B.
        # $C_{00} = A_{00}B_{00} + A_{01}B_{10} + A_{10}B_{01} + A_{11}B_{11}$.
        # Strassen defines $M_1, ..., M_7$.
        # $M_1 = (A_{00} + A_{11})(B_{00} + B_{11}) = A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # This $M_1$ contributes to multiple $C_{ij}$.
        # Specifically:
        # $C_{00} = M_1 + M_4 - M_5 + M_7$
        # $C_{01} = M_3 + M_5$
        # $C_{10} = M_2 + M_4$
        # $C_{11} = M_1 - M_2 + M_3 + M_6$
        
        # So, for each $M_r$, we create a column in U, V, W.
        # For $M_1$:
        # U has 1 at $A_{00}$ and 1 at $A_{11}$.
        # V has 1 at $B_{00}$ and 1 at $B_{11}$.
        # W has coefficients for $C_{00}, C_{01}, C_{10}, C_{11}$ based on the formula.
        # $W_{C_{00}} = 1$
        # $W_{C_{01}} = 0$
        # $W_{C_{10}} = 0$
        # $W_{C_{11}} = 1$
        
        # Let's define the 7 Strassen terms.
        
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        add_strassen_term(
            { (i_start, j_start): 1, (i_start+1, j_start+1): 1 },
            { (j_start, k_start): 1, (j_start+1, k_start+1): 1 },
            { (i_start, k_start): 1, (i_start+1, k_start+1): 1 }
        )
        
        # M2: (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1
        add_strassen_term(
            { (i_start+1, j_start): 1, (i_start+1, j_start+1): 1 },
            { (j_start, k_start): 1 },
            { (i_start+1, k_start): 1 }
        )
        
        # M3: A00(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1
        add_strassen_term(
            { (i_start, j_start): 1 },
            { (j_start, k_start+1): 1, (j_start+1, k_start+1): -1 },
            { (i_start, k_start+1): 1 }
        )
        
        # M4: A11(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00=1, C10=1
        add_strassen_term(
            { (i_start+1, j_start+1): 1 },
            { (j_start+1, k_start): 1, (j_start, k_start): -1 },
            { (i_start, k_start): 1, (i_start+1, k_start): 1 }
        )
        
        # M5: (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C01=1, C00=-1
        add_strassen_term(
            { (i_start, j_start): 1, (i_start, j_start+1): 1 },
            { (j_start+1, k_start+1): 1 },
            { (i_start, k_start+1): 1, (i_start, k_start): -1 }
        )
        
        # M6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        add_strassen_term(
            { (i_start+1, j_start): 1, (i_start, j_start): -1 },
            { (j_start, k_start): 1, (j_start, k_start+1): 1 },
            { (i_start+1, k_start+1): 1 }
        )
        
        # M7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1, C11=-1
        add_strassen_term(
            { (i_start, j_start+1): 1, (i_start+1, j_start+1): -1 },
            { (j_start+1, k_start): 1, (j_start+1, k_start+1): 1 },
            { (i_start, k_start): 1, (i_start+1, k_start+1): -1 }
        )

    def add_strassen_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), val in u_dict.items():
            u_col[i * m + j] = val
            
        for (j, k), val in v_dict.items():
            v_col[j * p + k] = val
            
        for (i, k), val in w_dict.items():
            w_col[k * n + i] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def add_schoolbook_1x2x7(i_fixed, j_start, k_start):
        # i is fixed
        # j in {j_start, j_start+1}
        # k in {k_start .. k_start+6}
        for j in range(2):
            for k in range(7):
                u_idx = i_fixed * m + (j_start + j)
                v_idx = (j_start + j) * p + (k_start + k)
                w_idx = (k_start + k) * n + i_fixed
                
                add_term(1, 1, 1, u_idx, v_idx, w_idx)

    def add_schoolbook_2x2x1(i_start, j_start, k_fixed):
        # i in {i_start, i_start+1}
        # j in {j_start, j_start+1}
        # k is fixed
        for i in range(2):
            for j in range(2):
                u_idx = (i_start + i) * m + (j_start + j)
                v_idx = (j_start + j) * p + k_fixed
                w_idx = k_fixed * n + (i_start + i)
                
                add_term(1, 1, 1, u_idx, v_idx, w_idx)

    # Block 1: j in {0, 1}
    # Part 1a: i in {0, 1}, j in {0, 1}, k in {0..6}
    # Split k into {0,1}, {2,3}, {4,5}, {6}
    add_strassen_2x2x2(0, 0, 0) # i=0,1; j=0,1; k=0,1
    add_strassen_2x2x2(0, 0, 2) # i=0,1; j=0,1; k=2,3
    add_strassen_2x2x2(0, 0, 4) # i=0,1; j=0,1; k=4,5
    add_schoolbook_2x2x1(0, 0, 6) # i=0,1; j=0,1; k=6
    
    # Part 1b: i=2, j in {0, 1}, k in {0..6}
    add_schoolbook_1x2x7(2, 0, 0)
    
    # Block 2: j in {2, 3}
    # Part 2a: i in {0, 1}, j in {2, 3}, k in {0..6}
    add_strassen_2x2x2(0, 2, 0) # i=0,1; j=2,3; k=0,1
    add_strassen_2x2x2(0, 2, 2) # i=0,1; j=2,3; k=2,3
    add_strassen_2x2x2(0, 2, 4) # i=0,1; j=2,3; k=4,5
    add_schoolbook_2x2x1(0, 2, 6) # i=0,1; j=2,3; k=6
    
    # Part 2b: i=2, j in {2, 3}, k in {0..6}
    add_schoolbook_1x2x7(2, 2, 0)

    rank = len(U_cols)
    U = U_cols
    V = V_cols
    W = W_cols
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
