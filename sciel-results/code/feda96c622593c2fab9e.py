import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit for safety
    
    n, m, p = DIMENSIONS
    
    # Block partitions
    # n=3 -> [2, 1]
    # m=5 -> [2, 2, 1]
    # p=6 -> [2, 2, 2]
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Calculate offsets
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 tensor
    def add_rank1(i, j, k, col):
        U.append([0] * col)
        U[-1][i * m + j] = 1
        V.append([0] * col)
        V[-1][j * p + k] = 1
        W.append([0] * col)
        W[-1][k * n + i] = 1

    # Helper to add Strassen 2x2x2
    # local indices 0,1 for i, j, k
    # global offsets oi, oj, ok
    def add_strassen(oi, oj, ok, col_start):
        # M1 = (A00 + A11)(B00 + B11)
        # U: (0,0)+1, (1,1)+1
        # V: (0,0)+1, (1,1)+1
        # W: (0,0)+1, (1,1)+1
        # But Strassen is a sum of 7 rank-1-like terms?
        # No, Strassen is 7 multiplications. Each multiplication is a rank-1 tensor in the bilinear form sense?
        # Yes, each of the 7 products $M_t$ corresponds to a rank-1 tensor in the decomposition of the 2x2x2 multiplication tensor.
        
        # The 7 terms for Strassen are:
        # t1: (A00+A11)(B00+B11)
        # t2: (A10+A11)B00
        # t3: A00(B01-B11)
        # t4: A11(B10-B00)
        # t5: (A00+A01)B11
        # t6: (A10-A00)(B00+B01)
        # t7: (A01-A11)(B10+B11)
        
        # We need to map these to U, V, W entries.
        # U corresponds to A, V to B, W to C.
        # For a term like (A00+A11)(B00+B11):
        # This is not a single rank-1 tensor $u v w^T$. It is a sum of 4 rank-1 tensors if expanded?
        # No, in tensor decomposition, a "multiplication" $M_t$ is a scalar product of two vectors (from A and B) which is then distributed to C.
        # Actually, the standard Strassen decomposition expresses the tensor as a sum of 7 rank-1 tensors.
        # Let's verify the rank-1 tensors for Strassen.
        
        # Rank 1 tensor: $U_{i,j} V_{j,k} W_{k,i} = 1$ for specific i,j,k?
        # No, the entries of U, V, W are coefficients.
        # For term 1: $(A_{00} + A_{11})(B_{00} + B_{11})$
        # This contributes to $C_{00}, C_{01}, C_{10}, C_{11}$?
        # No, Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each $M_t$ is a product of a linear combination of A's and a linear combination of B's.
        # This corresponds to a rank-1 tensor in the decomposition of the multiplication tensor?
        # Yes. The tensor $T$ is $\sum_{t=1}^7 u_t \otimes v_t \otimes w_t$.
        # $u_t$ is a vector in $\mathbb{R}^4$ (A), $v_t$ in $\mathbb{R}^4$ (B), $w_t$ in $\mathbb{R}^4$ (C).
        
        # Let's define the 7 vectors for A, B, C.
        # Indices for 2x2: 00, 01, 10, 11.
        
        # M1: A: (1, 0, 0, 1), B: (1, 0, 0, 1) -> C: (1, 0, 0, 1) ?
        # Wait, the C part of the rank-1 tensor determines how the product is added to C.
        # If $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$, and $C_{00}$ gets $+M_1$, $C_{11}$ gets $+M_1$.
        # So $w_1$ has 1 at $C_{00}$ and 1 at $C_{11}$.
        # $u_1$ has 1 at $A_{00}, A_{11}$.
        # $v_1$ has 1 at $B_{00}, B_{11}$.
        
        # Let's list them:
        # t1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        # t2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)  # M2=(A10+A11)B00, C10+=M2
        # t3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0)  # M3=A00(B01-B11), C01+=M3
        # t4: u=(0,0,0,1), v=(0,0,1,-1), w=(1,0,1,0)  # M4=A11(B10-B00), C00+=M4, C10+=M4
        # t5: u=(1,1,0,0), v=(0,0,0,1), w=(0,1,1,0)  # M5=(A00+A01)B11, C01+=M5, C10+=M5 ? 
        #    Wait, C01 = M3 + M5. C10 = M2 + M4.
        #    So w5 should have 1 at C01 and 1 at C10? No.
        #    C01 gets M5. C10 does NOT get M5.
        #    Let's re-read Strassen.
        #    C00 = M1 + M4 - M5 + M7
        #    C01 = M3 + M5
        #    C10 = M2 + M4
        #    C11 = M1 - M2 + M3 + M6
        #    
        #    So:
        #    M1: C00, C11. w1=(1,0,0,1)
        #    M2: C10. w2=(0,0,1,0)
        #    M3: C01, C11. w3=(0,1,0,1)
        #    M4: C00, C10. w4=(1,0,1,0)
        #    M5: C01, C00. w5=(1,1,0,0)  # C00 has -M5, C01 has +M5.
        #       So w5=(1,1,0,0) with signs?
        #       The rank-1 tensor decomposition uses fixed signs in U,V,W.
        #       If w5 has 1 at C00 and 1 at C01, then we need to adjust U or V to get the minus sign?
        #       No, the signs are in the U, V, W vectors.
        #       
        #    Let's check t5: M5 = (A00+A01)B11.
        #    u5=(1,1,0,0), v5=(0,0,0,1).
        #    w5: C00 gets -M5, C01 gets +M5.
        #    So w5=(1,1,0,0) is not correct if we want to add -M5 to C00 and +M5 to C01.
        #    We can put the sign in w5: w5=(-1, 1, 0, 0).
        
        #    Let's refine:
        #    t1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        #    t2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
        #    t3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,1)
        #    t4: u=(0,0,0,1), v=(0,0,1,-1), w=(1,0,1,0)
        #    t5: u=(1,1,0,0), v=(0,0,0,1), w=(-1,1,0,0)
        #    t6: u=(0,0,1,-1), v=(1,1,0,0), w=(0,0,0,1)  # M6=(A10-A00)(B00+B01), C11+=M6
        #    t7: u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,0)  # M7=(A01-A11)(B10+B11), C00+=M7
        
        #    Let's verify C11: M1 - M2 + M3 + M6
        #    w1[3]=1, w2[3]=0, w3[3]=1, w6[3]=1.
        #    Sum: 1 - 0 + 1 + 1 = 3? No.
        #    The coefficients in the formula are applied to the scalar products.
        #    The decomposition is $T = \sum u_t \otimes v_t \otimes w_t$.
        #    This means $C_{ij} = \sum_t (u_t \cdot A) (v_t \cdot B) w_t[k,l]$?
        #    No, $w_t$ is a vector. The tensor entry $T_{ijk}$ is $\sum_t U_{t,ij} V_{t,jk} W_{t,ki}$.
        #    Wait, the standard form is $C_{ij} = \sum_t M_t W_{t,ij}$.
        #    Where $M_t = \sum_{ab} U_{t,ab} A_{ab} \sum_{cd} V_{t,cd} B_{cd}$.
        #    
        #    So for t1:
        #    $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        #    $U_{t1} = (1,0,0,1)$, $V_{t1} = (1,0,0,1)$.
        #    $W_{t1}$ must be such that $C_{00}$ gets $+M_1$ and $C_{11}$ gets $+M_1$.
        #    So $W_{t1} = (1,0,0,1)$.
        
        #    For t5:
        #    $M_5 = (A_{00}+A_{01})B_{11}$.
        #    $U_{t5} = (1,1,0,0)$, $V_{t5} = (0,0,0,1)$.
        #    $C_{00}$ gets $-M_5$, $C_{01}$ gets $+M_5$.
        #    So $W_{t5} = (-1, 1, 0, 0)$.
        
        #    For t3:
        #    $M_3 = A_{00}(B_{01}-B_{11})$.
        #    $U_{t3} = (1,0,0,0)$, $V_{t3} = (0,1,0,-1)$.
        #    $C_{01}$ gets $+M_3$, $C_{11}$ gets $+M_3$.
        #    So $W_{t3} = (0,1,0,1)$.
        
        #    For t6:
        #    $M_6 = (A_{10}-A_{00})(B_{00}+B_{01})$.
        #    $U_{t6} = (0,0,1,-1)$? No, A10 is index 2, A00 is index 0.
        #    $U_{t6} = (-1, 0, 1, 0)$.
        #    $V_{t6} = (1,1,0,0)$.
        #    $C_{11}$ gets $+M_6$.
        #    So $W_{t6} = (0,0,0,1)$.
        
        #    For t7:
        #    $M_7 = (A_{01}-A_{11})(B_{10}+B_{11})$.
        #    $U_{t7} = (0,1,0,-1)$.
        #    $V_{t7} = (0,0,1,1)$.
        #    $C_{00}$ gets $+M_7$.
        #    So $W_{t7} = (1,0,0,0)$.
        
        #    Let's check C00: M1 + M4 - M5 + M7
        #    W1[0]=1, W4[0]=1, W5[0]=-1, W7[0]=1. Sum = 2?
        #    Wait, the formula is $C_{00} = M_1 + M_4 - M_5 + M_7$.
        #    My W vectors:
        #    W1=(1,0,0,1) -> contributes 1*M1 to C00.
        #    W4=(1,0,1,0) -> contributes 1*M4 to C00.
        #    W5=(-1,1,0,0) -> contributes -1*M5 to C00.
        #    W7=(1,0,0,0) -> contributes 1*M7 to C00.
        #    Total coeff for M1, M4, M5, M7 is 1, 1, -1, 1. Correct.
        
        #    C01: M3 + M5
        #    W3=(0,1,0,1) -> 1*M3 to C01.
        #    W5=(-1,1,0,0) -> 1*M5 to C01.
        #    Correct.
        
        #    C10: M2 + M4
        #    W2=(0,0,1,0) -> 1*M2 to C10.
        #    W4=(1,0,1,0) -> 1*M4 to C10.
        #    Correct.
        
        #    C11: M1 - M2 + M3 + M6
        #    W1=(1,0,0,1) -> 1*M1 to C11.
        #    W2=(0,0,1,0) -> 0*M2 to C11. (Wait, M2 is not in C11 formula? Correct.)
        #    W3=(0,1,0,1) -> 1*M3 to C11.
        #    W6=(0,0,0,1) -> 1*M6 to C11.
        #    But formula has -M2? No, C11 = M1 - M2 + M3 + M6.
        #    Does W2 contribute to C11? W2[3]=0. So M2 is not added.
        #    But the formula says -M2.
        #    Let's re-check Strassen C11.
        #    Standard Strassen:
        #    P1 = (A11+A22)(B11+B22)
        #    P2 = (A21+A22)B11
        #    P3 = A11(B12-B22)
        #    P4 = A22(B21-B11)
        #    P5 = (A11+A12)B22
        #    P6 = (A21-A11)(B11+B12)
        #    P7 = (A12-A22)(B21+B22)
        #    C11 = P1 + P4 - P5 + P7
        #    C12 = P3 + P5
        #    C21 = P2 + P4
        #    C22 = P1 - P2 + P3 + P6
        #    
        #    My indices: 00, 01, 10, 11 correspond to 11, 12, 21, 22.
        #    C22 (11) = P1 - P2 + P3 + P6.
        #    My W vectors for C11 (index 3):
        #    W1[3]=1 (P1)
        #    W2[3]=0 (P2) -> Should be -1?
        #    W3[3]=1 (P3)
        #    W6[3]=1 (P6)
        #    So I need W2[3] = -1.
        #    Let's update W2.
        #    P2 contributes to C21 (index 2) and C22 (index 3).
        #    C21 = P2 + P4. So W2[2] should be 1.
        #    C22 = P1 - P2 + P3 + P6. So W2[3] should be -1.
        #    So W2 = (0,0,1,-1).
        
        #    Let's re-verify all W vectors.
        #    t1: W1=(1,0,0,1)
        #    t2: W2=(0,0,1,-1)
        #    t3: W3=(0,1,0,1)
        #    t4: W4=(1,0,1,0)
        #    t5: W5=(-1,1,0,0)
        #    t6: W6=(0,0,0,1)
        #    t7: W7=(1,0,0,0)
        
        #    Check C11 (idx 3): W1[3]+W2[3]+W3[3]+W6[3] = 1 - 1 + 1 + 1 = 2?
        #    Formula: P1 - P2 + P3 + P6. Coeffs: 1, -1, 1, 1.
        #    Sum: 1 - 1 + 1 + 1 = 2.
        #    Wait, the tensor decomposition sums $U \cdot A \cdot V \cdot B \cdot W$.
        #    The coefficient of $M_t$ in $C_{ij}$ is $W_t[i,j]$.
        #    So for C22, we want $1 \cdot M_1 - 1 \cdot M_2 + 1 \cdot M_3 + 1 \cdot M_6$.
        #    So $W_1[3]=1, W_2[3]=-1, W_3[3]=1, W_6[3]=1$.
        #    This matches.
        
        #    Check C12 (idx 1): P3 + P5.
        #    W3[1]=1, W5[1]=1. Correct.
        
        #    Check C21 (idx 2): P2 + P4.
        #    W2[2]=1, W4[2]=1. Correct.
        
        #    Check C11 (idx 0): P1 + P4 - P5 + P7.
        #    W1[0]=1, W4[0]=1, W5[0]=-1, W7[0]=1. Correct.
        
        #    So the vectors are:
        #    t1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1)
        #    t2: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,-1)
        #    t3: U=(1,0,0,0), V=(0,1,0,-1), W=(0,1,0,1)
        #    t4: U=(0,0,0,1), V=(0,0,1,-1), W=(1,0,1,0)
        #    t5: U=(1,1,0,0), V=(0,0,0,1), W=(-1,1,0,0)
        #    t6: U=(-1,0,1,0), V=(1,1,0,0), W=(0,0,0,1)
        #    t7: U=(0,1,0,-1), V=(0,0,1,1), W=(1,0,0,0)
        
        #    Add these 7 rank-1 tensors to U, V, W.
        #    Global indices:
        #    i in {0,1} -> oi, oi+1
        #    j in {0,1} -> oj, oj+1
        #    k in {0,1} -> ok, ok+1
        
        #    U indices: i*m + j
        #    V indices: j*p + k
        #    W indices: k*n + i
        
        for t in range(7):
            col = col_start + t
            # U
            U.append([0] * col)
            # V
            V.append([0] * col)
            # W
            W.append([0] * col)
            
            if t == 0:
                u_vals = [(0,0,1), (1,1,1)] # (i,j,val)
                v_vals = [(0,0,1), (1,1,1)]
                w_vals = [(0,0,1), (1,1,1)]
            elif t == 1:
                u_vals = [(1,0,1), (1,1,1)]
                v_vals = [(0,0,1)]
                w_vals = [(0,0,1), (1,1,-1)]
            elif t == 2:
                u_vals = [(0,0,1)]
                v_vals = [(0,1,1), (1,1,-1)]
                w_vals = [(0,1,1), (1,1,1)]
            elif t == 3:
                u_vals = [(1,1,1)]
                v_vals = [(1,0,1), (0,0,-1)]
                w_vals = [(0,0,1), (1,0,1)]
            elif t == 4:
                u_vals = [(0,0,1), (0,1,1)]
                v_vals = [(1,1,1)]
                w_vals = [(0,0,-1), (0,1,1)]
            elif t == 5:
                u_vals = [(0,0,-1), (1,0,1)]
                v_vals = [(0,0,1), (0,1,1)]
                w_vals = [(1,1,1)]
            elif t == 6:
                u_vals = [(0,1,1), (1,1,-1)]
                v_vals = [(1,0,1), (1,1,1)]
                w_vals = [(0,0,1)]
            
            for i, j, val in u_vals:
                U[-1][oi + i, oj + j] = val
            for j, k, val in v_vals:
                V[-1][oj + j, ok + k] = val
            for k, i, val in w_vals:
                W[-1][ok + k, oi + i] = val

    # Iterate over blocks
    for r in range(2):
        for c in range(3):
            for p_idx in range(3):
                # Get block sizes
                a_h = n_splits[r]
                a_w = m_splits[c]
                b_h = m_splits[c]
                b_w = p_splits[p_idx]
                
                # Check if 2x2x2
                if a_h == 2 and a_w == 2 and b_h == 2 and b_w == 2:
                    # Strassen
                    # Get offsets
                    oi = n_offsets[r]
                    oj = m_offsets[c]
                    ok = p_offsets[p_idx]
                    
                    # Add 7 columns
                    col_start = len(U)
                    add_strassen(oi, oj, ok, col_start)
                else:
                    # Schoolbook
                    # a_h x a_w x b_w
                    for i in range(a_h):
                        for j in range(a_w):
                            for k in range(b_w):
                                col = len(U)
                                add_rank1(n_offsets[r] + i, m_offsets[c] + j, p_offsets[p_idx] + k, col)
    
    return {"U": U, "V": V, "W": W}
