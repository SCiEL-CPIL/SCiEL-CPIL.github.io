import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    
    # Helper to create a tensor decomposition for small n, m, p
    # Returns U, V, W as lists of lists
    def get_decomposition(sn, sm, sp):
        # Use schoolbook for very small or if Strassen not applicable
        # For 2x2x2 use Strassen
        if sn == 2 and sm == 2 and sp == 2:
            # Strassen rank 7
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
            # Strassen formulas
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B12 - B22)
            # M4 = A22(B21 - B11)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            
            # We need to map these to U, V, W
            # U rows: (0,0), (0,1), (1,0), (1,1)
            # V rows: (0,0), (0,1), (1,0), (1,1)
            # W rows: (0,0), (0,1), (1,0), (1,1) -- Wait, W index is k*n+i.
            # For 2x2x2, W index is k*2+i.
            
            # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            # U: A00=1, A11=1. V: B00=1, B11=1.
            # W: C00=1, C11=1.
            # This is not a single term. It's a sum.
            # Strassen decomposes the product into 7 products of linear forms.
            # Each product is $L_A \cdot L_B \cdot L_C$.
            # So each term r has U, V, W vectors.
            
            # Term 1: (A00+A11)(B00+B11)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1 # C00, C11?
            # Wait, $C_{00} = M1 - M4 - M6 + M7$?
            # Standard Strassen:
            # C00 = M1 - M4 - M6 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M5 - M7
            
            # So we need to add contributions to W.
            # Term 1 (M1): U: A00, A11. V: B00, B11.
            # W for C00: +1. W for C11: +1.
            U[0][0]=1; U[3][0]=1
            V[0][0]=1; V[3][0]=1
            W[0][0]=1; W[3][0]=1
            
            # Term 2 (M2): (A10+A11)B00
            U[2][1]=1; U[3][1]=1
            V[0][1]=1
            W[2][1]=1 # C10 += M2
            
            # Term 3 (M3): A00(B01-B11)
            U[0][2]=1
            V[1][2]=1; V[3][2]=-1
            W[1][2]=1 # C01 += M3
            
            # Term 4 (M4): A11(B10-B00)
            U[3][3]=1
            V[2][3]=1; V[0][3]=-1
            W[0][3]=-1; W[2][3]=1 # C00 -= M4, C10 += M4
            
            # Term 5 (M5): (A00+A01)B11
            U[0][4]=1; U[1][4]=1
            V[3][4]=1
            W[1][4]=1; W[3][4]=-1 # C01 += M5, C11 -= M5?
            # C11 = M1 - M3 + M5 - M7. So +M5.
            # C01 = M3 + M5. So +M5.
            # Wait, W[3][4] is C11.
            W[1][4]=1; W[3][4]=1
            
            # Term 6 (M6): (A10-A00)(B00+B01)
            U[2][5]=1; U[0][5]=-1
            V[0][5]=1; V[1][5]=1
            W[0][5]=-1 # C00 -= M6
            
            # Term 7 (M7): (A01-A11)(B10+B11)
            U[1][6]=1; U[3][6]=-1
            V[2][6]=1; V[3][6]=1
            W[0][6]=1; W[1][6]=-1 # C00 += M7, C01 -= M7?
            # C00 = ... + M7.
            # C11 = ... - M7.
            # C01 = M3 + M5. No M7.
            # Let's re-verify Strassen formulas.
            # C00 = M1 - M4 - M6 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M5 - M7
            
            # My W assignments:
            # T1: W[0]=1, W[3]=1. (C00, C11)
            # T2: W[2]=1. (C10)
            # T3: W[1]=1. (C01)
            # T4: W[0]=-1, W[2]=1. (C00, C10)
            # T5: W[1]=1, W[3]=1. (C01, C11)
            # T6: W[0]=-1. (C00)
            # T7: W[0]=1, W[3]=-1. (C00, C11)
            
            # Check C00: 1 (T1) -1 (T4) -1 (T6) +1 (T7) = 0?
            # Wait. C00 should be M1 - M4 - M6 + M7.
            # Coeff of M1 is 1.
            # Coeff of M4 is -1.
            # Coeff of M6 is -1.
            # Coeff of M7 is 1.
            # Sum = 1 - 1 - 1 + 1 = 0.
            # This is wrong. C00 is not 0.
            # The decomposition is $C = \sum M_k \cdot \text{matrix}_k$.
            # So $C_{00} = 1 \cdot M1 + (-1) \cdot M4 + (-1) \cdot M6 + 1 \cdot M7$.
            # My W[0] accumulates these coefficients.
            # W[0][0] = 1 (from T1)
            # W[0][3] = -1 (from T4)
            # W[0][5] = -1 (from T6)
            # W[0][6] = 1 (from T7)
            # Sum for C00 is $\sum_r W[0][r] \dots$
            # Yes, this is correct.
            
            # Check C11:
            # W[3][0] = 1 (T1)
            # W[3][4] = 1 (T5)
            # W[3][6] = -1 (T7)
            # C11 = M1 + M5 - M7.
            # Formula: M1 - M3 + M5 - M7.
            # Missing -M3?
            # T3: W[1][2]=1. No W[3].
            # So C11 does not get M3.
            # Formula says C11 has -M3.
            # Let's check Strassen again.
            # C11 = M1 - M3 + M5 - M7.
            # So W[3][2] should be -1.
            # I missed it in T3.
            W[3][2] = -1
            
            # Check C01:
            # W[1][2] = 1 (T3)
            # W[1][4] = 1 (T5)
            # C01 = M3 + M5.
            # Formula: M3 + M5. Correct.
            
            # Check C10:
            # W[2][1] = 1 (T2)
            # W[2][3] = 1 (T4)
            # C10 = M2 + M4.
            # Formula: M2 + M4. Correct.
            
            return U, V, W

        # Generic schoolbook
        rank = sn * sm * sp
        U = [[0]*rank for _ in range(sn*sm)]
        V = [[0]*rank for _ in range(sm*sp)]
        W = [[0]*rank for _ in range(sp*sn)]
        col = 0
        for i in range(sn):
            for j in range(sm):
                for k in range(sp):
                    U[i*sm+j][col] = 1
                    V[j*sp+k][col] = 1
                    W[k*sn+i][col] = 1
                    col += 1
        return U, V, W

    # Construct small tensors
    # 2x2x2
    U_222, V_222, W_222 = get_decomposition(2, 2, 2)
    # 2x3x2
    # M_232 = M_222 (A11 B11) + M_212 (A12 B21)
    # M_212 rank 4
    U_212, V_212, W_212 = get_decomposition(2, 1, 2) # Schoolbook 2*1*2=4
    # Combine
    rank_232 = 7 + 4
    U_232 = [[0]*rank_232 for _ in range(2*3)]
    V_232 = [[0]*rank_232 for _ in range(3*2)]
    W_232 = [[0]*rank_232 for _ in range(2*2)]
    
    # Copy 222 part
    # A indices: 0,1 (j=0,1). B indices: 0,1 (j=0,1).
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_232[i*3+j][r] = U_222[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_232[j*2+k][r] = V_222[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W_232[k*2+i][r] = W_222[k*2+i][r]
                
    # Copy 212 part
    # A indices: j=2 (local 0). B indices: j=0,1 (local 0,1).
    # A is 2x1. Local index 0 corresponds to global j=2.
    # B is 1x2. Local index j=0,1 corresponds to global j=2+0, 2+1?
    # Wait, B in M_232 is 3x2. Rows 0,1,2.
    # A_12 is 2x1 (col 2 of A). B_21 is 1x2 (row 2 of B).
    # So A local j=0 -> global j=2.
    # B local j=0 -> global j=2.
    # B local j=1 -> global j=2? No.
    # B_21 is the bottom row of B (row 2). It has 2 columns.
    # So B local row 0 corresponds to global row 2.
    # B local col 0,1 correspond to global col 0,1.
    # So V index: local (j,k) -> global (2, k).
    
    for r in range(4):
        r_off = 7 + r
        # U: A is 2x1. Local i=0,1. j=0.
        for i in range(2):
            U_232[i*3+2][r_off] = U_212[i*1+0][r]
        # V: B is 1x2. Local j=0. k=0,1.
        for k in range(2):
            V_232[2*2+k][r_off] = V_212[0*2+k][r]
        # W: C is 2x2.
        for k in range(2):
            for i in range(2):
                W_232[k*2+i][r_off] = W_212[k*1+i][r]

    # 2x2x3
    # M_223 = M_222 (A B11) + M_221 (A B12)
    U_221, V_221, W_221 = get_decomposition(2, 2, 1) # Rank 4
    rank_223 = 7 + 4
    U_223 = [[0]*rank_223 for _ in range(2*2)]
    V_223 = [[0]*rank_223 for _ in range(2*3)]
    W_223 = [[0]*rank_223 for _ in range(3*2)]
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_223[i*2+j][r] = U_222[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_223[j*3+k][r] = V_222[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W_223[k*2+i][r] = W_222[k*2+i][r]
                
    for r in range(4):
        r_off = 7 + r
        # U: A is 2x2.
        for i in range(2):
            for j in range(2):
                U_223[i*2+j][r_off] = U_221[i*2+j][r]
        # V: B is 2x1. Local k=0 -> global k=2.
        for j in range(2):
            V_223[j*3+2][r_off] = V_221[j*1+0][r]
        # W: C is 2x1 (local) -> 2x1 part of 2x3.
        # W index k*2+i. Local k=0.
        for i in range(2):
            W_223[0*2+i][r_off] = W_221[0*2+i][r] # Wait, W_221 is 1x2? No, 1*2=2 rows.
            # W_221 shape (1*2, 4).
            # W_223 shape (3*2, 11).
            # Local k=0. Global k=2.
            # W index 2*2+i.
            W_223[2*2+i][r_off] = W_221[0*2+i][r]

    # 2x3x3
    # M_233 = M_223 (A11 B11) + M_213 (A12 B21)
    U_213, V_213, W_213 = get_decomposition(2, 1, 3) # Rank 6
    rank_233 = 11 + 6
    U_233 = [[0]*rank_233 for _ in range(2*3)]
    V_233 = [[0]*rank_233 for _ in range(3*3)]
    W_233 = [[0]*rank_233 for _ in range(3*2)]
    
    for r in range(11):
        for i in range(2):
            for j in range(2):
                U_233[i*3+j][r] = U_223[i*2+j][r]
        for j in range(2):
            for k in range(3):
                V_233[j*3+k][r] = V_223[j*3+k][r]
        for k in range(3):
            for i in range(2):
                W_233[k*2+i][r] = W_223[k*2+i][r]
                
    for r in range(6):
        r_off = 11 + r
        # U: A12 (2x1). Local j=0 -> global j=2.
        for i in range(2):
            U_233[i*3+2][r_off] = U_213[i*1+0][r]
        # V: B21 (1x3). Local j=0 -> global j=2.
        for k in range(3):
            V_233[2*3+k][r_off] = V_213[0*3+k][r]
        # W: C is 2x3.
        for k in range(3):
            for i in range(2):
                W_233[k*2+i][r_off] = W_213[k*1+i][r]

    # Assemble 4x5x5
    rank_total = 2 * (7 + 11 + 11 + 17) # 2*46 = 92
    U = [[0]*rank_total for _ in range(4*5)]
    V = [[0]*rank_total for _ in range(5*5)]
    W = [[0]*rank_total for _ in range(5*4)]
    
    # Block 1: A11 B11 -> C11 (i=0,1; j=0,1; k=0,1)
    # Uses M_222
    r_off = 0
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*5+j][r_off+r] = U_222[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*5+k][r_off+r] = V_222[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*4+i][r_off+r] = W_222[k*2+i][r]
                
    # Block 2: A12 B21 -> C11 (i=0,1; j=2,3,4; k=0,1)
    # Uses M_232
    r_off = 7
    for r in range(11):
        for i in range(2):
            for j in range(3):
                U[i*5+(2+j)][r_off+r] = U_232[i*3+j][r]
        for j in range(3):
            for k in range(2):
                V[(2+j)*5+k][r_off+r] = V_232[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*4+i][r_off+r] = W_232[k*2+i][r]

    # Block 3: A11 B12 -> C12 (i=0,1; j=0,1; k=2,3,4)
    # Uses M_223
    r_off = 18
    for r in range(11):
        for i in range(2):
            for j in range(2):
                U[i*5+j][r_off+r] = U_223[i*2+j][r]
        for j in range(2):
            for k in range(3):
                V[j*5+(2+k)][r_off+r] = V_223[j*3+k][r]
        for k in range(3):
            for i in range(2):
                W[(2+k)*4+i][r_off+r] = W_223[k*2+i][r]

    # Block 4: A12 B22 -> C12 (i=0,1; j=2,3,4; k=2,3,4)
    # Uses M_233
    r_off = 29
    for r in range(17):
        for i in range(2):
            for j in range(3):
                U[i*5+(2+j)][r_off+r] = U_233[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V[(2+j)*5+(2+k)][r_off+r] = V_233[j*3+k][r]
        for k in range(3):
            for i in range(2):
                W[(2+k)*4+i][r_off+r] = W_233[k*2+i][r]

    # Blocks 5-8 for i=2,3
    r_off = 46
    # Block 5: A21 B11 -> C21 (i=2,3; j=0,1; k=0,1)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[(2+i)*5+j][r_off+r] = U_222[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*5+k][r_off+r] = V_222[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*4+(2+i)][r_off+r] = W_222[k*2+i][r]

    # Block 6: A22 B21 -> C21 (i=2,3; j=2,3,4; k=0,1)
    r_off = 53
    for r in range(11):
        for i in range(2):
            for j in range(3):
                U[(2+i)*5+(2+j)][r_off+r] = U_232[i*3+j][r]
        for j in range(3):
            for k in range(2):
                V[(2+j)*5+k][r_off+r] = V_232[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*4+(2+i)][r_off+r] = W_232[k*2+i][r]

    # Block 7: A21 B12 -> C22 (i=2,3; j=0,1; k=2,3,4)
    r_off = 64
    for r in range(11):
        for i in range(2):
            for j in range(2):
                U[(2+i)*5+j][r_off+r] = U_223[i*2+j][r]
        for j in range(2):
            for k in range(3):
                V[j*5+(2+k)][r_off+r] = V_223[j*3+k][r]
        for k in range(3):
            for i in range(2):
                W[(2+k)*4+(2+i)][r_off+r] = W_223[k*2+i][r]

    # Block 8: A22 B22 -> C22 (i=2,3; j=2,3,4; k=2,3,4)
    r_off = 75
    for r in range(17):
        for i in range(2):
            for j in range(3):
                U[(2+i)*5+(2+j)][r_off+r] = U_233[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V[(2+j)*5+(2+k)][r_off+r] = V_233[j*3+k][r]
        for k in range(3):
            for i in range(2):
                W[(2+k)*4+(2+i)][r_off+r] = W_233[k*2+i][r]

    return {"U": U, "V": V, "W": W}

