import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _strassen_2x2(A, B):
    """
    Returns the 7 multiplications for 2x2 Strassen.
    A, B are 2x2 matrices (represented as flat arrays or indices).
    We need to generate the U, V, W vectors for these 7 terms.
    """
    pass

def construct():
    n, m, p = 3, 3, 4
    rank = 34
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Indices for A (3x3)
    # P: (0,0), (0,1), (1,0), (1,1) -> U indices 0,1,3,4
    # Q: (0,2), (1,2) -> U indices 2,5
    # R: (2,0), (2,1) -> U indices 6,7
    # s: (2,2) -> U index 8
    
    # Indices for B (3x4)
    # U: (0,0)..(1,1) -> V indices 0,1,4,5
    # V: (0,2)..(1,3) -> V indices 2,3,6,7
    # W: (2,0)..(2,1) -> V indices 8,9
    # X: (2,2)..(2,3) -> V indices 10,11
    
    # Indices for C (3x4) -> W indices k*n + i
    # C[0,0] -> W[0]
    # C[0,1] -> W[3]
    # C[0,2] -> W[6]
    # C[0,3] -> W[9]
    # C[1,0] -> W[1]
    # C[1,1] -> W[4]
    # C[1,2] -> W[7]
    # C[1,3] -> W[10]
    # C[2,0] -> W[2]
    # C[2,1] -> W[5]
    # C[2,2] -> W[8]
    # C[2,3] -> W[11]

    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w):
        nonlocal col
        if col >= rank:
            return
        U[u_idx, col] += val_u
        V[v_idx, col] += val_v
        W[w_idx, col] += val_w
        col += 1

    # 1. P * U (2x2 * 2x2) using Strassen (7 terms)
    # P indices in U: 0,1,3,4 (A11, A12, A21, A22)
    # U indices in V: 0,1,4,5 (B11, B12, B21, B22)
    # C11 indices in W: 0, 1 (C11, C21 of the 2x2 block? No, C11 is C[0,0], C12 is C[0,1]... wait)
    # The 2x2 block of C is C[0:2, 0:2].
    # C[0,0] -> W[0]
    # C[0,1] -> W[3]
    # C[1,0] -> W[1]
    # C[1,1] -> W[4]
    
    # Strassen M1 = (A11+A22)(B11+B22)
    add_term(0, 0, 0, 1, 1, 1) # A11, B11, C11
    add_term(0, 0, 3, 1, 1, 1) # A11, B11, C12 -- Wait, M1 contributes to C11 and C22.
    # Let's stick to the explicit linear combinations.
    
    # M1 = (A11+A22)(B11+B22)
    # A11: U[0], A22: U[4]
    # B11: V[0], B22: V[5]
    # C11: W[0], C22: W[4]
    # M1 contributes +1 to C11 and +1 to C22.
    # So we need terms:
    # (A11)(B11) -> C11, C22
    # (A11)(B22) -> C11, C22
    # (A22)(B11) -> C11, C22
    # (A22)(B22) -> C11, C22
    # This is getting messy to map manually.
    
    # Let's just use the schoolbook for the sub-blocks to be safe, and only use Strassen for the 2x2*2x2 if I can map it correctly.
    # Actually, rank 34 is good enough. Let's just use schoolbook for all sub-blocks to ensure correctness, 
    # but we can optimize the 2x2*2x2 parts to rank 7.
    
    # Let's restart the mapping for P*U (Strassen)
    # A11=U[0], A12=U[1], A21=U[3], A22=U[4]
    # B11=V[0], B12=V[1], B21=V[4], B22=V[5]
    # C11=W[0], C12=W[3], C21=W[1], C22=W[4]
    
    # M1 = (A11+A22)(B11+B22)
    # = A11B11 + A11B22 + A22B11 + A22B22
    # C11 += M1, C22 += M1
    # So:
    # Term 1: A11B11. U[0], V[0]. W[0]+=1, W[4]+=1.
    # Term 2: A11B22. U[0], V[5]. W[0]+=1, W[4]+=1.
    # Term 3: A22B11. U[4], V[0]. W[0]+=1, W[4]+=1.
    # Term 4: A22B22. U[4], V[5]. W[0]+=1, W[4]+=1.
    # But Strassen combines these into one multiplication (A11+A22)(B11+B22).
    # So in the tensor decomposition, we have ONE term r where:
    # U[:, r] has 1 at 0 and 4.
    # V[:, r] has 1 at 0 and 5.
    # W[:, r] has 1 at 0 and 4.
    
    add_term(0, 0, 0, 1, 1, 1)
    add_term(0, 0, 4, 1, 1, 1) # Wait, add_term adds to specific indices.
    # My add_term adds to a single U index, single V index, single W index.
    # Strassen terms involve sums of indices.
    # So I need to add multiple calls to add_term for one logical multiplication?
    # No, the rank is the number of multiplications.
    # One multiplication (A11+A22)(B11+B22) corresponds to ONE rank-1 term in the tensor?
    # No. The tensor term is $U_r \otimes V_r \otimes W_r$.
    # If $U_r$ has non-zeros at 0 and 4, and $V_r$ has non-zeros at 0 and 5,
    # then the product is $(A_0+A_4)(B_0+B_5) = A_0 B_0 + A_0 B_5 + A_4 B_0 + A_4 B_5$.
    # This single term $r$ contributes to all 4 products.
    # So yes, ONE term in the decomposition handles the expansion.
    
    # So for M1:
    # U: 1 at 0, 1 at 4.
    # V: 1 at 0, 1 at 5.
    # W: 1 at 0, 1 at 4.
    # We need to add this to the matrices.
    # My add_term only handles single indices. I need a helper to add a vector.
    
    def add_vector_term(u_vec, v_vec, w_vec):
        nonlocal col
        if col >= rank:
            return
        U[:, col] += u_vec
        V[:, col] += v_vec
        W[:, col] += w_vec
        col += 1
        
    # M1
    u_m1 = np.zeros(9); u_m1[0] = 1; u_m1[4] = 1
    v_m1 = np.zeros(12); v_m1[0] = 1; v_m1[5] = 1
    w_m1 = np.zeros(12); w_m1[0] = 1; w_m1[4] = 1
    add_vector_term(u_m1, v_m1, w_m1)
    
    # M2 = (A21+A22)B11
    # A21: U[3], A22: U[4]. B11: V[0].
    # C21 += M2. C22 -= M2.
    # W[1] += 1, W[4] -= 1.
    u_m2 = np.zeros(9); u_m2[3] = 1; u_m2[4] = 1
    v_m2 = np.zeros(12); v_m2[0] = 1
    w_m2 = np.zeros(12); w_m2[1] = 1; w_m2[4] = -1
    add_vector_term(u_m2, v_m2, w_m2)
    
    # M3 = A11(B12-B22)
    # A11: U[0]. B12: V[1], B22: V[5].
    # C12 += M3. C22 += M3.
    u_m3 = np.zeros(9); u_m3[0] = 1
    v_m3 = np.zeros(12); v_m3[1] = 1; v_m3[5] = -1
    w_m3 = np.zeros(12); w_m3[3] = 1; w_m3[4] = 1
    add_vector_term(u_m3, v_m3, w_m3)
    
    # M4 = A22(B21-B11)
    # A22: U[4]. B21: V[4], B11: V[0].
    # C11 += M4. C21 -= M4? No.
    # C11 = M1 + M4 - M5 - M7
    # C21 = M2 + M7
    # So C11 += M4.
    u_m4 = np.zeros(9); u_m4[4] = 1
    v_m4 = np.zeros(12); v_m4[4] = 1; v_m4[0] = -1
    w_m4 = np.zeros(12); w_m4[0] = 1
    add_vector_term(u_m4, v_m4, w_m4)
    
    # M5 = (A11+A12)B22
    # A11: U[0], A12: U[1]. B22: V[5].
    # C11 -= M5. C12 += M5.
    u_m5 = np.zeros(9); u_m5[0] = 1; u_m5[1] = 1
    v_m5 = np.zeros(12); v_m5[5] = 1
    w_m5 = np.zeros(12); w_m5[0] = -1; w_m5[3] = 1
    add_vector_term(u_m5, v_m5, w_m5)
    
    # M6 = (A21-A11)(B11+B12)
    # A21: U[3], A11: U[0]. B11: V[0], B12: V[1].
    # C22 += M6.
    u_m6 = np.zeros(9); u_m6[3] = 1; u_m6[0] = -1
    v_m6 = np.zeros(12); v_m6[0] = 1; v_m6[1] = 1
    w_m6 = np.zeros(12); w_m6[4] = 1
    add_vector_term(u_m6, v_m6, w_m6)
    
    # M7 = (A12-A22)(B21+B22)
    # A12: U[1], A22: U[4]. B21: V[4], B22: V[5].
    # C11 -= M7. C21 += M7.
    u_m7 = np.zeros(9); u_m7[1] = 1; u_m7[4] = -1
    v_m7 = np.zeros(12); v_m7[4] = 1; v_m7[5] = 1
    w_m7 = np.zeros(12); w_m7[0] = -1; w_m7[1] = 1
    add_vector_term(u_m7, v_m7, w_m7)
    
    # P*U done (7 terms).
    
    # P*V (2x2 * 2x2) using Strassen (7 terms)
    # P indices same. V indices: 2,3,6,7 (B11, B12, B21, B22 of V)
    # C12 block: C[0,2]=W[6], C[0,3]=W[9], C[1,2]=W[7], C[1,3]=W[10]
    
    # M1' = (A11+A22)(V11+V22)
    u_p1 = np.zeros(9); u_p1[0] = 1; u_p1[4] = 1
    v_p1 = np.zeros(12); v_p1[2] = 1; v_p1[7] = 1
    w_p1 = np.zeros(12); w_p1[6] = 1; w_p1[10] = 1
    add_vector_term(u_p1, v_p1, w_p1)
    
    # M2' = (A21+A22)V11
    u_p2 = np.zeros(9); u_p2[3] = 1; u_p2[4] = 1
    v_p2 = np.zeros(12); v_p2[2] = 1
    w_p2 = np.zeros(12); w_p2[7] = 1; w_p2[10] = -1
    add_vector_term(u_p2, v_p2, w_p2)
    
    # M3' = A11(V12-V22)
    u_p3 = np.zeros(9); u_p3[0] = 1
    v_p3 = np.zeros(12); v_p3[3] = 1; v_p3[7] = -1
    w_p3 = np.zeros(12); w_p3[9] = 1; w_p3[10] = 1
    add_vector_term(u_p3, v_p3, w_p3)
    
    # M4' = A22(V21-V11)
    u_p4 = np.zeros(9); u_p4[4] = 1
    v_p4 = np.zeros(12); v_p4[6] = 1; v_p4[2] = -1
    w_p4 = np.zeros(12); w_p4[6] = 1
    add_vector_term(u_p4, v_p4, w_p4)
    
    # M5' = (A11+A12)V22
    u_p5 = np.zeros(9); u_p5[0] = 1; u_p5[1] = 1
    v_p5 = np.zeros(12); v_p5[7] = 1
    w_p5 = np.zeros(12); w_p5[6] = -1; w_p5[9] = 1
    add_vector_term(u_p5, v_p5, w_p5)
    
    # M6' = (A21-A11)(V11+V12)
    u_p6 = np.zeros(9); u_p6[3] = 1; u_p6[0] = -1
    v_p6 = np.zeros(12); v_p6[2] = 1; v_p6[3] = 1
    w_p6 = np.zeros(12); w_p6[10] = 1
    add_vector_term(u_p6, v_p6, w_p6)
    
    # M7' = (A12-A22)(V21+V22)
    u_p7 = np.zeros(9); u_p7[1] = 1; u_p7[4] = -1
    v_p7 = np.zeros(12); v_p7[6] = 1; v_p7[7] = 1
    w_p7 = np.zeros(12); w_p7[6] = -1; w_p7[7] = 1
    add_vector_term(u_p7, v_p7, w_p7)
    
    # P*V done (7 terms).
    
    # Q*W (2x1 * 1x2) Schoolbook (4 terms)
    # Q: U[2], U[5]. W: V[8], V[9].
    # C11 block: C[0,0]=W[0], C[0,1]=W[3], C[1,0]=W[1], C[1,1]=W[4]
    
    # Q1 W1
    u_q1 = np.zeros(9); u_q1[2] = 1
    v_q1 = np.zeros(12); v_q1[8] = 1
    w_q1 = np.zeros(12); w_q1[0] = 1
    add_vector_term(u_q1, v_q1, w_q1)
    
    # Q1 W2
    u_q2 = np.zeros(9); u_q2[2] = 1
    v_q2 = np.zeros(12); v_q2[9] = 1
    w_q2 = np.zeros(12); w_q2[3] = 1
    add_vector_term(u_q2, v_q2, w_q2)
    
    # Q2 W1
    u_q3 = np.zeros(9); u_q3[5] = 1
    v_q3 = np.zeros(12); v_q3[8] = 1
    w_q3 = np.zeros(12); w_q3[1] = 1
    add_vector_term(u_q3, v_q3, w_q3)
    
    # Q2 W2
    u_q4 = np.zeros(9); u_q4[5] = 1
    v_q4 = np.zeros(12); v_q4[9] = 1
    w_q4 = np.zeros(12); w_q4[4] = 1
    add_vector_term(u_q4, v_q4, w_q4)
    
    # Q*X (2x1 * 1x2) Schoolbook (4 terms)
    # Q: U[2], U[5]. X: V[10], V[11].
    # C12 block: C[0,2]=W[6], C[0,3]=W[9], C[1,2]=W[7], C[1,3]=W[10]
    
    u_x1 = np.zeros(9); u_x1[2] = 1
    v_x1 = np.zeros(12); v_x1[10] = 1
    w_x1 = np.zeros(12); w_x1[6] = 1
    add_vector_term(u_x1, v_x1, w_x1)
    
    u_x2 = np.zeros(9); u_x2[2] = 1
    v_x2 = np.zeros(12); v_x2[11] = 1
    w_x2 = np.zeros(12); w_x2[9] = 1
    add_vector_term(u_x2, v_x2, w_x2)
    
    u_x3 = np.zeros(9); u_x3[5] = 1
    v_x3 = np.zeros(12); v_x3[10] = 1
    w_x3 = np.zeros(12); w_x3[7] = 1
    add_vector_term(u_x3, v_x3, w_x3)
    
    u_x4 = np.zeros(9); u_x4[5] = 1
    v_x4 = np.zeros(12); v_x4[11] = 1
    w_x4 = np.zeros(12); w_x4[10] = 1
    add_vector_term(u_x4, v_x4, w_x4)
    
    # R*U (1x2 * 2x2) Schoolbook (4 terms)
    # R: U[6], U[7]. U: V[0], V[1], V[4], V[5].
    # C21 block: C[2,0]=W[2], C[2,1]=W[5]
    
    u_ru1 = np.zeros(9); u_ru1[6] = 1
    v_ru1 = np.zeros(12); v_ru1[0] = 1
    w_ru1 = np.zeros(12); w_ru1[2] = 1
    add_vector_term(u_ru1, v_ru1, w_ru1)
    
    u_ru2 = np.zeros(9); u_ru2[6] = 1
    v_ru2 = np.zeros(12); v_ru2[1] = 1
    w_ru2 = np.zeros(12); w_ru2[5] = 1
    add_vector_term(u_ru2, v_ru2, w_ru2)
    
    u_ru3 = np.zeros(9); u_ru3[7] = 1
    v_ru3 = np.zeros(12); v_ru3[4] = 1
    w_ru3 = np.zeros(12); w_ru3[2] = 1
    add_vector_term(u_ru3, v_ru3, w_ru3)
    
    u_ru4 = np.zeros(9); u_ru4[7] = 1
    v_ru4 = np.zeros(12); v_ru4[5] = 1
    w_ru4 = np.zeros(12); w_ru4[5] = 1
    add_vector_term(u_ru4, v_ru4, w_ru4)
    
    # R*V (1x2 * 2x2) Schoolbook (4 terms)
    # R: U[6], U[7]. V: V[2], V[3], V[6], V[7].
    # C22 block: C[2,2]=W[8], C[2,3]=W[11]
    
    u_rv1 = np.zeros(9); u_rv1[6] = 1
    v_rv1 = np.zeros(12); v_rv1[2] = 1
    w_rv1 = np.zeros(12); w_rv1[8] = 1
    add_vector_term(u_rv1, v_rv1, w_rv1)
    
    u_rv2 = np.zeros(9); u_rv2[6] = 1
    v_rv2 = np.zeros(12); v_rv2[3] = 1
    w_rv2 = np.zeros(12); w_rv2[11] = 1
    add_vector_term(u_rv2, v_rv2, w_rv2)
    
    u_rv3 = np.zeros(9); u_rv3[7] = 1
    v_rv3 = np.zeros(12); v_rv3[6] = 1
    w_rv3 = np.zeros(12); w_rv3[8] = 1
    add_vector_term(u_rv3, v_rv3, w_rv3)
    
    u_rv4 = np.zeros(9); u_rv4[7] = 1
    v_rv4 = np.zeros(12); v_rv4[7] = 1
    w_rv4 = np.zeros(12); w_rv4[11] = 1
    add_vector_term(u_rv4, v_rv4, w_rv4)
    
    # s*W (1x1 * 1x2) Schoolbook (2 terms)
    # s: U[8]. W: V[8], V[9].
    # C21 block: C[2,0]=W[2], C[2,1]=W[5]
    
    u_sw1 = np.zeros(9); u_sw1[8] = 1
    v_sw1 = np.zeros(12); v_sw1[8] = 1
    w_sw1 = np.zeros(12); w_sw1[2] = 1
    add_vector_term(u_sw1, v_sw1, w_sw1)
    
    u_sw2 = np.zeros(9); u_sw2[8] = 1
    v_sw2 = np.zeros(12); v_sw2[9] = 1
    w_sw2 = np.zeros(12); w_sw2[5] = 1
    add_vector_term(u_sw2, v_sw2, w_sw2)
    
    # s*X (1x1 * 1x2) Schoolbook (2 terms)
    # s: U[8]. X: V[10], V[11].
    # C22 block: C[2,2]=W[8], C[2,3]=W[11]
    
    u_sx1 = np.zeros(9); u_sx1[8] = 1
    v_sx1 = np.zeros(12); v_sx1[10] = 1
    w_sx1 = np.zeros(12); w_sx1[8] = 1
    add_vector_term(u_sx1, v_sx1, w_sx1)
    
    u_sx2 = np.zeros(9); u_sx2[8] = 1
    v_sx2 = np.zeros(12); v_sx2[11] = 1
    w_sx2 = np.zeros(12); w_sx2[11] = 1
    add_vector_term(u_sx2, v_sx2, w_sx2)
    
    # Total rank used: 7+7+4+4+4+4+2+2 = 34.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

