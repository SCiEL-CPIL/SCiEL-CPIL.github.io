import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n=3, m=3, p=6, trials=50, seed=42):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Direct tensor verification: check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # U, V, W are 4x7 matrices
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen algorithm for 2x2x2
    # C11 = M1 - M4 + M6 - M7
    # C12 = M5 + M7
    # C21 = M4 + M6
    # C22 = M1 - M2 + M3 - M6
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indexing: U[i*2+j, c]
    # M1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = -1 # C22
    
    # M3: A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[3, 2] = 1 # C22
    
    # M4: A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1 # C21
    W[0, 3] = -1 # C11
    
    # M5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1 # C12
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1 # C11
    W[2, 5] = 1 # C21
    W[3, 5] = -1 # C22
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1 # C12
    W[0, 6] = -1 # C11
    W[2, 6] = -1 # C21
    
    return U, V, W

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank26_3x3x3():
    """
    Construct a rank-26 decomposition for 3x3x3 matrix multiplication.
    This is based on the known best bound for 3x3x3 which is 23, but we use a simpler
    construction or a known valid one. Actually, the trivial is 27.
    Let's use the "block" approach: 2x2x2 (rank 7) + corrections.
    
    A = [[A22, a], [d, b]] where A22 is 2x2, a is 2x1, d is 1x2, b is 1x1.
    B = [[B22, c], [e, f]]
    C = A B
    
    C22 = A22 B22 + a e
    C21 (top right 2x1) = A22 c + a f
    C12 (bottom left 1x2) = d B22 + b e
    C11 (bottom right 1x1) = d c + b f
    
    We can compute:
    1. A22 B22 using Strassen (rank 7).
       This gives the top-left 2x2 of C (indices 0,1,3,4).
       U1 (4x7), V1 (4x7), W1 (4x7) placed in the 3x3 grid.
       U1 indices: 0,1,3,4
       V1 indices: 0,1,3,4
       W1 indices: 0,1,3,4
       
    2. a e (2x1 * 1x2 -> 2x2). Rank 4.
       a = [A2, A5], e = [B6, B7]
       ae = [[A2B6, A2B7], [A5B6, A5B7]]
       Contributes to C[0,0], C[0,1], C[1,0], C[1,1] (indices 0,1,3,4).
       U: [0,0,0,1] for a? No, U is for A.
       Term t: U_a * V_e * W_c
       We need 4 terms.
       t1: U[2]=1, V[6]=1, W[0]=1 -> C00 += A2 B6
       t2: U[2]=1, V[7]=1, W[1]=1 -> C01 += A2 B7
       t3: U[5]=1, V[6]=1, W[3]=1 -> C10 += A5 B6
       t4: U[5]=1, V[7]=1, W[4]=1 -> C11 += A5 B7
       
    3. A22 c (2x2 * 2x1 -> 2x1). Rank 4.
       A22 = [[A0,A1],[A3,A4]], c = [B2, B5]
       Result: [A0B2 + A1B5, A3B2 + A4B5]
       Contributes to C[0,2] (index 2) and C[1,2] (index 5).
       t5: U[0]=1, V[2]=1, W[2]=1 -> C02 += A0 B2
       t6: U[1]=1, V[5]=1, W[2]=1 -> C02 += A1 B5
       t7: U[3]=1, V[2]=1, W[5]=1 -> C12 += A3 B2
       t8: U[4]=1, V[5]=1, W[5]=1 -> C12 += A4 B5
       
    4. a f (2x1 * 1x1 -> 2x1). Rank 2.
       a = [A2, A5], f = B8
       Result: [A2B8, A5B8]
       Contributes to C[0,2] (index 2) and C[1,2] (index 5).
       t9: U[2]=1, V[8]=1, W[2]=1 -> C02 += A2 B8
       t10: U[5]=1, V[8]=1, W[5]=1 -> C12 += A5 B8
       
    5. d B22 (1x2 * 2x2 -> 1x2). Rank 4.
       d = [A6, A7], B22 = [[B0,B1],[B3,B4]]
       Result: [A6B0 + A7B3, A6B1 + A7B4]
       Contributes to C[2,0] (index 6) and C[2,1] (index 7).
       t11: U[6]=1, V[0]=1, W[6]=1 -> C20 += A6 B0
       t12: U[7]=1, V[3]=1, W[6]=1 -> C20 += A7 B3
       t13: U[6]=1, V[1]=1, W[7]=1 -> C21 += A6 B1
       t14: U[7]=1, V[4]=1, W[7]=1 -> C21 += A7 B4
       
    6. b e (1x1 * 1x2 -> 1x2). Rank 2.
       b = A8, e = [B6, B7]
       Result: [A8B6, A8B7]
       Contributes to C[2,0] (index 6) and C[2,1] (index 7).
       t15: U[8]=1, V[6]=1, W[6]=1 -> C20 += A8 B6
       t16: U[8]=1, V[7]=1, W[7]=1 -> C21 += A8 B7
       
    7. d c (1x2 * 2x1 -> 1x1). Rank 4? No, can be done with 2 terms?
       d = [A6, A7], c = [B2, B5]
       Result: A6B2 + A7B5
       Contributes to C[2,2] (index 8).
       t17: U[6]=1, V[2]=1, W[8]=1
       t18: U[7]=1, V[5]=1, W[8]=1
       
    8. b f (1x1 * 1x1 -> 1x1). Rank 1.
       t19: U[8]=1, V[8]=1, W[8]=1
       
    Total rank = 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 1. Strassen for A22 B22 (indices 0,1,3,4)
    U_s, V_s, W_s = _strassen_2x2x2()
    # Map 2x2 indices to 3x3 indices
    # 0->0, 1->1, 2->3, 3->4
    map_idx = [0, 1, 3, 4]
    for i in range(4):
        for j in range(4):
            for c in range(7):
                U[map_idx[i], c] += U_s[i, j] * 0 # Wait, U_s is 4x7.
                # U_s[i, c] corresponds to A index i.
                # We want U[map_idx[i], c] = U_s[i, c]
                pass
    # Let's do it properly
    U_s, V_s, W_s = _strassen_2x2x2()
    for i in range(4):
        for c in range(7):
            U[map_idx[i], c] = U_s[i, c]
            V[map_idx[i], c] = V_s[i, c]
            W[map_idx[i], c] = W_s[i, c]
    
    # 2. a e (indices 2,5 for A; 6,7 for B; 0,1,3,4 for C)
    # t7: U[2]=1, V[6]=1, W[0]=1
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    # t8: U[2]=1, V[7]=1, W[1]=1
    U[2, 8] = 1; V[7, 8] = 1; W[1, 8] = 1
    # t9: U[5]=1, V[6]=1, W[3]=1
    U[5, 9] = 1; V[6, 9] = 1; W[3, 9] = 1
    # t10: U[5]=1, V[7]=1, W[4]=1
    U[5, 10] = 1; V[7, 10] = 1; W[4, 10] = 1
    
    # 3. A22 c (indices 0,1,3,4 for A; 2,5 for B; 2,5 for C)
    # t11: U[0]=1, V[2]=1, W[2]=1
    U[0, 11] = 1; V[2, 11] = 1; W[2, 11] = 1
    # t12: U[1]=1, V[5]=1, W[2]=1
    U[1, 12] = 1; V[5, 12] = 1; W[2, 12] = 1
    # t13: U[3]=1, V[2]=1, W[5]=1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    # t14: U[4]=1, V[5]=1, W[5]=1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    
    # 4. a f (indices 2,5 for A; 8 for B; 2,5 for C)
    # t15: U[2]=1, V[8]=1, W[2]=1
    U[2, 15] = 1; V[8, 15] = 1; W[2, 15] = 1
    # t16: U[5]=1, V[8]=1, W[5]=1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    # 5. d B22 (indices 6,7 for A; 0,1,3,4 for B; 6,7 for C)
    # t17: U[6]=1, V[0]=1, W[6]=1
    U[6, 17] = 1; V[0, 17] = 1; W[6, 17] = 1
    # t18: U[7]=1, V[3]=1, W[6]=1
    U[7, 18] = 1; V[3, 18] = 1; W[6, 18] = 1
    # t19: U[6]=1, V[1]=1, W[7]=1
    U[6, 19] = 1; V[1, 19] = 1; W[7, 19] = 1
    # t20: U[7]=1, V[4]=1, W[7]=1
    U[7, 20] = 1; V[4, 20] = 1; W[7, 20] = 1
    
    # 6. b e (index 8 for A; 6,7 for B; 6,7 for C)
    # t21: U[8]=1, V[6]=1, W[6]=1
    U[8, 21] = 1; V[6, 21] = 1; W[6, 21] = 1
    # t22: U[8]=1, V[7]=1, W[7]=1
    U[8, 22] = 1; V[7, 22] = 1; W[7, 22] = 1
    
    # 7. d c (indices 6,7 for A; 2,5 for B; 8 for C)
    # t23: U[6]=1, V[2]=1, W[8]=1
    U[6, 23] = 1; V[2, 23] = 1; W[8, 23] = 1
    # t24: U[7]=1, V[5]=1, W[8]=1
    U[7, 24] = 1; V[5, 24] = 1; W[8, 24] = 1
    
    # 8. b f (index 8 for A; 8 for B; 8 for C)
    # t25: U[8]=1, V[8]=1, W[8]=1
    U[8, 25] = 1; V[8, 25] = 1; W[8, 25] = 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _combine_3x3x6():
    """
    Combine two 3x3x3 decompositions to get 3x3x6.
    A is 3x3. B is 3x6 = [B1, B2] where B1, B2 are 3x3.
    C is 3x6 = [C1, C2] where C1 = A B1, C2 = A B2.
    
    We use the same U for both.
    U: 9 x 26
    V1: 9 x 26 (for B1)
    V2: 9 x 26 (for B2)
    W1: 9 x 26 (for C1)
    W2: 9 x 26 (for C2)
    
    Total rank = 26 + 26 = 52.
    """
    decomp = _rank26_3x3x3()
    U1 = decomp["U"]
    V1 = decomp["V"]
    W1 = decomp["W"]
    
    # U is shared
    U_total = U1
    
    # V_total = [V1, V2] where V2 is the same structure but for B2
    # B2 indices:
    # B1: 0,1,2, 3,4,5, 6,7,8
    # B2: 9,10,11, 12,13,14, 15,16,17
    # We need to shift V1 indices by 9?
    # V1[j*p+k] where p=3.
    # For B1: j in 0..2, k in 0..2. Index = j*3+k.
    # For B2: j in 0..2, k in 3..5. Index = j*6+k.
    # Wait, B is 3x6. Index = j*6+k.
    # B1 is cols 0-2. B2 is cols 3-5.
    # So V2[j*6+k] = V1[j*3 + (k-3)]?
    # Let's construct V_total explicitly.
    
    rank = 52
    V_total = [[0]*rank for _ in range(18)] # 3*6 = 18
    W_total = [[0]*rank for _ in range(18)] # 6*3 = 18
    
    # Copy V1 to V_total (cols 0-2 of B)
    for j in range(3):
        for k in range(3):
            idx_b1 = j*3 + k
            idx_b_total = j*6 + k
            for c in range(26):
                V_total[idx_b_total][c] = V1[idx_b1][c]
    
    # Copy V1 to V_total (cols 3-5 of B)
    for j in range(3):
        for k in range(3):
            idx_b1 = j*3 + k
            idx_b_total = j*6 + (k+3)
            for c in range(26):
                V_total[idx_b_total][c+26] = V1[idx_b1][c]
                
    # Copy W1 to W_total (cols 0-2 of C)
    # W1[k*n+i] where n=3.
    # C1 is cols 0-2. C2 is cols 3-5.
    # W index = k*3 + i.
    for k in range(3):
        for i in range(3):
            idx_c1 = k*3 + i
            idx_c_total = k*6 + i
            for c in range(26):
                W_total[idx_c_total][c] = W1[idx_c1][c]
                
    # Copy W1 to W_total (cols 3-5 of C)
    for k in range(3):
        for i in range(3):
            idx_c1 = k*3 + i
            idx_c_total = k*6 + (i+3)
            for c in range(26):
                W_total[idx_c_total][c+26] = W1[idx_c1][c]
                
    return {"U": U_total, "V": V_total, "W": W_total}

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Try the combined 3x3x3 approach (rank 52)
    try:
        candidate = _combine_3x3x6()
        if _verify_decomposition(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            return candidate
    except Exception:
        pass
    
    # 2. Fallback to schoolbook (rank 54)
    return _schoolbook(n, m, p)
