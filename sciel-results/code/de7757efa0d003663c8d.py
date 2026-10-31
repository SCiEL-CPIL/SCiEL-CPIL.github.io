"""Rank-49 decomposition of 4x4 matrix multiplication via tensor product of Strassen."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrices (rank 7)
    # We use the standard Strassen formulas for multiplying 2x2 matrices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # For 2x2 matrices, the tensor indices are:
    # A[i,j] where i,j in {0,1}, B[j,k] where j,k in {0,1}, C[i,k] where i,k in {0,1}
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*2+j, t] = coefficient of A[i,j] in term t
    # V[j*2+k, t] = coefficient of B[j,k] in term t
    # W[k*2+i, t] = coefficient of C[i,k] in term t (note: trace convention W[k*n+i])
    
    # Let's define the 7 Strassen terms for 2x2:
    # Term 0: M1 = (a+d)(e+h) -> A: a=1,d=1; B: e=1,h=1; C: top-left +=1, top-right +=0, bottom-left +=0, bottom-right +=1... wait
    # Let me be more careful.
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # M1 = (a+d)(e+h) = a*e + a*h + d*e + d*h
    # M2 = (c+d)e = c*e + d*e
    # M3 = a(f-h) = a*f - a*h
    # M4 = d(g-e) = d*g - d*e
    # M5 = (a+b)h = a*h + b*h
    # M6 = (c-a)(e+f) = c*e + c*f - a*e - a*f
    # M7 = (b-d)(g+h) = b*g + b*h - d*g - d*h
    
    # So for each term t, we need U[.,t], V[.,t], W[.,t]
    # U[i*2+j, t] is the coefficient of A[i,j] in M_t
    # V[j*2+k, t] is the coefficient of B[j,k] in M_t
    # W[k*2+i, t] is the coefficient of C[i,k] in M_t (trace convention: W[k*n+i] where n=2 for 2x2)
    
    # Let's build the 2x2 Strassen factors:
    # U2: shape (4, 7), indices: row = i*2+j, col = term
    # V2: shape (4, 7), indices: row = j*2+k, col = term
    # W2: shape (4, 7), indices: row = k*2+i, col = term
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0: M1 = a*e + a*h + d*e + d*h
    # A: a=A[0,0], d=A[1,1]
    # B: e=B[0,0], h=B[1,1]
    U2[0*2+0][0] = 1  # a
    U2[1*2+1][0] = 1  # d
    V2[0*2+0][0] = 1  # e
    V2[1*2+1][0] = 1  # h
    # C contributions: C[0,0] += 1, C[1,1] += 1
    # W index: k*2+i
    # C[0,0]: i=0,k=0 -> W2[0*2+0] = W2[0] += 1
    # C[1,1]: i=1,k=1 -> W2[1*2+1] = W2[3] += 1
    W2[0][0] = 1
    W2[3][0] = 1
    
    # Term 1: M2 = c*e + d*e
    # A: c=A[1,0], d=A[1,1]
    # B: e=B[0,0]
    U2[1*2+0][1] = 1  # c
    U2[1*2+1][1] = 1  # d
    V2[0*2+0][1] = 1  # e
    # C: C[1,0] += 1
    # C[1,0]: i=1,k=0 -> W2[0*2+1] = W2[1] += 1
    W2[1][1] = 1
    
    # Term 2: M3 = a*f - a*h
    # A: a=A[0,0]
    # B: f=B[0,1], h=B[1,1]
    U2[0*2+0][2] = 1  # a
    V2[0*2+1][2] = 1  # f
    V2[1*2+1][2] = -1 # h
    # C: C[0,1] += 1
    # C[0,1]: i=0,k=1 -> W2[1*2+0] = W2[2] += 1
    W2[2][2] = 1
    
    # Term 3: M4 = d*g - d*e
    # A: d=A[1,1]
    # B: g=B[1,0], e=B[0,0]
    U2[1*2+1][3] = 1  # d
    V2[1*2+0][3] = 1  # g
    V2[0*2+0][3] = -1 # e
    # C: C[0,0] += 1, C[1,0] += 1
    W2[0][3] = 1   # C[0,0]
    W2[1][3] = 1   # C[1,0]
    
    # Term 4: M5 = a*h + b*h
    # A: a=A[0,0], b=A[0,1]
    # B: h=B[1,1]
    U2[0*2+0][4] = 1  # a
    U2[0*2+1][4] = 1  # b
    V2[1*2+1][4] = 1  # h
    # C: C[0,0] -= 1, C[0,1] += 1
    W2[0][4] = -1   # C[0,0]
    W2[2][4] = 1    # C[0,1]
    
    # Term 5: M6 = c*e + c*f - a*e - a*f
    # A: c=A[1,0], a=A[0,0]
    # B: e=B[0,0], f=B[0,1]
    U2[1*2+0][5] = 1  # c
    U2[0*2+0][5] = -1 # a
    V2[0*2+0][5] = 1  # e
    V2[0*2+1][5] = 1  # f
    # C: C[1,1] += 1
    W2[3][5] = 1
    
    # Term 6: M7 = b*g + b*h - d*g - d*h
    # A: b=A[0,1], d=A[1,1]
    # B: g=B[1,0], h=B[1,1]
    U2[0*2+1][6] = 1  # b
    U2[1*2+1][6] = -1 # d
    V2[1*2+0][6] = 1  # g
    V2[1*2+1][6] = 1  # h
    # C: C[0,0] += 1
    W2[0][6] = 1
    
    # Now for 4x4 matrices, we view them as 2x2 blocks of 2x2 matrices.
    # The 4x4 multiplication can be seen as multiplying two 2x2 block matrices
    # where each block is 2x2. But that would give rank 7*7=49 if we could
    # directly tensor product the Strassen decomposition.
    # 
    # Actually, the correct approach: 4x4 matrix multiplication tensor is the
    # tensor product of two 2x2 matrix multiplication tensors. So if we have
    # rank-7 decompositions of the 2x2 tensor, we get rank 7*7=49 for 4x4.
    #
    # For the tensor product: if T1 has factors (U1,V1,W1) with ranks r1 and
    # T2 has factors (U2,V2,W2) with ranks r2, then T1⊗T2 has factors:
    # U = U1 ⊗ U2 (Kronecker product), etc.
    #
    # For 4x4: we index A as A[i1*2+i2, j1*2+j2] where i1,j1 in {0,1} (block)
    # and i2,j2 in {0,1} (within block).
    #
    # U for 4x4: shape (16, 49), U[i1*4+i2*2+j2, t1*7+t2] = U1[i1*2+j1, t1] * U2[i2*2+j2, t2]
    # Wait, let me think about the indexing more carefully.
    #
    # For 2x2: A has indices (i,j), B has (j,k), C has (i,k)
    # U2[i*2+j, t], V2[j*2+k, t], W2[k*2+i, t]
    #
    # For 4x4: A has indices (i1,i2,j1,j2) -> flattened as (i1*2+i2)*2 + (j1*2+j2) = i1*4+i2*2+j1*2+j2
    # Wait, that's not right. Let me use a clearer convention.
    #
    # Let's say A[i,j] where i = i1*2+i2, j = j1*2+j2
    # So flattened index for A: i*4+j = (i1*2+i2)*4 + (j1*2+j2)
    #
    # Similarly for B[j,k]: j = j1*2+j2, k = k1*2+k2
    # Flattened: j*4+k = (j1*2+j2)*4 + (k1*2+k2)
    #
    # For C[i,k]: i = i1*2+i2, k = k1*2+k2
    # Flattened: k*4+i (trace convention) = (k1*2+k2)*4 + (i1*2+i2)
    #
    # The 4x4 multiplication tensor is the tensor product of two 2x2 tensors.
    # Term t = t1*7+t2 where t1 is the "outer" Strassen term and t2 is "inner".
    #
    # U4[i*4+j, t1*7+t2] = U2[i1*2+j1, t1] * U2[i2*2+j2, t2]
    # where i = i1*2+i2, j = j1*2+j2
    #
    # V4[j*4+k, t1*7+t2] = V2[j1*2+k1, t1] * V2[j2*2+k2, t2]
    # where j = j1*2+j2, k = k1*2+k2
    #
    # W4[k*4+i, t1*7+t2] = W2[k1*2+i1, t1] * W2[k2*2+i2, t2]
    # where k = k1*2+k2, i = i1*2+i2
    
    n = 4
    rank = 49
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    for t1 in range(7):
        for t2 in range(7):
            t = t1*7 + t2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2+j1][t1]
                    if u1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2+j2][t2]
                            if u2 == 0:
                                continue
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U[i*4+j][t] += u1 * u2
            
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V2[j1*2+k1][t1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = V2[j2*2+k2][t2]
                            if v2 == 0:
                                continue
                            j = j1*2 + j2
                            k = k1*2 + k2
                            V[j*4+k][t] += v1 * v2
            
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W2[k1*2+i1][t1]
                    if w1 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = W2[k2*2+i2][t2]
                            if w2 == 0:
                                continue
                            k = k1*2 + k2
                            i = i1*2 + i2
                            W[k*4+i][t] += w1 * w2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
