import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition
    rank = 34
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def set_term(u_idx, v_idx, w_idx, val=1):
        # u_idx, v_idx, w_idx can be lists
        if not isinstance(u_idx, list): u_idx = [u_idx]
        if not isinstance(v_idx, list): v_idx = [v_idx]
        if not isinstance(w_idx, list): w_idx = [w_idx]
        for u in u_idx:
            for v in v_idx:
                for w in w_idx:
                    # This logic is wrong for Strassen where U and V are sums.
                    # We need to define columns directly.
                    pass

    # We will define columns explicitly.
    # Each column c corresponds to a term.
    # U[:, c], V[:, c], W[:, c]
    
    # Helper to add a column
    def add_col(u_entries, v_entries, w_entries):
        # u_entries: list of (index, value)
        # v_entries: list of (index, value)
        # w_entries: list of (index, value)
        pass

    # Let's just fill the arrays directly for clarity.
    
    # Block 1: B cols 0-2
    # Strassen 1-7
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1 # A00, A11
    V[0, 0] = 1; V[7, 0] = 1 # B00, B11 (B is 3x6, idx j*6+k. B00=0*6+0=0, B11=1*6+1=7)
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # M2: (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1 # A10, A11 (A10=1*3+0=3, A11=1*3+1=4? No. A is 2x3. A[i,j]. A00=0, A01=1, A02=2, A10=3, A11=4, A12=5.)
    # Correction on A indices:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # A_left is A[:, 0:2] -> indices 0,1 and 3,4.
    # A00=0, A01=1, A10=3, A11=4.
    # M2: (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1 # C10 (idx 1*2+0=2? No. W[k*n+i]. C10 is i=1, k=0. W[0*2+1]=W[1]. Correct.)
    # Wait, C11 is M1 - M2 + ...
    # M2 contributes to C21 (bottom-left) and C22 (bottom-right).
    # In 2x2 block: C21 is i=1, k=0. C22 is i=1, k=1.
    # W indices: C10 -> W[1], C11 -> W[3].
    # M2 contributes to C10 and C11?
    # Strassen: C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So M2 contributes to W[1] and W[3] with sign -1 for W[3].
    # My previous M1 mapping:
    # M1: C11 (top-left, i=0,k=0 -> W[0]) and C22 (bottom-right, i=1,k=1 -> W[3]).
    # M2: C21 (bottom-left, i=1,k=0 -> W[1]) and C22 (bottom-right, i=1,k=1 -> W[3]) with -1.
    
    # Let's restart the Strassen mapping carefully.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1
    # C11 += M1
    # M2 = (A10+A11)B00
    # C10 += M2
    # C11 -= M2
    # M3 = A00(B01-B11)
    # C01 += M3
    # C11 += M3
    # M4 = A11(B10-B00)
    # C00 += M4
    # C10 += M4
    # M5 = (A00+A01)B11
    # C00 -= M5
    # C01 += M5
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7
    
    # Indices:
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=6, B11=7 (Block 1, k=0,1)
    # C00=W[0], C01=W[2], C10=W[1], C11=W[3]
    
    # Col 0: M1
    U[0,0]=1; U[4,0]=1
    V[0,0]=1; V[7,0]=1
    W[0,0]=1; W[3,0]=1
    
    # Col 1: M2
    U[3,1]=1; U[4,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    
    # Col 2: M3
    U[0,2]=1
    V[1,2]=1; V[7,2]=-1
    W[2,2]=1; W[3,2]=1
    
    # Col 3: M4
    U[4,3]=1
    V[6,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1
    
    # Col 4: M5
    U[0,4]=1; U[1,4]=1
    V[7,4]=1
    W[0,4]=-1; W[2,4]=1
    
    # Col 5: M6
    U[3,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # Col 6: M7
    U[1,6]=1; U[4,6]=-1
    V[6,6]=1; V[7,6]=1
    W[0,6]=1

    # A_left * b1 (Cols 7-10)
    # b1 is B[:, 2] for rows 0,1. B[0,2]=2, B[1,2]=8.
    # C02 (W[4]) and C12 (W[5])
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Col 7: A00*B02
    U[0,7]=1
    V[2,7]=1
    W[4,7]=1
    
    # Col 8: A01*B12
    U[1,8]=1
    V[8,8]=1
    W[4,8]=1
    
    # Col 9: A10*B02
    U[3,9]=1
    V[2,9]=1
    W[5,9]=1
    
    # Col 10: A11*B12
    U[4,10]=1
    V[8,10]=1
    W[5,10]=1

    # a * bT (Cols 11-16)
    # a is A[:, 2]. A[0,2]=2, A[1,2]=5.
    # bT is B[2, 0:3]. B[2,0]=12, B[2,1]=13, B[2,2]=14.
    # C00(W[0]), C01(W[2]), C02(W[4])
    # C10(W[1]), C11(W[3]), C12(W[5])
    
    # Col 11: A02*B20
    U[2,11]=1
    V[12,11]=1
    W[0,11]=1
    
    # Col 12: A02*B21
    U[2,12]=1
    V[13,12]=1
    W[2,12]=1
    
    # Col 13: A02*B22
    U[2,13]=1
    V[14,13]=1
    W[4,13]=1
    
    # Col 14: A12*B20
    U[5,14]=1
    V[12,14]=1
    W[1,14]=1
    
    # Col 15: A12*B21
    U[5,15]=1
    V[13,15]=1
    W[3,15]=1
    
    # Col 16: A12*B22
    U[5,16]=1
    V[14,16]=1
    W[5,16]=1

    # Block 2: B cols 3-5
    # Strassen 1-7 (Cols 17-23)
    # B indices: B[0,3]=3, B[0,4]=4, B[1,3]=9, B[1,4]=10.
    # W indices: C03=W[6], C04=W[8], C13=W[7], C14=W[9].
    
    # Col 17: M1
    U[0,17]=1; U[4,17]=1
    V[3,17]=1; V[10,17]=1
    W[6,17]=1; W[9,17]=1
    
    # Col 18: M2
    U[3,18]=1; U[4,18]=1
    V[3,18]=1
    W[7,18]=1; W[9,18]=-1
    
    # Col 19: M3
    U[0,19]=1
    V[4,19]=1; V[10,19]=-1
    W[8,19]=1; W[9,19]=1
    
    # Col 20: M4
    U[4,20]=1
    V[9,20]=1; V[3,20]=-1
    W[6,20]=1; W[7,20]=1
    
    # Col 21: M5
    U[0,21]=1; U[1,21]=1
    V[10,21]=1
    W[6,21]=-1; W[8,21]=1
    
    # Col 22: M6
    U[3,22]=1; U[0,22]=-1
    V[3,22]=1; V[4,22]=1
    W[9,22]=1
    
    # Col 23: M7
    U[1,23]=1; U[4,23]=-1
    V[9,23]=1; V[10,23]=1
    W[6,23]=1

    # A_left * b1 (Cols 24-27)
    # b1 is B[:, 5] for rows 0,1. B[0,5]=5, B[1,5]=11.
    # C05(W[10]), C15(W[11])
    
    # Col 24: A00*B05
    U[0,24]=1
    V[5,24]=1
    W[10,24]=1
    
    # Col 25: A01*B15
    U[1,25]=1
    V[11,25]=1
    W[10,25]=1
    
    # Col 26: A10*B05
    U[3,26]=1
    V[5,26]=1
    W[11,26]=1
    
    # Col 27: A11*B15
    U[4,27]=1
    V[11,27]=1
    W[11,27]=1

    # a * bT (Cols 28-33)
    # bT is B[2, 3:6]. B[2,3]=15, B[2,4]=16, B[2,5]=17.
    # C03(W[6]), C04(W[8]), C05(W[10])
    # C13(W[7]), C14(W[9]), C15(W[11])
    
    # Col 28: A02*B23
    U[2,28]=1
    V[15,28]=1
    W[6,28]=1
    
    # Col 29: A02*B24
    U[2,29]=1
    V[16,29]=1
    W[8,29]=1
    
    # Col 30: A02*B25
    U[2,30]=1
    V[17,30]=1
    W[10,30]=1
    
    # Col 31: A12*B23
    U[5,31]=1
    V[15,31]=1
    W[7,31]=1
    
    # Col 32: A12*B24
    U[5,32]=1
    V[16,32]=1
    W[9,32]=1
    
    # Col 33: A12*B25
    U[5,33]=1
    V[17,33]=1
    W[11,33]=1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
