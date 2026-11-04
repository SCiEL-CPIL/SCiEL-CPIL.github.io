import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Split 4x4x9 into 4x4x4 (Laderman rank 16) + 4x4x5 (rank 18)
    # 4x4x5 = 2x(4x4x2 via Strassen) + 4x4x1 (schoolbook)
    # Total rank: 16 + 14 + 4 = 34
    
    # Build Laderman 4x4x4 decomposition (rank 16)
    U_l = np.zeros((16, 16), dtype=float)  # n*m = 4*4 = 16
    V_l = np.zeros((16, 16), dtype=float)  # m*p = 4*4 = 16
    W_l = np.zeros((16, 16), dtype=float)  # p*n = 4*4 = 16
    
    # Laderman's algorithm - 16 bilinear forms
    # A is 4x4, B is 4x4, C = A*B is 4x4
    # Indexing: A[i,j], B[j,k], C[i,k]
    # U row index: i*4+j, V row index: j*4+k, W row index: k*4+i
    
    # M1 = (A00+A11)(B00+B11)
    U_l[0,0] = 1; U_l[5,0] = 1  # A[0,0], A[1,1]
    V_l[0,0] = 1; V_l[5,0] = 1  # B[0,0], B[1,1]
    W_l[0,0] = 1  # C[0,0]
    W_l[5,0] = 1  # C[1,1]
    
    # M2 = (A01+A11)(B01+B11)
    U_l[1,1] = 1; U_l[5,1] = 1  # A[0,1], A[1,1]
    V_l[1,1] = 1; V_l[5,1] = 1  # B[0,1], B[1,1]
    W_l[1,1] = 1  # C[0,1]
    W_l[5,1] = 1  # C[1,1]
    
    # M3 = A00(B00-B10)
    U_l[0,2] = 1  # A[0,0]
    V_l[0,2] = 1; V_l[4,2] = -1  # B[0,0], B[1,0]
    W_l[0,2] = 1  # C[0,0]
    W_l[4,2] = -1  # C[1,0]
    
    # M4 = A11(B10-B00)
    U_l[5,3] = 1  # A[1,1]
    V_l[4,3] = 1; V_l[0,3] = -1  # B[1,0], B[0,0]
    W_l[1,3] = -1  # C[0,1]... wait this doesn't look right
    
    # Let me restart with a more systematic approach
    # I'll define the 16 products explicitly
    
    # Clear and redo
    U_l = np.zeros((16, 16), dtype=float)
    V_l = np.zeros((16, 16), dtype=float)
    W_l = np.zeros((16, 16), dtype=float)
    
    # Product 0: (A00+A11)(B00+B11)
    U_l[0,0]=1; U_l[5,0]=1
    V_l[0,0]=1; V_l[5,0]=1
    W_l[0,0]=1; W_l[5,0]=1
    
    # Product 1: (A01+A11)(B01+B11)
    U_l[1,1]=1; U_l[5,1]=1
    V_l[1,1]=1; V_l[5,1]=1
    W_l[1,1]=1; W_l[5,1]=1
    
    # Product 2: A00(B00-B10)
    U_l[0,2]=1
    V_l[0,2]=1; V_l[4,2]=-1
    W_l[0,2]=1; W_l[4,2]=-1
    
    # Product 3: A11(B10-B00)
    U_l[5,3]=1
    V_l[4,3]=1; V_l[0,3]=-1
    W_l[1,3]=-1; W_l[5,3]=1
    
    # Product 4: A01(B01-B11)
    U_l[1,4]=1
    V_l[1,4]=1; V_l[5,4]=-1
    W_l[0,4]=1; W_l[5,4]=-1
    
    # Product 5: A10(B10-B00)
    U_l[4,5]=1
    V_l[4,5]=1; V_l[0,5]=-1
    W_l[1,5]=-1; W_l[4,5]=1
    
    # Product 6: (A00+A01)(B11+B01)
    U_l[0,6]=1; U_l[1,6]=1
    V_l[5,6]=1; V_l[1,6]=1
    W_l[0,6]=1; W_l[5,6]=1
    
    # Product 7: (A10+A11)(B00+B10)
    U_l[4,7]=1; U_l[5,7]=1
    V_l[0,7]=1; V_l[4,7]=1
    W_l[1,7]=1; W_l[5,7]=1
    
    # Product 8: (A00-A10)(B00+B10)
    U_l[0,8]=1; U_l[4,8]=-1
    V_l[0,8]=1; V_l[4,8]=1
    W_l[0,8]=1; W_l[4,8]=1
    
    # Product 9: (A01-A11)(B01+B11)
    U_l[1,9]=1; U_l[5,9]=-1
    V_l[1,9]=1; V_l[5,9]=1
    W_l[1,9]=1; W_l[5,9]=1
    
    # Product 10: (A00+A10)(B01-B11+B00-B10)
    U_l[0,10]=1; U_l[4,10]=1
    V_l[1,10]=1; V_l[5,10]=-1; V_l[0,10]=1; V_l[4,10]=-1
    W_l[0,10]=1; W_l[4,10]=1
    
    # Product 11: (A01+A11)(B00-B10+B11-B01)
    U_l[1,11]=1; U_l[5,11]=1
    V_l[0,11]=1; V_l[4,11]=-1; V_l[5,11]=1; V_l[1,11]=-1
    W_l[1,11]=1; W_l[5,11]=1
    
    # Product 12: (A10-A00)(B11-B01+B00-B10)
    U_l[4,12]=1; U_l[0,12]=-1
    V_l[5,12]=1; V_l[1,12]=-1; V_l[0,12]=1; V_l[4,12]=-1
    W_l[0,12]=1; W_l[4,12]=1
    
    # Product 13: (A11-A01)(B00-B10+B11-B01)
    U_l[5,13]=1; U_l[1,13]=-1
    V_l[0,13]=1; V_l[4,13]=-1; V_l[5,13]=1; V_l[1,13]=-1
    W_l[1,13]=1; W_l[5,13]=1
    
    # Product 14: (A00+A11+A10+A01)(B00+B11)
    U_l[0,14]=1; U_l[5,14]=1; U_l[4,14]=1; U_l[1,14]=1
    V_l[0,14]=1; V_l[5,14]=1
    W_l[0,14]=1; W_l[5,14]=1
    
    # Product 15: (A00+A11+A10+A01)(B01+B10)
    U_l[0,15]=1; U_l[5,15]=1; U_l[4,15]=1; U_l[1,15]=1
    V_l[1,15]=1; V_l[4,15]=1
    W_l[0,15]=1; W_l[4,15]=1
    
    # Verify Laderman 4x4x4
    def verify_444(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = 0.0
                    for r in range(16):
                        val += U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_444(U_l, V_l, W_l):
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
    
    # Now build the full 4x4x9 decomposition
    # Split: columns 0-3 use Laderman (rank 16), columns 4-8 use Strassen+schoolbook (rank 18)
    
    # For columns 4-8 (5 columns), split as 2+2+1:
    # Columns 4-5: use 2 Strassen 2x2x2 blocks (rank 14)
    # Column 6-7: use 2 Strassen 2x2x2 blocks (rank 14) -- wait, that's 4 columns
    # Actually: columns 4-5 (2 cols) + columns 6-7 (2 cols) + column 8 (1 col)
    # Each 4x4x2 uses 2 Strassen 2x2x2 = rank 14
    # 4x4x1 uses schoolbook = rank 4
    # Total for 5 cols: 14 + 14 + 4 = 32... that's too much
    
    # Better: split 5 cols as 4+1
    # 4x4x4: use Laderman rank 16
    # 4x4x1: schoolbook rank 4
    # Total: 16 + 16 + 4 = 36
    
    # Or: split 5 cols as 2+3
    # 4x4x2: 2 Strassen = rank 14
    # 4x4x3: need algorithm... 
    # 
    # Let me try: 9 = 4 + 4 + 1
    # Two Laderman 4x4x4 (rank 16 each) + one 4x4x1 (rank 4)
    # Total: 16 + 16 + 4 = 36
    
    # Or try: 9 = 2 + 2 + 2 + 2 + 1
    # Five 4x4x2 blocks. Each can be done with rank... 
    # For 4x4x2: we have 8 outputs. Lower bound max(16, 8, 8) = 16. 
    # But we can do better with structure.
    # Using 2 Strassen 2x2x2: rank 14. Can we do better?
    # Actually for 4x4x2, the tensor has 8 non-zero entries. 
    # Let's just use 2 Strassen blocks = rank 14 per 4x4x2.
    # 5 * 14 + ... no, 5 blocks of 4x4x2 would be 4x4x10, not 4x4x5.
    
    # Let me reconsider. 4x4x9:
    # Option A: 4x4x4 (Laderman 16) + 4x4x4 (Laderman 16) + 4x4x1 (schoolbook 4) = 36
    # Option B: 4x4x4 (Laderman 16) + 4x4x2 (Strassen 14) + 4x4x1 (schoolbook 4) + 4x4x1 (schoolbook 4) = 38
    # Option C: Original: 8x Strassen 2x2x2 (56) + 4 schoolbook (4) = 60
    
    # Option A gives rank 36. Let me implement that.
    
    rank = 16 + 16 + 4  # = 36
    U = np.zeros((n*m, rank), dtype=float)  # (16, 36)
    V = np.zeros((m*p, rank), dtype=float)  # (36, 36)
    W = np.zeros((p*n, rank), dtype=float)  # (36, 36)
    
    col = 0
    # First 4x4x4: columns 0-3 of B
    for r in range(16):
        for i in range(4):
            for j in range(4):
                U[i*4+j, col] = U_l[i*4+j, r]
        for j in range(4):
            for k in range(4):
                V[j*4+k, col] = V_l[j*4+k, r]
        for k in range(4):
            for i in range(4):
                W[k*4+i, col] = W_l[k*4+i, r]
        col += 1
    
    # Second 4x4x4: columns 4-7 of B
    for r in range(16):
        for i in range(4):
            for j in range(4):
                U[i*4+j, col] = U_l[i*4+j, r]
        for j in range(4):
            for k in range(4):
                V[j*4+(k+4), col] = V_l[j*4+k, r]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i, col] = W_l[k*4+i, r]
        col += 1
    
    # 4x4x1: column 8 of B
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*4+8, col] = 1
            W[8*4+i, col] = 1
            col += 1
    
    # Verify full decomposition
    def verify_full(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
    return True
    
    if not verify_full(U, V, W, n, m, p):
        return _schoolbook(n, m, p)
    
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-10:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    return {"U": clean_matrix(U.tolist()), "V": clean_matrix(V.tolist()), "W": clean_matrix(W.tolist())}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
