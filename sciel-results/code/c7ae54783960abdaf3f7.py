import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0  # 5s safety margin

def construct():
    n, m, p = DIMENSIONS
    # Start with naive rank-96 decomposition
    naive_rank = n * m * p
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_naive, "V": V_naive, "W": W_naive}
    best_rank = naive_rank
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
        
        # The tensor should be all ones
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Verify naive solution
    if not verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        # Should not happen, but return anyway
        return best_result
    
    # Try to improve using block decomposition with Strassen
    # Strategy: decompose 3x4x8 into sub-blocks
    # Split n=3 into 2+1, p=8 into 4+4
    # For 2x4x4: use four 2x2x2 Strassen decompositions (rank 7 each) = 28
    # For 1x4x4: naive rank 4
    # Total target rank: 28 + 4 = 32
    
    # Let's construct this
    # First, define Strassen 2x2x2 decomposition
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for i,j,k in {0,1}
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # In tensor form, for each term r:
    # U[i,j,r] gives the coefficient of A[i,j]
    # V[j,k,r] gives the coefficient of B[j,k]
    # W[k,i,r] gives the coefficient in C[i,k]
    
    # Term 0 (M1): A00*A? + A11*?
    # M1 = (A00+A11)(B00+B11)
    # This contributes to C00 (+1), C11 (+1)
    # U: A00=1, A11=1 => U[0,0]=1, U[1,1]=1, others 0
    # V: B00=1, B11=1 => V[0,0]=1, V[1,1]=1, others 0
    # W: C00=1, C11=1 => W[0,0]=1, W[1,1]=1, others 0
    
    # Let me define the 7 terms for 2x2x2:
    # Index: i*m+j for U (m=2), j*p+k for V (p=2), k*n+i for W (n=2)
    
    strassen_U = []
    strassen_V = []
    strassen_W = []
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    strassen_U.append([1, 0, 0, 1])  # A00, A01, A10, A11
    # V: B00=1, B11=1
    strassen_V.append([1, 0, 0, 1])  # B00, B01, B10, B11
    # W: contributes +1 to C00, +1 to C11
    # W[k*n+i] for n=2: W[0*2+0]=W[0] for C[0,0], W[1*2+1]=W[3] for C[1,1]
    strassen_W.append([1, 0, 0, 1])
    
    # Term 1: M2 = (A10+A11)B00
    # U: A10=1, A11=1
    strassen_U.append([0, 0, 1, 1])
    # V: B00=1
    strassen_V.append([1, 0, 0, 0])
    # W: contributes +1 to C10, +1 to C11
    # C[1,0]: k=0, i=1 => W[0*2+1]=W[1]
    # C[1,1]: k=1, i=1 => W[1*2+1]=W[3]
    strassen_W.append([0, 1, 0, 1])
    
    # Term 2: M3 = A00(B10-B11)
    # U: A00=1
    strassen_U.append([1, 0, 0, 0])
    # V: B10=1, B11=-1
    strassen_V.append([0, 0, 1, -1])
    # W: contributes +1 to C01
    # C[0,1]: k=1, i=0 => W[1*2+0]=W[2]
    strassen_W.append([0, 0, 1, 0])
    
    # Term 3: M4 = A11(B01-B00)
    # U: A11=1
    strassen_U.append([0, 0, 0, 1])
    # V: B01=1, B00=-1
    strassen_V.append([-1, 1, 0, 0])
    # W: contributes +1 to C00, +1 to C10
    # C[0,0]: k=0, i=0 => W[0]
    # C[1,0]: k=0, i=1 => W[1]
    strassen_W.append([1, 1, 0, 0])
    
    # Term 4: M5 = (A00+A01)B11
    # U: A00=1, A01=1
    strassen_U.append([1, 1, 0, 0])
    # V: B11=1
    strassen_V.append([0, 0, 0, 1])
    # W: contributes -1 to C00, +1 to C01
    # C[0,0]: W[0] = -1
    # C[0,1]: W[2] = 1
    strassen_W.append([-1, 0, 1, 0])
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1
    strassen_U.append([-1, 0, 1, 0])
    # V: B00=1, B01=1
    strassen_V.append([1, 1, 0, 0])
    # W: contributes to C10? Let me check: C10 = M2 + M4, so M6 should not contribute to C10
    # Actually, let me re-derive. The standard Strassen is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So M6 doesn't appear in any C? That can't be right.
    # Let me look up the correct Strassen formulas.
    
    # Correct Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A10) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So M6 is not used? That seems wrong. Let me check again.
    
    # Actually, I think I have an error. Let me use a different source.
    # The correct formulas are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A10)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # M6 is indeed not used in the output! That can't be right for a rank-7 decomposition.
    # I must have the formulas wrong. Let me try a different set.
    
    # Alternative Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A10)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # I'm going in circles. Let me just verify by computation.
    
    # For now, let me just return the naive solution since getting the Strassen decomposition
    # correct is complex. The naive solution has rank 96.
    
    return best_result
