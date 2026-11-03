import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 5
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def validate(U, V, W):
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False

    # Strategy: Use Strassen (rank 7) for the 2x2x2 sub-block (i,j,k in {0,1})
    # and schoolbook for all other terms.
    # 
    # Total terms:
    # - Strassen for i,j,k in {0,1}: 7 terms
    # - Schoolbook for all other (i,j,k) where at least one is 2:
    #   Total schoolbook = 3*3*5 = 45
    #   Terms in 2x2x2 block = 2*2*2 = 8
    #   Other terms = 45 - 8 = 37
    # Total rank = 7 + 37 = 44
    
    # Let's verify: which terms are NOT in the 2x2x2 block?
    # Terms where i=2 OR j=2 OR k=2 (for k in {0,1,2}) or k in {3,4}
    # Actually, k goes 0..4, so k=3,4 are always "other"
    # For k=0,1,2: terms where i=2 or j=2
    # For k=3,4: all terms (i in 0..2, j in 0..2)
    
    # Let me count:
    # k=0,1,2: 
    #   i=2, j=0,1,2: 3 terms per k -> 9
    #   i=0,1, j=2: 2 terms per k -> 6
    #   Total for k=0,1,2 where i=2 or j=2: 15
    # k=3,4:
    #   all i,j: 9 terms per k -> 18
    # Total non-2x2x2 terms: 15 + 18 = 33
    # Plus Strassen: 7
    # Total: 40
    
    # Wait, I need to be more careful. The 2x2x2 block is i in {0,1}, j in {0,1}, k in {0,1,2}?
    # No, k only goes 0,1 for the 2x2x2 block since we're doing 3x3x3 subproblem.
    # But here p=5, so k goes 0..4.
    # The 2x2x2 sub-block would be i in {0,1}, j in {0,1}, k in {0,1} (only 2 columns of B)
    # That's 2*2*2 = 8 schoolbook terms replaced by 7 Strassen terms.
    
    # Let me reconsider. Use Strassen for i,j in {0,1} and k in {0,1}:
    # 8 schoolbook terms -> 7 Strassen terms, saving 1.
    # Total: 45 - 8 + 7 = 44.
    
    # Alternatively, can we do better? What about using Strassen for k=0,1,2?
    # The 2x2x3 subproblem (i,j in {0,1}, k in {0,1,2}) has rank?
    # 2x2x3 matrix multiplication: this is not standard.
    
    # Let's just go with the simple approach: Strassen for 2x2x2 (k=0,1) + schoolbook for rest.
    # Rank = 44.
    
    # Actually, let me try a different approach. What if I use Strassen for the 2x2 block
    # across all 5 columns of B? No, that doesn't make sense because Strassen is for
    # 2x2 x 2x2 -> 2x2.
    
    # Let me just implement the 44-rank solution and verify it.
    
    U = np.zeros((9, 44))
    V = np.zeros((15, 44))
    W = np.zeros((15, 44))
    col = 0
    
    # Strassen terms for i,j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # In our indexing: A indices are i*3+j, B indices are j*5+k, C indices are k*3+i
    # A00 = A[0,0] = index 0, A11 = A[1,1] = index 4
    # B00 = B[0,0] = index 0, B11 = B[1,1] = index 6
    # C00 = C[0,0] = index 0, C11 = C[1,1] = index 4
    
    # M1: U[0,c]=1, U[4,c]=1; V[0,c]=1, V[6,c]=1; W[0,c]=1, W[4,c]=1
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[6, col] = 1
    W[0, col] = 1; W[4, col] = 1
    col += 1
    
    # M2 = (A10+A11)(B00+B10) -> C10 += M2, C11 += M2
    # A10 = A[1,0] = index 3, A11 = A[1,1] = index 4
    # B00 = B[0,0] = index 0, B10 = B[1,0] = index 5
    # C10 = C[0,1] = index 1, C11 = C[1,1] = index 4
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[5, col] = 1
    W[1, col] = 1; W[4, col] = 1
    col += 1
    
    # M3 = (A00+A01)(B11-B01) -> C01 += M3, C00 += M3
    # A00 = index 0, A01 = A[0,1] = index 1
    # B11 = index 6, B01 = B[0,1] = index 1
    # C01 = C[1,0] = index 3, C00 = C[0,0] = index 0
    U[0, col] = 1; U[1, col] = 1
    V[6, col] = 1; V[1, col] = -1
    W[3, col] = 1; W[0, col] = 1
    col += 1
    
    # M4 = A00(B10-B11) -> C00 += M4, C01 += M4
    # A00 = index 0
    # B10 = index 5, B11 = index 6
    # C00 = index 0, C01 = index 3
    U[0, col] = 1
    V[5, col] = 1; V[6, col] = -1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    
    # M5 = A11(B01-B00) -> C11 += M5, C10 += M5
    # A11 = index 4
    # B01 = index 1, B00 = index 0
    # C11 = index 4, C10 = index 1
    U[4, col] = 1
    V[1, col] = 1; V[0, col] = -1
    W[4, col] = 1; W[1, col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B10) -> C10 += M6, C00 += M6
    # A10 = index 3, A00 = index 0
    # B00 = index 0, B10 = index 5
    # C10 = index 1, C00 = index 0
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[5, col] = 1
    W[1, col] = 1; W[0, col] = 1
    col += 1
    
    # M7 = (A01-A11)(B01+B11) -> C01 += M7, C11 += M7
    # A01 = index 1, A11 = index 4
    # B01 = index 1, B11 = index 6
    # C01 = index 3, C11 = index 4
    U[1, col] = 1; U[4, col] = -1
    V[1, col] = 1; V[6, col] = 1
    W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # Now schoolbook for all other terms
    # Terms where NOT (i in {0,1} AND j in {0,1} AND k in {0,1})
    for i in range(3):
        for j in range(3):
            for k in range(5):
                if i in [0,1] and j in [0,1] and k in [0,1]:
                    continue  # Already handled by Strassen
                U[i*3+j, col] = 1
                V[j*5+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    
    # Verify
    if not validate(U, V, W):
        # Fallback to schoolbook
        U_sb = np.zeros((9, 45))
        V_sb = np.zeros((15, 45))
        W_sb = np.zeros((15, 45))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    U_sb[i*3+j, col] = 1
                    V_sb[j*5+k, col] = 1
                    W_sb[k*3+i, col] = 1
                    col += 1
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
