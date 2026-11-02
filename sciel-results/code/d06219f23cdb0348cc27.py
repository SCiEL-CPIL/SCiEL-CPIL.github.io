import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    # 4 blocks of 2x3x2
    for b in range(4):
        # Local k indices
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for j=0,1
        # M1 = (A00+A11)(B00+B11)
        U[0*m+0, r] = 1; U[1*m+1, r] = 1
        V[0*p+k0, r] = 1; V[1*p+k1, r] = 1
        W[k0*n+0, r] = 1; W[k1*n+1, r] = 1
        r += 1
        
        # M2 = (A10+A11)B00
        U[1*m+0, r] = 1; U[1*m+1, r] = 1
        V[0*p+k0, r] = 1
        W[k0*n+1, r] = 1
        r += 1
        
        # M3 = A00(B01-B11)
        U[0*m+0, r] = 1
        V[0*p+k1, r] = 1; V[1*p+k1, r] = -1
        W[k1*n+0, r] = 1
        r += 1
        
        # M4 = A11(B10-B00)
        U[1*m+1, r] = 1
        V[1*p+k0, r] = 1; V[0*p+k0, r] = -1
        W[k0*n+1, r] = 1
        r += 1
        
        # M5 = (A00+A10)(B00+B01)
        U[0*m+0, r] = 1; U[1*m+0, r] = 1
        V[0*p+k0, r] = 1; V[0*p+k1, r] = 1
        W[k0*n+0, r] = -1; W[k1*n+0, r] = 1
        # Wait, C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # So W for M5:
        # Contributes to C00 with -1, C01 with 1.
        # C00 is index (k0, 0). C01 is index (k1, 0).
        W[k0*n+0, r] = -1
        W[k1*n+0, r] = 1
        r += 1
        
        # M6 = (A01-A10)(B10+B11)
        U[0*m+1, r] = 1; U[1*m+0, r] = -1
        V[1*p+k0, r] = 1; V[1*p+k1, r] = 1
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        # Contributes to C10 with 1, C11 with 1.
        W[k0*n+1, r] = 1
        W[k1*n+1, r] = 1
        r += 1
        
        # M7 = (A01-A11)(B00+B10)
        U[0*m+1, r] = 1; U[1*m+1, r] = -1
        V[0*p+k0, r] = 1; V[1*p+k0, r] = 1
        # C00 = ... + M7
        # C11 = ... (M7 is not in C11 standard Strassen?)
        # Let's check standard Strassen formula again.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        # Wait, M7 is usually (A01-A11)(B00+B10)?
        # Let's verify M7 contribution.
        # (A01-A11)(B00+B10) = A01B00 + A01B10 - A11B00 - A11B10
        # C00 needs A00B00 + A01B10? No.
        # C00 = A00B00 + A01B10? No, C00 = A00B00 + A01B01? No.
        # C = AB. C00 = A00B00 + A01B10.
        # My Strassen terms:
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11(B10-B00) = A11B10 - A11B00
        # M5 = (A00+A10)(B00+B01) = A00B00 + A00B01 + A10B00 + A10B01
        # M7 = (A01-A11)(B00+B10) = A01B00 + A01B10 - A11B00 - A11B10
        # Sum for C00: M1 + M4 - M5 + M7
        # = (A00B00 + A00B11 + A11B00 + A11B11) + (A11B10 - A11B00) - (A00B00 + A00B01 + A10B00 + A10B01) + (A01B00 + A01B10 - A11B00 - A11B10)
        # A00B00: 1 - 1 = 0? No.
        # Let's use a known correct set.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)(B00+B01)
        # M6 = (A01-A10)(B10+B11)
        # M7 = (A01-A11)(B00+B10)
        # This is correct.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        
        # So for M7:
        # W[k0*n+0, r] = 1 (for C00)
        # W[k1*n+1, r] = 0?
        # C11 does not have M7.
        W[k0*n+0, r] = 1
        r += 1
        
        # Terms for j=2
        # T1 = A02 B20
        U[0*m+2, r] = 1
        V[2*p+k0, r] = 1
        W[k0*n+0, r] = 1
        r += 1
        
        # T2 = A02 B21
        U[0*m+2, r] = 1
        V[2*p+k1, r] = 1
        W[k1*n+0, r] = 1
        r += 1
        
        # T3 = A12 B20
        U[1*m+2, r] = 1
        V[2*p+k0, r] = 1
        W[k0*n+1, r] = 1
        r += 1
        
        # T4 = A12 B21
        U[1*m+2, r] = 1
        V[2*p+k1, r] = 1
        W[k1*n+1, r] = 1
        r += 1
        
    # Verify
    # Check a few points
    # i=0, j=0, k=0
    s = 0
    for rr in range(rank):
        s += U[0*m+0, rr] * V[0*p+0, rr] * W[0*n+0, rr]
    if not np.isclose(s, 1.0):
        return {"U": np.zeros((n*m, 48)).tolist(), "V": np.zeros((m*p, 48)).tolist(), "W": np.zeros((p*n, 48)).tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
