import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Trivial decomposition fallback (Rank 36)
    def trivial_decomposition():
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m+jj][c] = 1
                    V_t[jj*p+kk][c] = 1
                    W_t[kk*n+ii][c] = 1
                    c += 1
        return {"U": U_t, "V": V_t, "W": W_t}

    # Target Rank: 32
    # Split p=9 into 4 pairs (Strassen) and 1 single (Schoolbook)
    # Pairs: (0,1), (2,3), (4,5), (6,7) -> 4 * 7 = 28 terms
    # Single: 8 -> 4 terms
    # Total Rank = 32
    
    terms = []
    
    # Helper to add a Strassen block for columns k0, k1
    def add_strassen(k0, k1):
        # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
        # B indices: 
        #   row 0: k0, k1
        #   row 1: p+k0, p+k1
        # C indices:
        #   col k0: 2*k0, 2*k0+1
        #   col k1: 2*k1, 2*k1+1
        
        # M1 = (a+d)(e+h) -> C[0,k0] +1, C[1,k1] +1
        # a=A00, d=A11, e=B0k0, h=B1k1
        # V indices: e is k0, h is p+k1
        # W indices: C0k0 is 2*k0, C1k1 is 2*k1+1
        terms.append(({0:1, 3:1}, {k0:1, p+k1:1}, {2*k0:1, 2*k1+1:1}))
        
        # M2 = (c+d)e -> C[1,k0] +1, C[1,k1] -1
        # c=A10, d=A11, e=B0k0
        terms.append(({2:1, 3:1}, {k0:1}, {2*k0+1:1, 2*k1+1:-1}))
        
        # M3 = a(f-h) -> C[0,k1] +1, C[1,k1] +1
        # a=A00, f=B0k1, h=B1k1
        terms.append(({0:1}, {k1:1, p+k1:-1}, {2*k1:1, 2*k1+1:1}))
        
        # M4 = d(g-e) -> C[0,k0] +1, C[1,k0] +1
        # d=A11, g=B1k0, e=B0k0
        terms.append(({3:1}, {p+k0:1, k0:-1}, {2*k0:1, 2*k0+1:1}))
        
        # M5 = (a+b)h -> C[0,k0] -1, C[0,k1] +1
        # a=A00, b=A01, h=B1k1
        terms.append(({0:1, 1:1}, {p+k1:1}, {2*k0:-1, 2*k1:1}))
        
        # M6 = (c-a)(e+f) -> C[1,k1] +1
        # c=A10, a=A00, e=B0k0, f=B0k1
        terms.append(({2:1, 0:-1}, {k0:1, k1:1}, {2*k1+1:1}))
        
        # M7 = (b-d)(g+h) -> C[0,k0] +1
        # b=A01, d=A11, g=B1k0, h=B1k1
        terms.append(({1:1, 3:-1}, {p+k0:1, p+k1:1}, {2*k0:1}))

    # Helper to add a Schoolbook block for column k
    def add_schoolbook(k):
        # S1 = A[0,0]*B[0,k] -> C[0,k]
        terms.append(({0:1}, {k:1}, {2*k:1}))
        # S2 = A[0,1]*B[1,k] -> C[0,k]
        terms.append(({1:1}, {p+k:1}, {2*k:1}))
        # S3 = A[1,0]*B[0,k] -> C[1,k]
        terms.append(({2:1}, {k:1}, {2*k+1:1}))
        # S4 = A[1,1]*B[1,k] -> C[1,k]
        terms.append(({3:1}, {p+k:1}, {2*k+1:1}))

    # Build the decomposition
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    
    rank = len(terms)
    
    if time.time() > deadline:
        return trivial_decomposition()

    # Construct matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val

    # Full Verification
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should be 1 for all i,j,k
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        if time.time() > deadline:
            return trivial_decomposition()
        # If verification fails, return trivial to be safe, though it shouldn't happen
        return trivial_decomposition()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
