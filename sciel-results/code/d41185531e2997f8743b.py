"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a rank-28 decomposition for 2x2 * 2x8 matrix multiplication.
    
    We use a Strassen-like block decomposition. The 2x2 matrix A is split into 2x1 blocks
    (since m=2, we can split along the m dimension). The 2x8 matrix B is split into 2x4 blocks.
    
    Actually, let's think more carefully. We want to multiply A (2x2) by B (2x8).
    Result C (2x8).
    
    C[0,0:4] = A[0,0]*B[0,0:4] + A[0,1]*B[1,0:4]
    C[0,4:8] = A[0,0]*B[0,4:8] + A[0,1]*B[1,4:8]
    C[1,0:4] = A[1,0]*B[0,0:4] + A[1,1]*B[1,0:4]
    C[1,4:8] = A[1,0]*B[0,4:8] + A[1,1]*B[1,4:8]
    
    We can use Strassen on the 2x2 block structure if we view this as two independent
    2x2 multiplications (for the left half and right half of B). But that would give
    2 * 7 = 14 multiplications... wait, no.
    
    Let me reconsider. The tensor is for multiplying (2x2) by (2x8).
    The schoolbook rank is 2*2*8 = 32.
    
    A known result: the tensor rank of matrix multiplication <n,m,p> is at most
    the rank of <n,m,p//2> + <n,m,p//2> if we can split p. But that's just schoolbook.
    
    For 2x2x8, we can think of it as two copies of 2x2x4.
    The rank of 2x2x4 is known to be at most 10 (actually it's 10 or 11).
    So 2*10 = 20 would be ideal, but we need to check if we can combine.
    
    Actually, let's use a simpler approach. We can use the fact that for 2x2x2,
    Strassen gives rank 7. For 2x2x8, we can view it as 4 copies of 2x2x2,
    giving 4*7 = 28.
    
    Let me verify this. If we split the 8 columns into 4 groups of 2:
    B = [B1, B2, B3, B4] where each Bi is 2x2.
    C = [C1, C2, C3, C4] where each Ci is 2x2.
    Ci = A * Bi for i=1,2,3,4.
    
    Each Ci = A * Bi is a 2x2x2 multiplication, which has rank 7 (Strassen).
    Since the four multiplications are independent (they share A but B is different),
    we can use 4*7 = 28 rank-1 terms.
    
    This should give rank 28, which is better than 32.
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen decomposition for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m3-m5-m7, m5+m7], [m2+m4, m1-m2+m4]]
    
    # For 2x2x8, we have 4 independent 2x2x2 multiplications.
    # A is shared: A = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B has 8 columns, split into 4 groups of 2.
    
    # U has indices i*m+j where i in [0,1], j in [0,1]
    # U[0] = a, U[1] = b, U[2] = c, U[3] = d
    
    # V has indices j*p+k where j in [0,1], k in [0,7]
    # For each group g in [0,3], the 2x2 block B_g has entries:
    # B_g[0,0] = B[0, 2g] = V[0*8 + 2g] = V[2g]
    # B_g[0,1] = B[0, 2g+1] = V[2g+1]
    # B_g[1,0] = B[1, 2g] = V[1*8 + 2g] = V[8+2g]
    # B_g[1,1] = B[1, 2g+1] = V[8+2g+1] = V[9+2g]
    
    # W has indices k*n+i where k in [0,7], i in [0,1]
    # C[0, 2g] = W[2g*2 + 0] = W[4g]
    # C[0, 2g+1] = W[2g*2 + 1] = W[4g+1]
    # C[1, 2g] = W[2g*2 + 2] = W[4g+2]
    # C[1, 2g+1] = W[2g*2 + 3] = W[4g+3]
    
    # Strassen terms for each group g:
    # m1 = (a+d)(e+h)  -> U: a=1,d=1; V: e=1,h=1
    # m2 = (c+d)e      -> U: c=1,d=1; V: e=1
    # m3 = a(f-h)      -> U: a=1; V: f=1,h=-1
    # m4 = d(g-e)      -> U: d=1; V: g=1,e=-1
    # m5 = (a+b)h      -> U: a=1,b=1; V: h=1
    # m6 = (c-a)(e+f)  -> U: c=1,a=-1; V: e=1,f=1
    # m7 = (b-d)(g+h)  -> U: b=1,d=-1; V: g=1,h=1
    
    # C[0,0] = m1+m3-m5-m7
    # C[0,1] = m5+m7
    # C[1,0] = m2+m4
    # C[1,1] = m1-m2+m4
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each is a dict mapping index to coefficient
        # m1
        ({0: 1, 3: 1}, {0: 1, 1: 1}, {0: 1, 3: 1}),
        # m2
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # m3
        ({0: 1}, {1: 1, 1: -1}, {0: 1}),  # Wait, this is wrong. f and h are different indices.
    ]
    
    # Let me redo this more carefully.
    # For group g, the B indices are:
    # e = B[0, 2g] -> V index 2g
    # f = B[0, 2g+1] -> V index 2g+1
    # g_ = B[1, 2g] -> V index 8+2g
    # h = B[1, 2g+1] -> V index 9+2g
    
    # The C indices are:
    # C[0, 2g] -> W index 4g
    # C[0, 2g+1] -> W index 4g+1
    # C[1, 2g] -> W index 4g+2
    # C[1, 2g+1] -> W index 4g+3
    
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g_-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g_+h)
    
    # C[0,0] = m1+m3-m5-m7
    # C[0,1] = m5+m7
    # C[1,0] = m2+m4
    # C[1,1] = m1-m2+m4
    
    for g in range(4):
        base = g * 7
        e_idx = 2 * g
        f_idx = 2 * g + 1
        g_idx = 8 + 2 * g
        h_idx = 9 + 2 * g
        
        c00_idx = 4 * g
        c01_idx = 4 * g + 1
        c10_idx = 4 * g + 1  # Wait, this is wrong
        # C[1, 2g] -> W index (2g)*2 + 1 = 4g + 2
        # C[1, 2g+1] -> W index (2g+1)*2 + 1 = 4g + 3
        c10_idx = 4 * g + 2
        c11_idx = 4 * g + 3
        
        # m1 = (a+d)(e+h)
        col = base + 0
        U[0][col] += 1   # a
        U[3][col] += 1   # d
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        
        # m2 = (c+d)e
        col = base + 1
        U[2][col] += 1   # c
        U[3][col] += 1   # d
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        
        # m3 = a(f-h)
        col = base + 2
        U[0][col] += 1   # a
        V[f_idx][col] += 1
        V[h_idx][col] += -1
        W[c00_idx][col] += 1
        
        # m4 = d(g_-e)
        col = base + 3
        U[3][col] += 1   # d
        V[g_idx][col] += 1
        V[e_idx][col] += -1
        W[c10_idx][col] += 1
        W[c11_idx][col] += 1
        
        # m5 = (a+b)h
        col = base + 4
        U[0][col] += 1   # a
        U[1][col] += 1   # b
        V[h_idx][col] += 1
        W[c00_idx][col] += -1
        W[c01_idx][col] += 1
        
        # m6 = (c-a)(e+f)
        col = base + 5
        U[2][col] += 1   # c
        U[0][col] += -1  # -a
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        # m6 doesn't appear in C directly in the standard Strassen formula I wrote.
        # Let me recheck the Strassen formula.
        
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 - M5 + M7
        # C12 = M5 + M3
        # C21 = M2 + M4
        # C22 = M1 - M2 + M6
        
        # So I had the formulas slightly wrong. Let me redo.
        
    # Let me restart with the correct Strassen formula.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for g in range(4):
        base = g * 7
        e_idx = 2 * g        # B[0, 2g]
        f_idx = 2 * g + 1    # B[0, 2g+1]
        g_idx = 8 + 2 * g    # B[1, 2g]
        h_idx = 9 + 2 * g    # B[1, 2g+1]
        
        c00_idx = 4 * g      # C[0, 2g]
        c01_idx = 4 * g + 1  # C[0, 2g+1]
        c10_idx = 4 * g + 2  # C[1, 2g]
        c11_idx = 4 * g + 3  # C[1, 2g+1]
        
        # M1 = (A11 + A22)(B11 + B22)
        col = base + 0
        U[0][col] += 1   # A11
        U[3][col] += 1   # A22
        V[e_idx][col] += 1  # B11
        V[h_idx][col] += 1  # B22
        W[c00_idx][col] += 1   # C11
        W[c11_idx][col] += 1   # C22
        
        # M2 = (A21 + A22)B11
        col = base + 1
        U[2][col] += 1   # A21
        U[3][col] += 1   # A22
        V[e_idx][col] += 1  # B11
        W[c10_idx][col] += 1   # C21
        W[c11_idx][col] += -1  # C22 (with -)
        
        # M3 = A11(B21 - B22)
        col = base + 2
        U[0][col] += 1   # A11
        V[g_idx][col] += 1  # B21
        V[h_idx][col] += -1 # B22
        W[c01_idx][col] += 1   # C12
        
        # M4 = A22(B12 - B11)
        col = base + 3
        U[3][col] += 1   # A22
        V[f_idx][col] += 1  # B12
        V[e_idx][col] += -1 # B11
        W[c10_idx][col] += 1   # C21
        
        # M5 = (A11 + A12)B22
        col = base + 4
        U[0][col] += 1   # A11
        U[1][col] += 1   # A12
        V[h_idx][col] += 1  # B22
        W[c00_idx][col] += -1  # C11 (with -)
        W[c01_idx][col] += 1   # C12
        
        # M6 = (A21 - A11)(B11 + B12)
        col = base + 5
        U[2][col] += 1   # A21
        U[0][col] += -1  # -A11
        V[e_idx][col] += 1  # B11
        V[f_idx][col] += 1  # B12
        W[c11_idx][col] += 1   # C22
        
        # M7 = (A12 - A22)(B21 + B22)
        col = base + 6
        U[1][col] += 1   # A12
        U[3][col] += -1  # -A22
        V[g_idx][col] += 1  # B21
        V[h_idx][col] += 1  # B22
        W[c00_idx][col] += 1   # C11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
