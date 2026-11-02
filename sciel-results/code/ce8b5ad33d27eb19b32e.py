import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # We'll use a block decomposition approach.
    # Split A (3x4) into blocks, B (4x8) into blocks.
    # For each block multiplication, we can use fast matrix multiplication.
    
    # Strategy: Use 2x2 Strassen-like decomposition where possible.
    # For a 2x2 * 2x2 multiplication, Strassen uses 7 multiplications instead of 8.
    
    # Let's think about this more carefully.
    # A is 3x4, B is 4x8, C is 3x8.
    
    # We can decompose the inner dimension (4) into 2+2.
    # A = [A1 A2] where A1 is 3x2, A2 is 3x2
    # B = [B1; B2] where B1 is 2x8, B2 is 2x8
    # C = A1*B1 + A2*B2
    
    # Each of A1*B1 and A2*B2 is a 3x2 * 2x8 multiplication.
    # We can further decompose these.
    
    # For 3x2 * 2x8:
    # Split the 8 dimension into 4+4: B1 = [B1a B1b] where B1a is 2x4, B1b is 2x4
    # C1 = A1*B1a + A1*B1b (each is 3x2 * 2x4)
    
    # For 3x2 * 2x4:
    # Split the 4 dimension into 2+2: B1a = [B1a1 B1a2] where each is 2x2
    # C1a = A1*B1a1 + A1*B1a2 (each is 3x2 * 2x2)
    
    # For 3x2 * 2x2:
    # Split the 3 dimension into 1+2: A1 = [A1a; A1b] where A1a is 1x2, A1b is 2x2
    # C1a1 = A1a*B1a1 + A1b*B1a1
    # A1a*B1a1 is 1x2 * 2x2 (trivial, 4 mults)
    # A1b*B1a1 is 2x2 * 2x2 (can use Strassen: 7 mults)
    
    # Let's count the rank for this decomposition:
    # Original: 3*4*8 = 96
    # 
    # C = A1*B1 + A2*B2
    # A1*B1 = A1*[B1a B1b] = A1*B1a + A1*B1b
    # A1*B1a = A1*[B1a1 B1a2] = A1*B1a1 + A1*B1a2
    # A1*B1a1 = [A1a;A1b]*B1a1 = A1a*B1a1 + A1b*B1a1
    # A1a*B1a1: 1x2*2x2 = 4 scalar mults
    # A1b*B1a1: 2x2*2x2 = 7 (Strassen)
    # So A1*B1a1 uses 4+7 = 11 mults
    # Similarly A1*B1a2 uses 11 mults
    # So A1*B1a uses 22 mults
    # Similarly A1*B1b uses 22 mults
    # So A1*B1 uses 44 mults
    # Similarly A2*B2 uses 44 mults
    # Total: 88 mults
    # 
    # That's an improvement from 96 to 88!
    
    # Let's implement this. We'll build the U, V, W matrices.
    
    # For each "multiplication" in the decomposition, we add one rank-1 term.
    # U[i*m+j, rank_idx] = coefficient for A[i,j]
    # V[j*p+k, rank_idx] = coefficient for B[j,k]  
    # W[k*n+i, rank_idx] = coefficient for C[i,k]
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_val, v_val, w_val):
        nonlocal rank
        U.append([0] * (rank + 1) if not U else [0] * rank + [u_val])
        # Actually, let's build these differently
        pass
    
    # Let me rebuild this more carefully.
    # We'll collect all the rank-1 terms.
    
    terms = []  # Each term is (u, v, w) where u, v, w are dicts mapping index to coefficient
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Helper: add a scalar multiplication term A[i,j] * B[j,k] -> C[i,k]
    def add_scalar_mult(i, j, k):
        u_dict = {i * m + j: 1}
        v_dict = {j * p + k: 1}
        w_dict = {k * n + i: 1}
        add_rank1_term(u_dict, v_dict, w_dict)
    
    # Helper: add a Strassen 2x2*2x2 multiplication
    # A_block is 2x2, B_block is 2x2, C_block is 2x2
    # A_block indices: (i0,j0), (i0,j1), (i1,j0), (i1,j1)
    # B_block indices: (j0,k0), (j0,k1), (j1,k0), (j1,k1)
    # C_block indices: (i0,k0), (i0,k1), (i1,k0), (i1,k1)
    def add_strassen_2x2(i0, j0, k0):
        # Strassen's 7 products for 2x2*2x2:
        # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        # M1 = (a+d)(e+h)
        # This is a*(e+h) + d*(e+h) = a*e + a*h + d*e + d*h
        # But we need to express this as rank-1 terms.
        # Actually, Strassen combines products. Let me think again.
        
        # The 7 products are:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)*e
        # M3 = a*(f-h)
        # M4 = d*(g-e)
        # M5 = (a+b)*h
        # M6 = (c-a)*(e+f)
        # M7 = (b-d)*(g+h)
        
        # Each Mi is a product of a linear combination of A entries and a linear combination of B entries.
        # So each Mi corresponds to one rank-1 term in the tensor decomposition.
        
        # For M1: U part = a + d, V part = e + h, W part = contributes to C
        # C[0,0] += M1 + M4 - M5 + M7
        # C[0,1] += M3 + M5
        # C[1,0] += M2 + M4
        # C[1,1] += M1 - M2 + M3 + M6
        
        # M1: U: {(i0,j0):1, (i1,j1):1}, V: {(j0,k0):1, (j0,k1):1}, W: {(k0,i0):1, (k1,i0):1, (k0,i1):-1, (k1,i1):1}
        # Wait, I need to be more careful.
        
        # Let me define the indices:
        # A entries: a=A[i0,j0], b=A[i0,j1], c=A[i1,j0], d=A[i1,j1]
        # B entries: e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
        # C entries: C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
        
        # M1 = (a+d)(e+h)
        # U_M1 = {(i0,j0):1, (i1,j1):1}
        # V_M1 = {(j0,k0):1, (j0,k1):1}
        # W_M1: contributes to C[i0,k0] with +1, C[i0,k1] with +1, C[i1,k0] with -1, C[i1,k1] with +1
        # Wait, that's not right. Let me recompute.
        
        # C[0,0] = ae + bg = M1 + M4 - M5 + M7
        # C[0,1] = af + bh = M3 + M5
        # C[1,0] = ce + dg = M2 + M4
        # C[1,1] = cf + dh = M1 - M2 + M3 + M6
        
        # So M1 contributes: +1 to C[0,0], 0 to C[0,1], 0 to C[1,0], +1 to C[1,1]
        # M2 contributes: 0 to C[0,0], 0 to C[0,1], +1 to C[1,0], -1 to C[1,1]
        # M3 contributes: 0 to C[0,0], +1 to C[0,1], 0 to C[1,0], +1 to C[1,1]
        # M4 contributes: +1 to C[0,0], 0 to C[0,1], +1 to C[1,0], 0 to C[1,1]
        # M5 contributes: -1 to C[0,0], +1 to C[0,1], 0 to C[1,0], 0 to C[1,1]
        # M6 contributes: 0 to C[0,0], 0 to C[0,1], 0 to C[1,0], +1 to C[1,1]
        # M7 contributes: +1 to C[0,0], 0 to C[0,1], 0 to C[1,0], 0 to C[1,1]
        
        # Let me verify:
        # C[0,0] = M1 + M4 - M5 + M7 ✓
        # C[0,1] = M3 + M5 ✓
        # C[1,0] = M2 + M4 ✓
        # C[1,1] = M1 - M2 + M3 + M6 ✓
        
        # Now let's add each Mi as a rank-1 term.
        
        # M1 = (a+d)(e+h)
        u_m1 = {(i0,j0):1, (i1,j1):1}
        v_m1 = {(j0,k0):1, (j0,k1):1}
        w_m1 = {(k0,i0):1, (k1,i1):1}  # Contributes +1 to C[i0,k0] and +1 to C[i1,k1]
        add_rank1_term(u_m1, v_m1, w_m1)
        
        # M2 = (c+d)*e
        u_m2 = {(i1,j0):1, (i1,j1):1}
        v_m2 = {(j0,k0):1}
        w_m2 = {(k0,i1):1, (k1,i1):-1}  # Contributes +1 to C[i1,k0] and -1 to C[i1,k1]
        add_rank1_term(u_m2, v_m2, w_m2)
        
        # M3 = a*(f-h)
        u_m3 = {(i0,j0):1}
        v_m3 = {(j0,k1):1, (j1,k1):-1}
        w_m3 = {(k1,i0):1, (k1,i1):1}  # Contributes +1 to C[i0,k1] and +1 to C[i1,k1]
        add_rank1_term(u_m3, v_m3, w_m3)
        
        # M4 = d*(g-e)
        u_m4 = {(i1,j1):1}
        v_m4 = {(j1,k0):1, (j0,k0):-1}
        w_m4 = {(k0,i0):1, (k0,i1):1}  # Contributes +1 to C[i0,k0] and +1 to C[i1,k0]
        add_rank1_term(u_m4, v_m4, w_m4)
        
        # M5 = (a+b)*h
        u_m5 = {(i0,j0):1, (i0,j1):1}
        v_m5 = {(j1,k1):1}
        w_m5 = {(k1,i0):-1, (k1,i1):1}  # Contributes -1 to C[i0,k0] and +1 to C[i0,k1]
        add_rank1_term(u_m5, v_m5, w_m5)
        
        # M6 = (c-a)*(e+f)
        u_m6 = {(i1,j0):1, (i0,j0):-1}
        v_m6 = {(j0,k0):1, (j0,k1):1}
        w_m6 = {(k1,i1):1}  # Contributes +1 to C[i1,k1]
        add_rank1_term(u_m6, v_m6, w_m6)
        
        # M7 = (b-d)*(g+h)
        u_m7 = {(i0,j1):1, (i1,j1):-1}
        v_m7 = {(j1,k0):1, (j1,k1):1}
        w_m7 = {(k0,i0):1}  # Contributes +1 to C[i0,k0]
        add_rank1_term(u_m7, v_m7, w_m7)
    
    # Now let's build the decomposition:
    # C = A1*B1 + A2*B2 where A1=A[:,0:2], A2=A[:,2:4], B1=B[0:2,:], B2=B[2:4,:]
    
    # A1*B1: 3x2 * 2x8
    # Split B1 into B1a=B[0:2,0:4] and B1b=B[0:2,4:8]
    # A1*B1a: 3x2 * 2x4
    # Split B1a into B1a1=B[0:2,0:2] and B1a2=B[0:2,2:4]
    # A1*B1a1: 3x2 * 2x2
    # Split A1 into A1a=A[0:1,0:2] (1x2) and A1b=A[1:3,0:2] (2x2)
    # A1a*B1a1: 1x2 * 2x2 = 4 scalar mults
    # A1b*B1a1: 2x2 * 2x2 = 7 (Strassen)
    
    # Let's add these terms:
    
    # A1a*B1a1: i=0, j=0,1, k=0,1
    for j in range(2):
        for k in range(2):
            add_scalar_mult(0, j, k)
    
    # A1b*B1a1: i=1,2, j=0,1, k=0,1 -> Strassen with i0=1, j0=0, k0=0
    add_strassen_2x2(1, 0, 0)
    
    # A1*B1a2: i=0,1,2, j=0,1, k=2,3
    # A1a*B1a2: i=0, j=0,1, k=2,3
    for j in range(2):
        for k in range(2, 4):
            add_scalar_mult(0, j, k)
    
    # A1b*B1a2: i=1,2, j=0,1, k=2,3 -> Strassen with i0=1, j0=0, k0=2
    add_strassen_2x2(1, 0, 2)
    
    # A1*B1b: i=0,1,2, j=0,1, k=4,5,6,7
    # A1a*B1b: i=0, j=0,1, k=4,5,6,7
    for j in range(2):
        for k in range(4, 8):
            add_scalar_mult(0, j, k)
    
    # A1b*B1b: i=1,2, j=0,1, k=4,5,6,7
    # Split B1b into B1b1=B[0:2,4:6] and B1b2=B[0:2,6:8]
    # A1b*B1b1: 2x2 * 2x2 -> Strassen with i0=1, j0=0, k0=4
    add_strassen_2x2(1, 0, 4)
    # A1b*B1b2: 2x2 * 2x2 -> Strassen with i0=1, j0=0, k0=6
    add_strassen_2x2(1, 0, 6)
    
    # A2*B2: i=0,1,2, j=2,3, k=0,1,2,3,4,5,6,7
    # Split B2 into B2a=B[2:4,0:4] and B2b=B[2:4,4:8]
    # A2*B2a: i=0,1,2, j=2,3, k=0,1,2,3
    # A2a*B2a: i=0, j=2,3, k=0,1,2,3
    for j in range(2, 4):
        for k in range(0, 4):
            add_scalar_mult(0, j, k)
    
    # A2b*B2a: i=1,2, j=2,3, k=0,1,2,3
    # Split B2a into B2a1=B[2:4,0:2] and B2a2=B[2:4,2:4]
    # A2b*B2a1: 2x2 * 2x2 -> Strassen with i0=1, j0=2, k0=0
    add_strassen_2x2(1, 2, 0)
    # A2b*B2a2: 2x2 * 2x2 -> Strassen with i0=1, j0=2, k0=2
    add_strassen_2x2(1, 2, 2)
    
    # A2*B2b: i=0,1,2, j=2,3, k=4,5,6,7
    # A2a*B2b: i=0, j=2,3, k=4,5,6,7
    for j in range(2, 4):
        for k in range(4, 8):
            add_scalar_mult(0, j, k)
    
    # A2b*B2b: i=1,2, j=2,3, k=4,5,6,7
    # Split B2b into B2b1=B[2:4,4:6] and B2b2=B[2:4,6:8]
    # A2b*B2b1: 2x2 * 2x2 -> Strassen with i0=1, j0=2, k0=4
    add_strassen_2x2(1, 2, 4)
    # A2b*B2b2: 2x2 * 2x2 -> Strassen with i0=1, j0=2, k0=6
    add_strassen_2x2(1, 2, 6)
    
    # Now build U, V, W from the terms
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}
