"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 58: 8 Strassen blocks (7 each) + 2 tail terms (1 each)
    rank = 58
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U, V, W entries for the 7 terms
    # Term 0: (A00+A11)(B00+B11)
    # Term 1: (A10+A11)B00
    # Term 2: A00(B10-B11)
    # Term 3: A11(B01-B00)
    # Term 4: (A00+A10)B11
    # Term 5: (A10-A00)(B00+B10)
    # Term 6: (A01-A11)(B11+B01)
    
    # We need to map local (li, lj, lk) to global (i, j, k)
    # and add to U, V, W with appropriate signs/values.
    
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # 7 terms
        # 0: (A00+A11)(B00+B11)
        # U: i0j0 + i1j1
        # V: j0k0 + j1k1
        # W: k0i0 + k1i1? No, W is k*n+i.
        # W0: k0 i0, W1: k1 i1?
        # C00 = M0 - M1 - M3 + M6
        # C01 = M2 + M3
        # C10 = M1 + M4
        # C11 = M0 - M4 + M5 - M6
        
        # Let's define the 7 M's
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)B00
        # M2 = A00(B10-B11)
        # M3 = A11(B01-B00)
        # M4 = (A00+A10)B11
        # M5 = (A10-A00)(B00+B10)
        # M6 = (A01-A11)(B11+B01)
        
        # U factors (A side)
        # M0: A00, A11
        # M1: A10, A11
        # M2: A00
        # M3: A11
        # M4: A00, A10
        # M5: A10, -A00
        # M6: A01, -A11
        
        # V factors (B side)
        # M0: B00, B11
        # M1: B00
        # M2: B10, -B11
        # M3: B01, -B00
        # M4: B11
        # M5: B00, B10
        # M6: B11, B01
        
        # W factors (C side)
        # M0: C00, -C11? No.
        # C00 += M0
        # C11 += M0
        # C00 -= M1
        # C10 += M1
        # C01 += M2
        # C01 += M3
        # C11 -= M3
        # C10 += M4
        # C11 -= M4
        # C11 += M5
        # C00 += M6
        # C11 -= M6
        
        # Wait, standard Strassen:
        # C00 = M0 - M1 - M3 + M6
        # C01 = M2 + M3
        # C10 = M1 + M4
        # C11 = M0 - M4 + M5 - M6
        
        # So W contributions:
        # M0: W(k0, i0) += 1, W(k1, i1) += 1
        # M1: W(k0, i1) += 1, W(k0, i0) -= 1
        # M2: W(k0, i0) += 1
        # M3: W(k0, i0) -= 1, W(k1, i0) += 1
        # M4: W(k1, i1) -= 1, W(k1, i0) += 1
        # M5: W(k1, i1) += 1
        # M6: W(k0, i0) += 1, W(k1, i1) -= 1
        
        # Let's re-verify M1, M3, M4, M6 signs.
        # M1: (A10+A11)B00.
        # C10 += M1. C00 -= M1.
        # i=1, k=0: +1. i=0, k=0: -1.
        # M2: A00(B10-B11).
        # C01 += M2.
        # i=0, k=1: +1.
        # M3: A11(B01-B00).
        # C01 += M3. C11 -= M3.
        # i=0, k=1: +1. i=1, k=1: -1.
        # M4: (A00+A10)B11.
        # C10 += M4. C11 -= M4.
        # i=1, k=0: +1. i=1, k=1: -1.
        # M5: (A10-A00)(B00+B10).
        # C11 += M5.
        # i=1, k=1: +1.
        # M6: (A01-A11)(B11+B01).
        # C00 += M6. C11 -= M6.
        # i=0, k=0: +1. i=1, k=1: -1.
        
        # U assignments
        # M0: U(i0,j0)=1, U(i1,j1)=1
        # M1: U(i1,j0)=1, U(i1,j1)=1
        # M2: U(i0,j0)=1
        # M3: U(i1,j1)=1
        # M4: U(i0,j0)=1, U(i1,j0)=1
        # M5: U(i1,j0)=1, U(i0,j0)=-1
        # M6: U(i0,j1)=1, U(i1,j1)=-1
        
        # V assignments
        # M0: V(j0,k0)=1, V(j1,k1)=1
        # M1: V(j0,k0)=1
        # M2: V(j1,k0)=1, V(j1,k1)=-1
        # M3: V(j0,k1)=1, V(j0,k0)=-1
        # M4: V(j1,k1)=1
        # M5: V(j0,k0)=1, V(j0,k1)=1
        # M6: V(j1,k1)=1, V(j0,k1)=1
        
        # W assignments
        # M0: W(k0,i0)=1, W(k1,i1)=1
        # M1: W(k0,i1)=1, W(k0,i0)=-1
        # M2: W(k0,i0)=1
        # M3: W(k0,i0)=-1, W(k1,i0)=1
        # M4: W(k1,i0)=1, W(k1,i1)=-1
        # M5: W(k1,i1)=1
        # M6: W(k0,i0)=1, W(k1,i1)=-1
        
        for t in range(7):
            r = col + t
            # U
            if t == 0:
                U[i_base + 0][m * (j_base + 0) + r] += 0 # U is n*m x rank. Index i*m+j.
                # Correct Index: row = i*m + j
                U[(i_base + 0) * m + (j_base + 0)][r] = 1
                U[(i_base + 1) * m + (j_base + 1)][r] = 1
            elif t == 1:
                U[(i_base + 1) * m + (j_base + 0)][r] = 1
                U[(i_base + 1) * m + (j_base + 1)][r] = 1
            elif t == 2:
                U[(i_base + 0) * m + (j_base + 0)][r] = 1
            elif t == 3:
                U[(i_base + 1) * m + (j_base + 1)][r] = 1
            elif t == 4:
                U[(i_base + 0) * m + (j_base + 0)][r] = 1
                U[(i_base + 1) * m + (j_base + 0)][r] = 1
            elif t == 5:
                U[(i_base + 1) * m + (j_base + 0)][r] = 1
                U[(i_base + 0) * m + (j_base + 0)][r] = -1
            elif t == 6:
                U[(i_base + 0) * m + (j_base + 1)][r] = 1
                U[(i_base + 1) * m + (j_base + 1)][r] = -1
            
            # V
            if t == 0:
                V[(j_base + 0) * p + (k_base + 0)][r] = 1
                V[(j_base + 1) * p + (k_base + 1)][r] = 1
            elif t == 1:
                V[(j_base + 0) * p + (k_base + 0)][r] = 1
            elif t == 2:
                V[(j_base + 1) * p + (k_base + 0)][r] = 1
                V[(j_base + 1) * p + (k_base + 1)][r] = -1
            elif t == 3:
                V[(j_base + 0) * p + (k_base + 1)][r] = 1
                V[(j_base + 0) * p + (k_base + 0)][r] = -1
            elif t == 4:
                V[(j_base + 1) * p + (k_base + 1)][r] = 1
            elif t == 5:
                V[(j_base + 0) * p + (k_base + 0)][r] = 1
                V[(j_base + 0) * p + (k_base + 1)][r] = 1
            elif t == 6:
                V[(j_base + 1) * p + (k_base + 1)][r] = 1
                V[(j_base + 0) * p + (k_base + 1)][r] = 1
            
            # W
            if t == 0:
                W[(k_base + 0) * n + (i_base + 0)][r] = 1
                W[(k_base + 1) * n + (i_base + 1)][r] = 1
            elif t == 1:
                W[(k_base + 0) * n + (i_base + 1)][r] = 1
                W[(k_base + 0) * n + (i_base + 0)][r] = -1
            elif t == 2:
                W[(k_base + 0) * n + (i_base + 0)][r] = 1
            elif t == 3:
                W[(k_base + 0) * n + (i_base + 0)][r] = -1
                W[(k_base + 1) * n + (i_base + 0)][r] = 1
            elif t == 4:
                W[(k_base + 1) * n + (i_base + 0)][r] = 1
                W[(k_base + 1) * n + (i_base + 1)][r] = -1
            elif t == 5:
                W[(k_base + 1) * n + (i_base + 1)][r] = 1
            elif t == 6:
                W[(k_base + 0) * n + (i_base + 0)][r] = 1
                W[(k_base + 1) * n + (i_base + 1)][r] = -1
        
        col += 7

    # 8 Strassen blocks
    add_strassen(0, 0, 0) # I1, J1, K1
    add_strassen(0, 2, 0) # I1, J2, K1
    add_strassen(0, 0, 2) # I1, J1, K2
    add_strassen(0, 2, 2) # I1, J2, K2
    add_strassen(2, 0, 0) # I2, J1, K1
    add_strassen(2, 2, 0) # I2, J2, K1
    add_strassen(2, 0, 2) # I2, J1, K2
    add_strassen(2, 2, 2) # I2, J2, K2
    
    # Tail 1: j=4
    # U[i, 4] = 1 for all i
    # V[4, k] = 1 for all k
    # W[k, i] = 1 for all k, i
    r = col
    for i in range(n):
        U[i * m + 4][r] = 1
    for k in range(p):
        V[4 * p + k][r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][r] = 1
    col += 1
    
    # Tail 2: k=4, j in 0..3
    # U[i, j] = 1 for j in 0..3
    # V[j, 4] = 1 for j in 0..3
    # W[4, i] = 1 for all i
    r = col
    for i in range(n):
        for j in range(4):
            U[i * m + j][r] = 1
    for j in range(4):
        V[j * p + 4][r] = 1
    for i in range(n):
        W[4 * n + i][r] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
