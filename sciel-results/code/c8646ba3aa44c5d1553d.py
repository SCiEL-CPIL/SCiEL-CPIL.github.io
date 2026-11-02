import time
import numpy as np

def construct():
    """
    Matrix multiplication tensor decomposition for 3x3 x 3x4.
    Returns a rank-34 decomposition using Strassen-like optimizations.
    """
    n, m, p = 3, 3, 4
    rank = 34

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    c = 0

    # 1. A_22 * B_24 (Strassen on two 2x2x2x2 blocks)
    # B_24 is split into B_L (cols 0,1) and B_R (cols 2,3)
    
    # Block L (cols 0,1 of B, contributing to cols 0,1 of C)
    # A_22 is rows 0,1 cols 0,1 of A
    # B_L is rows 0,1 cols 0,1 of B
    
    # P1 = (A00 + A11)(B00 + B11)
    U[0, c] = 1; U[3, c] = 1 # A00, A11
    V[0, c] = 1; V[3, c] = 1 # B00, B11
    W[0*3+0, c] = 1 # C00
    W[1*3+1, c] = 1 # C11
    c += 1
    
    # P2 = (A10 + A11) B00
    U[3, c] = 1; U[4, c] = 1 # A10, A11 (indices: row 1 col 0 is 1*3+0=3, row 1 col 1 is 1*3+1=4)
    V[0, c] = 1 # B00
    W[1*3+0, c] = 1 # C10
    c += 1
    
    # P3 = A00 (B01 - B11)
    U[0, c] = 1 # A00
    V[1, c] = 1; V[3, c] = -1 # B01, B11
    W[0*3+1, c] = 1 # C01
    c += 1
    
    # P4 = A11 (B10 - A10)
    U[4, c] = 1 # A11
    V[2, c] = 1; U[3, c] = -1 # B10, A10
    W[1*3+1, c] = 1 # C11
    c += 1
    
    # P5 = (A00 + A01) B11
    U[0, c] = 1; U[1, c] = 1 # A00, A01
    V[3, c] = 1 # B11
    W[0*3+1, c] = -1 # C01
    W[1*3+0, c] = -1 # C10
    c += 1
    
    # P6 = (A10 - A00) (B00 + B01)
    U[3, c] = 1; U[0, c] = -1 # A10, A00
    V[0, c] = 1; V[1, c] = 1 # B00, B01
    W[1*3+1, c] = 1 # C11
    c += 1
    
    # P7 = (A01 - A11) (B10 + B11)
    U[1, c] = 1; U[4, c] = -1 # A01, A11
    V[2, c] = 1; V[3, c] = 1 # B10, B11
    W[0*3+0, c] = 1 # C00
    c += 1

    # Block R (cols 2,3 of B, contributing to cols 2,3 of C)
    # B_R is rows 0,1 cols 2,3 of B
    
    # P1 = (A00 + A11)(B02 + B12)
    U[0, c] = 1; U[3, c] = 1 # A00, A11
    V[2, c] = 1; V[5, c] = 1 # B02, B12 (indices: row 0 col 2 is 0*4+2=2, row 1 col 2 is 1*4+2=6? No, V is 3x4. V[j*p+k]. j=0,k=2 -> 2. j=1,k=2 -> 6. Wait. m=3, p=4. V is 3x4. Index j*p+k.
    # Let's re-verify V indices.
    # V is (m*p, rank) = (12, rank).
    # B_{j,k} is at V[j*p+k].
    # B_{0,2} is V[2]. B_{1,2} is V[6].
    # My previous indices were wrong.
    
    # Let's restart the index mapping carefully.
    # A is 3x3. U is 9xR. A_{i,j} at U[i*m+j] = U[i*3+j].
    # B is 3x4. V is 12xR. B_{j,k} at V[j*p+k] = V[j*4+k].
    # C is 3x4. W is 12xR. C_{i,k} at W[k*n+i] = W[k*3+i].

    # Reset and redo.
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0

    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal c
        if u_val != 0:
            U[u_idx, c] += u_val
        if v_val != 0:
            V[v_idx, c] += v_val
        if w_val != 0:
            W[w_idx, c] += w_val
        c += 1

    # 1. A_22 * B_L (Strassen)
    # A_22: A00(0), A01(1), A10(3), A11(4)
    # B_L: B00(0), B01(1), B10(4), B11(5)
    
    # P1 = (A00 + A11)(B00 + B11)
    add_term(0, 1, 0, 1, 0, 1) # C00
    add_term(4, 1, 0, 1, 4, 1) # C11 -- Wait, Strassen P1 contributes to C00 and C11?
    # C00 = P1 + P4 - P5 + P7
    # C11 = P1 - P2 + P3 + P6
    # So P1 adds to both. We need separate terms for U,V,W or combine?
    # The tensor decomposition requires each rank-1 term to be a single multiplication.
    # P1 is ONE multiplication: (A00+A11) * (B00+B11).
    # So we have ONE term in U,V,W.
    # U[0]=1, U[4]=1. V[0]=1, V[5]=1.
    # This product contributes to C00 and C11.
    # So W[0] += 1, W[4] += 1.
    
    # Let's re-implement add_term to handle multiple W outputs for one multiplication?
    # No, a rank-1 term is $u \otimes v \otimes w$.
    # If the multiplication result contributes to multiple C entries, the w vector has multiple non-zero entries.
    
    def add_mult(u_coeffs, v_coeffs, w_coeffs):
        nonlocal c
        for idx, val in u_coeffs.items():
            U[idx, c] += val
        for idx, val in v_coeffs.items():
            V[idx, c] += val
        for idx, val in w_coeffs.items():
            W[idx, c] += val
        c += 1

    # A indices: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22
    # B indices: 0:00, 1:01, 2:02, 3:03, 4:10, 5:11, 6:12, 7:13, 8:20, 9:21, 10:22, 11:23
    # C indices (k*3+i): 
    # k=0: 0:00, 1:10, 2:20
    # k=1: 3:01, 4:11, 5:21
    # k=2: 6:02, 7:12, 8:22
    # k=3: 9:03, 10:13, 11:23

    # 1. A_22 * B_L
    # P1 = (A00+A11)(B00+B11) -> C00, C11
    add_mult({0:1, 4:1}, {0:1, 5:1}, {0:1, 4:1})
    
    # P2 = (A10+A11)B00 -> C10
    add_mult({3:1, 4:1}, {0:1}, {1:1}) # C10 is index 1
    
    # P3 = A00(B01-B11) -> C01
    add_mult({0:1}, {1:1, 5:-1}, {3:1}) # C01 is index 3
    
    # P4 = A11(B10-A10) -> C11
    add_mult({4:1}, {4:1, 3:-1}, {4:1}) # C11 is index 4. Note: A10 is in U, not V.
    # Wait, P4 = d(g-c). d=A11, g=B10, c=A10.
    # So U[4]=1, V[4]=1, U[3]=-1.
    # This is a valid rank-1 term.
    
    # P5 = (A00+A01)B11 -> C01, C10
    add_mult({0:1, 1:1}, {5:1}, {3:-1, 1:-1}) # C01 idx 3, C10 idx 1
    
    # P6 = (A10-A00)(B00+B01) -> C11
    add_mult({3:1, 0:-1}, {0:1, 1:1}, {4:1}) # C11 idx 4
    
    # P7 = (A01-A11)(B10+B11) -> C00
    add_mult({1:1, 4:-1}, {4:1, 5:1}, {0:1}) # C00 idx 0

    # 2. A_22 * B_R
    # B_R: B02(2), B03(3), B12(6), B13(7)
    # Outputs: C02(6), C03(9), C12(7), C13(10)
    
    # P1 = (A00+A11)(B02+B12) -> C02, C12
    add_mult({0:1, 4:1}, {2:1, 6:1}, {6:1, 7:1})
    
    # P2 = (A10+A11)B02 -> C12
    add_mult({3:1, 4:1}, {2:1}, {7:1})
    
    # P3 = A00(B03-B13) -> C03
    add_mult({0:1}, {3:1, 7:-1}, {9:1})
    
    # P4 = A11(B12-A10) -> C12
    add_mult({4:1}, {6:1, 3:-1}, {7:1}) # A10 is U[3]
    
    # P5 = (A00+A01)B12 -> C03, C12
    add_mult({0:1, 1:1}, {6:1}, {9:-1, 7:-1})
    
    # P6 = (A10-A00)(B02+B03) -> C12
    add_mult({3:1, 0:-1}, {2:1, 3:1}, {7:1})
    
    # P7 = (A01-A11)(B12+B13) -> C02
    add_mult({1:1, 4:-1}, {6:1, 7:1}, {6:1})

    # 3. a * B_14
    # a = A02(2), A12(5)
    # B_14 = B20(8), B21(9), B22(10), B23(11)
    # Outputs: C00(0), C01(3), C02(6), C03(9) for A02
    #          C10(1), C11(4), C12(7), C13(10) for A12
    
    add_mult({2:1}, {8:1}, {0:1}) # A02*B20 -> C00
    add_mult({2:1}, {9:1}, {3:1}) # A02*B21 -> C01
    add_mult({2:1}, {10:1}, {6:1}) # A02*B22 -> C02
    add_mult({2:1}, {11:1}, {9:1}) # A02*B23 -> C03
    
    add_mult({5:1}, {8:1}, {1:1}) # A12*B20 -> C10
    add_mult({5:1}, {9:1}, {4:1}) # A12*B21 -> C11
    add_mult({5:1}, {10:1}, {7:1}) # A12*B22 -> C12
    add_mult({5:1}, {11:1}, {10:1}) # A12*B23 -> C13

    # 4. b^T * B_24
    # b = A20(6), A21(7)
    # B_24 = B00(0)..B03(3), B10(4)..B13(7)
    # Outputs: C20(2), C21(5), C22(8), C23(11)
    
    add_mult({6:1}, {0:1}, {2:1}) # A20*B00 -> C20
    add_mult({6:1}, {1:1}, {5:1}) # A20*B01 -> C21
    add_mult({6:1}, {2:1}, {8:1}) # A20*B02 -> C22
    add_mult({6:1}, {3:1}, {11:1}) # A20*B03 -> C23
    
    add_mult({7:1}, {4:1}, {2:1}) # A21*B10 -> C20
    add_mult({7:1}, {5:1}, {5:1}) # A21*B11 -> C21
    add_mult({7:1}, {6:1}, {8:1}) # A21*B12 -> C22
    add_mult({7:1}, {7:1}, {11:1}) # A21*B13 -> C23

    # 5. c * B_14
    # c = A22(8)
    # B_14 = B20(8)..B23(11)
    # Outputs: C20(2)..C23(11)
    
    add_mult({8:1}, {8:1}, {2:1}) # A22*B20 -> C20
    add_mult({8:1}, {9:1}, {5:1}) # A22*B21 -> C21
    add_mult({8:1}, {10:1}, {8:1}) # A22*B22 -> C22
    add_mult({8:1}, {11:1}, {11:1}) # A22*B23 -> C23

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
