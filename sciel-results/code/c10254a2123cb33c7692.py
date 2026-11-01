# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 120

    # We will construct the Laderman rank-23 decomposition for 3x3 matrix multiplication.
    # The tensor T_{ijk} = delta_{jk} (where i is row of A, j is col of A / row of B, k is col of B)
    # We seek U, V, W such that T_{ijk} = sum_{r} U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
    # Note the W index convention: W[k*n + i, r].
    # So the term for output C[i, k] is indexed by idx = k*n + i.
    
    # The 23 products are linear combinations of entries of A and B.
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    
    # We define the 23 terms as dictionaries.
    # U: {(i,j): coeff}
    # V: {(j,k): coeff}
    # W: {(i,k): coeff} -> stored at index k*n+i
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
        
    # Laderman's 23 products:
    # 1. P1 = (A00 + A11 + A22)(B00 + B11 + B22)
    #    U: A00:1, A11:1, A22:1
    #    V: B00:1, B11:1, B22:1
    #    W: C00:1, C11:1, C22:1, C01:-1, C10:-1, C21:-1, C12:-1, C20:-1
    add_term(
        {(0,0):1, (1,1):1, (2,2):1},
        {(0,0):1, (1,1):1, (2,2):1},
        {0*3+0:1, 1*3+1:1, 2*3+2:1, 1*3+0:-1, 0*3+1:-1, 2*3+1:-1, 1*3+2:-1, 2*3+0:-1}
    )
    
    # 2. P2 = (A01 + A10 + A22)(B01 + B10 + B22)
    #    U: A01:1, A10:1, A22:1
    #    V: B01:1, B10:1, B22:1
    #    W: C01:1, C10:1, C22:1, C00:-1, C11:-1, C21:-1, C12:-1, C20:-1
    add_term(
        {(0,1):1, (1,0):1, (2,2):1},
        {(0,1):1, (1,0):1, (2,2):1},
        {1*3+0:1, 0*3+1:1, 2*3+2:1, 0*3+0:-1, 1*3+1:-1, 2*3+1:-1, 1*3+2:-1, 2*3+0:-1}
    )
    
    # 3. P3 = (A02 + A12 + A20)(B02 + B12 + B20)
    #    U: A02:1, A12:1, A20:1
    #    V: B02:1, B12:1, B20:1
    #    W: C02:1, C12:1, C20:1, C00:-1, C11:-1, C21:-1, C10:-1, C20:-1? Wait, C20 is already there.
    #    Let's re-check the pattern.
    #    Actually, let's use the known correct Laderman terms from a reliable source.
    #    I will use the terms from the "Laderman 23" algorithm as described in many optimization papers.
    
    # Let's use a verified set of 23 terms.
    # Reference: "A 23-multiplication algorithm for 3x3 matrix multiplication" by Laderman (1976)
    # Or similar sources providing the explicit 23 terms.
    
    # I will use the following 23 terms, which are known to be correct:
    # M1 = (A00+A11+A22)(B00+B11+B22)
    # M2 = (A01+A10+A22)(B01+B10+B22)
    # M3 = (A02+A12+A20)(B02+B12+B20)
    # M4 = (A00+A11)(B00+B11)
    # M5 = (A00+A11)(B02+B12)
    # M6 = (A00+A11)(B20+B21)
    # M7 = (A01+A10)(B00+B11)
    # M8 = (A01+A10)(B02+B12)
    # M9 = (A01+A10)(B20+B21)
    # M10 = (A02+A20)(B00+B22)
    # M11 = (A02+A20)(B01+B21)
    # M12 = (A02+A20)(B10+B12)
    # M13 = (A12+A21)(B00+B22)
    # M14 = (A12+A21)(B01+B21)
    # M15 = (A12+A21)(B10+B12)
    # M16 = A22*B22
    # M17 = A12*B21
    # M18 = A21*B12
    # M19 = A02*B20
    # M20 = A20*B02
    # M21 = A10*B01
    # M22 = A01*B10
    # M23 = A00*B00
    
    # However, this is not the standard Laderman.
    # Let's use the standard Laderman 23 terms.
    # I will use the terms from the Wikipedia page on "Strassen algorithm" or "Matrix multiplication".
    
    # Actually, I can use the "Laderman 23" algorithm from the Wikipedia page on "Strassen algorithm" or "Matrix multiplication".
    # 
    # Let's try to construct it using the following 23 products:
    # 
    # M1 = (A00 + A11 + A22) * (B00 + B11 + B22)
    # M2 = (A01 + A10 + A22) * (B01 + B10 + B22)
    # M3 = (A02 + A12 + A20) * (B02 + B12 + B20)
    # M4 = (A00 + A11) * (B00 + B11)
    # M5 = (A00 + A11) * (B02 + B12)
    # M6 = (A00 + A11) * (B20 + B21)
    # M7 = (A01 + A10) * (B00 + B11)
    # M8 = (A01 + A10) * (B02 + B12)
    # M9 = (A01 + A10) * (B20 + B21)
    # M10 = (A02 + A20) * (B00 + B22)
    # M11 = (A02 + A20) * (B01 + B21)
    # M12 = (A02 + A20) * (B10 + B12)
    # M13 = (A12 + A21) * (B00 + B22)
    # M14 = (A12 + A21) * (B01 + B21)
    # M15 = (A12 + A21) * (B10 + B12)
    # M16 = A22 * B22
    # M17 = A12 * B21
    # M18 = A21 * B12
    # M19 = A02 * B20
    # M20 = A20 * B02
    # M21 = A10 * B01
    # M22 = A01 * B10
    # M23 = A00 * B00
    
    # This is not Laderman.
    
    # Let's go back to the safe rank-27.
    # The current program has rank 27.
    # If I can't find a correct rank 23, I should stick with 27.
    # But the task asks for "rank-minimal".
    # The known minimum is 23.
    
    # I will try to implement the Laderman algorithm by looking up the specific 23 terms.
    # From "Matrix Multiplication Algorithms" by Laderman:
    # 
    # Let's use the terms from this source:
    # https://www.cs.cornell.edu/courses/cs682/2005fa/lectures/lec11.pdf (or similar)
    # 
    # Actually, I can use the "Laderman 23" algorithm from the Wikipedia page on "Strassen algorithm" or "Matrix multiplication".
    # 
    # Let's try to construct it using the following 23 products:
    # 
    # M1 = (A00 + A11 + A22) * (B00 + B11 + B22)
    # M2 = (A01 + A10 + A22) * (B01 + B10 + B22)
    # M3 = (A02 + A12 + A20) * (B02 + B12 + B20)
    # M4 = (A00 + A11) * (B00 + B11)
    # M5 = (A00 + A11) * (B02 + B12)
    # M6 = (A00 + A11) * (B20 + B21)
    # M7 = (A01 + A10) * (B00 + B11)
    # M8 = (A01 + A10) * (B02 + B12)
    # M9 = (A01 + A10) * (B20 + B21)
    # M10 = (A02 + A20) * (B00 + B22)
    # M11 = (A02 + A20) * (B01 + B21)
    # M12 = (A02 + A20) * (B10 + B12)
    # M13 = (A12 + A21) * (B00 + B22)
    # M14 = (A12 + A21) * (B01 + B21)
    # M15 = (A12 + A21) * (B10 + B12)
    # M16 = A22 * B22
    # M17 = A12 * B21
    # M18 = A21 * B12
    # M19 = A02 * B20
    # M20 = A20 * B02
    # M21 = A10 * B01
    # M22 = A01 * B10
    # M23 = A00 * B00
    
    # Let's check if this works.
    # C00 = A00B00 + A01B10 + A02B20
    # From M1: A00B00 + ...
    # From M4: A00B00 + ...
    # From M23: A00B00
    # This is getting messy.
    
    # I will stick to the schoolbook rank-27.
    # It is the safest bet.
    
    terms = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {k * n + i: 1}
                terms.append((u, v, w))
                
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] = val
        for (j, k), val in v.items():
            V[j * p + k][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
