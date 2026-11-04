"""
Matrix multiplication tensor decomposition for M_{2,3,10}.
Uses Strassen's algorithm on 2x2 sub-blocks for the first 2x2 part of the 
multiplication and schoolbook for the remaining 2x1x10 part.
Total rank: 5 * (7 + 4) = 55.
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a rank-55 exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Rank breakdown:
    # Split m=3 into m1=2 and m2=1.
    # A1 is 2x2, B1 is 2x10. Split B1 into 5 blocks of 2x2.
    # Each 2x2x2 block uses Strassen (rank 7). Total 35.
    # A2 is 2x1, B2 is 1x10. Split into 5 blocks of 2x1x2.
    # Each 2x1x2 block uses schoolbook (rank 4). Total 20.
    # Total rank = 55.
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # L vectors (coeffs for a_ij)
    # a00: idx 0, a01: idx 1, a10: idx 2, a11: idx 3 (local)
    # Global U index: i*m + j. m=3.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    
    L = [
        {0: 1, 3: 1},       # M1: a00, a11
        {2: 1, 3: 1},       # M2: a10, a11
        {0: 1},             # M3: a00
        {3: 1},             # M4: a11
        {0: 1, 1: 1},       # M5: a00, a01
        {2: 1, 0: -1},      # M6: a10, -a00
        {1: 1, 3: -1}       # M7: a01, -a11
    ]
    
    # R vectors (coeffs for b_jk)
    # b00: local 0, b01: local 1, b10: local 2, b11: local 3
    # Global V index: j*p + k. p=10.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 10
    # j=1, k=1 -> 11
    
    R = [
        {0: 1, 3: 1},       # M1: b00, b11
        {0: 1},             # M2: b00
        {1: 1, 3: -1},      # M3: b01, -b11
        {2: 1, 0: -1},      # M4: b10, -b00
        {3: 1},             # M5: b11
        {0: 1, 1: 1},       # M6: b00, b01
        {2: 1, 3: 1}        # M7: b10, b11
    ]
    
    # W vectors (coeffs for c_ik)
    # c00: local 0, c01: local 1, c10: local 2, c11: local 3
    # Global W index: k*n + i. n=2.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    W_coeffs = [
        {0: 1, 3: 1},       # M1: c00, c11
        {2: 1, 3: -1},      # M2: c10, -c11
        {1: 1, 3: 1},       # M3: c01, c11
        {0: 1, 2: 1},       # M4: c00, c10
        {1: 1, 0: -1},      # M5: c01, -c00
        {3: 1},             # M6: c11
        {0: 1}              # M7: c00
    ]
    
    # Fill Strassen blocks
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            # Fill U
            for local_idx, val in L[s].items():
                # local_idx 0->a00, 1->a01, 2->a10, 3->a11
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j][r] = val
            
            # Fill V
            for local_idx, val in R[s].items():
                # local_idx 0->b00, 1->b01, 2->b10, 3->b11
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k)][r] = val
            
            # Fill W
            for local_idx, val in W_coeffs[s].items():
                # local_idx 0->c00, 1->c01, 2->c10, 3->c11
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i][r] = val
            
            col += 1
            
    # Part 2: A2 B2 (2x1 * 1x10)
    # A2 is 2x1, indices i in {0,1}, j=2
    # B2 is 1x10, indices j=2, k in {0..9}
    # C2 is 2x10, indices i in {0,1}, k in {0..9}
    # C_ik = A_i2 B_2k
    
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            col += 1
            
    # Safety check
    if time.time() - start_time > 140:
        return {"U": U, "V": V, "W": W}
        
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    n, m, p = 2, 3, 10
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    print("Error:", err)
