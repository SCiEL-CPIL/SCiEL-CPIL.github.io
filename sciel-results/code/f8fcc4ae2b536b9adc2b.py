"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x4 matrix.
    
    The tensor T[i,j,k] = 1 if the output index (k*n+i) corresponds to
    C[i,k] = sum_j A[i,j]*B[j,k].
    
    We use the direct sum structure: 2x2x4 = (2x2x2) ⊕ (2x2x2).
    Each 2x2x2 block can be decomposed with rank 7 (Strassen).
    Total rank = 14.
    """
    n, m, p = DIMENSIONS
    # The tensor has dimensions (n*m) x (m*p) x (p*n) = 4 x 8 x 8
    # But structurally it's two independent 2x2x2 blocks.
    
    # For a 2x2x2 matrix multiplication, Strassen's algorithm gives rank 7.
    # The 7 rank-1 terms have specific integer coefficients.
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For the first block (i in {0,1}, j in {0,1}, k in {0,1}):
    # U indices: i*m+j where i in {0,1}, j in {0,1} -> 0,1,2,3
    # V indices: j*p+k where j in {0,1}, k in {0,1} -> 0,1,4,5
    # W indices: k*n+i where k in {0,1}, i in {0,1} -> 0,1,2,3
    
    # For the second block (i in {0,1}, j in {0,1}, k in {2,3}):
    # U indices: same as first block: 0,1,2,3
    # V indices: j*p+k where j in {0,1}, k in {2,3} -> 2,3,6,7
    # W indices: k*n+i where k in {2,3}, i in {0,1} -> 4,5,6,7
    
    # Strassen coefficients for each of the 7 terms:
    # Term t has: U coefficient for each (i,j), V coefficient for each (j,k), W coefficient for each (k,i)
    
    # Let's define the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs)
    # where U_coeffs is a dict from (i,j) to coeff, etc.
    
    strassen_terms = [
        # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C00=1, C11=1
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        # M2 = (c+d)*e: U: c=1,d=1; V: e=1; W: C10=1, C11=-1
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1}},
        # M3 = a*(f-h): U: a=1; V: f=1,h=-1; W: C01=1, C11=1
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}},
        # M4 = d*(g-e): U: d=1; V: g=1,e=-1; W: C00=1, C10=1
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}},
        # M5 = (a+b)*h: U: a=1,b=1; V: h=1; W: C00=-1, C01=1
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}},
        # M6 = (c-a)*(e+f): U: c=1,a=-1; V: e=1,f=1; W: C11=1
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        # M7 = (b-d)*(g+h): U: b=1,d=-1; V: g=1,h=1; W: C00=1
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]
    
    rank = 14
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for block in range(2):
        # block 0: k in {0,1}, block 1: k in {2,3}
        for term in strassen_terms:
            for (i, j), coeff_u in term['U'].items():
                U[i * m + j, col] += coeff_u
            for (j, k_local), coeff_v in term['V'].items():
                k = k_local + block * 2
                V[j * p + k, col] += coeff_v
            for (k_local, i), coeff_w in term['W'].items():
                k = k_local + block * 2
                W[k * n + i, col] += coeff_w
            col += 1
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
    
    # Clean up: convert to int where possible
    def clean_matrix(M):
        result = []
        for row in M:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    new_row.append(int(round(val)))
                elif abs(2*val - round(2*val)) < 1e-9:
                    new_row.append(round(2*val) / 2)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U_clean = clean_matrix(U_list)
    V_clean = clean_matrix(V_list)
    W_clean = clean_matrix(W_list)
    
    return {"U": U_clean, "V": V_clean, "W": W_clean}
# EVOLVE-BLOCK-END
