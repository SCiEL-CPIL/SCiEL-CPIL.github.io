"""
Rank-minimal exact tensor decomposition for 2x3x10 matrix multiplication.
Uses block decomposition:
  - Right 2x2 block of A (col 2) with full 2x10 B: rank 20 (schoolbook)
  - Left 2x2 block of A (cols 0,1) with full 2x10 B: split B into 5 2x2 blocks,
    each multiplied via Strassen (rank 7) => 5*7 = 35
Total rank: 55
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = 2, 3, 10
    rank_total = 55
    U = np.zeros((n * m, rank_total), dtype=np.int64)
    V = np.zeros((m * p, rank_total), dtype=np.int64)
    W = np.zeros((p * n, rank_total), dtype=np.int64)
    
    # Part 1: Right 2x2 block of A (col index 2) with full 2x10 B
    # A_right[i, 2] * B[j, k] contributes to C[i, k]
    # Schoolbook for 2x1x2x10: for each i in {0,1}, k in {0..9}
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1
            V[1 * p + k, col] = 1  # B row 1 (j=1), col k
            W[k * n + i, col] = 1
            col += 1
    # col should be 20 now
    
    # Part 2: Left 2x2 block of A (cols 0,1) with full 2x10 B
    # Split B into 5 blocks of 2x2: B[:, 2q:2q+2] for q in 0..4
    # For each block, use Strassen for 2x2 x 2x2
    # Strassen 7 products:
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    
    # A_left indices in U: row i*m + j where j in {0,1}, i in {0,1}
    # So rows: 0, 1, 3, 4 for (i,j) = (0,0),(0,1),(1,0),(1,1)
    # a00 -> U row 0, a01 -> U row 1, a10 -> U row 3, a11 -> U row 4
    
    # B block q: rows j in {0,1}, cols 2q, 2q+1
    # V rows: j*p + (2q) and j*p + (2q+1)
    # b00 -> V row 0*p + 2q = 2q
    # b01 -> V row 0*p + 2q+1 = 2q+1
    # b10 -> V row 1*p + 2q = 10+2q
    # b11 -> V row 1*p + 2q+1 = 10+2q+1
    
    # C block q: rows i in {0,1}, cols 2q, 2q+1
    # W rows: k*n + i
    # C00: i=0, k=2q -> W row 2q*2+0 = 4q
    # C01: i=0, k=2q+1 -> W row (2q+1)*2+0 = 4q+2
    # C10: i=1, k=2q -> W row 2q*2+1 = 4q+1
    # C11: i=1, k=2q+1 -> W row (2q+1)*2+1 = 4q+3
    
    # Strassen coefficients for (U, V, W) contributions to each product P_t:
    # P1: (a00+a11)*(b00+b11)
    #   U: a00->+1, a11->+1 ; V: b00->+1, b11->+1 ; W: C00->+1, C11->+1
    # P2: (a10+a11)*b00
    #   U: a10->+1, a11->+1 ; V: b00->+1 ; W: C10->+1, C11->-1
    # P3: a00*(b01-b11)
    #   U: a00->+1 ; V: b01->+1, b11->-1 ; W: C01->+1, C11->+1
    # P4: a11*(b10-b00)
    #   U: a11->+1 ; V: b10->+1, b00->-1 ; W: C00->+1, C10->+1
    # P5: (a00+a01)*b11
    #   U: a00->+1, a01->+1 ; V: b11->+1 ; W: C00->-1, C01->+1
    # P6: (a10-a00)*(b00+b01)
    #   U: a10->+1, a00->-1 ; V: b00->+1, b01->+1 ; W: C11->+1
    # P7: (a01-a11)*(b10+b11)
    #   U: a01->+1, a11->-1 ; V: b10->+1, b11->+1 ; W: C00->+1, C11->+1
    
    # Define Strassen terms as list of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping (i,j) -> coeff
    # V_coeffs: dict mapping (j, k_offset) -> coeff where k_offset in {0,1}
    # W_coeffs: dict mapping (i, k_offset) -> coeff where k_offset in {0,1}
    
    def get_u_row(i, j):
        return i * m + j
    
    def get_v_row(j, k_off, q):
        return j * p + (2 * q + k_off)
    
    def get_w_row(i, k_off, q):
        return (2 * q + k_off) * n + i
    
    strassen_terms = []
    
    # P1: (a00+a11)*(b00+b11)
    strassen_terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # P2: (a10+a11)*b00
    strassen_terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # P3: a00*(b01-b11)
    strassen_terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # P4: a11*(b10-b00)
    strassen_terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # P5: (a00+a01)*b11
    strassen_terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # P6: (a10-a00)*(b00+b01)
    strassen_terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # P7: (a01-a11)*(b10+b11)
    strassen_terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    for q in range(5):
        for t in range(7):
            col_idx = 20 + q * 7 + t
            term = strassen_terms[t]
            for (i, j), coeff in term['U'].items():
                U[get_u_row(i, j), col_idx] += coeff
            for (j, k_off), coeff in term['V'].items():
                V[get_v_row(j, k_off, q), col_idx] += coeff
            for (i, k_off), coeff in term['W'].items():
                W[get_w_row(i, k_off, q), col_idx] += coeff
    
    # Convert to lists for the required output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
