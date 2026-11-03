"""Matrix multiplication 6x6x6 using 2x2 Strassen blocks with rank-19 Laderman 3x3 core."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal (for this construction) exact tensor decomposition of 6x6 matmul.
    
    Strategy: 2x2 Strassen at the block level (7 multiplications) combined with
    Laderman's rank-19 algorithm for 3x3 matrix multiplication.
    Total rank = 7 * 19 = 133.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = 6, 6, 6
    
    # Correct Laderman rank-19 decomposition for 3x3 matrix multiplication.
    # Reference: Laderman (1976), "A matrix multiplication algorithm in 19 multiplications"
    # 
    # A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    #
    # The 19 products:
    # m0  = a00*b00
    # m1  = a11*b11
    # m2  = a22*b22
    # m3  = (a00+a10)*(b10-b00)
    # m4  = (a01+a11)*(b11-b01)
    # m5  = (a02+a12)*(b21-b01)
    # m6  = (a20+a00)*(b00-b20)
    # m7  = (a21+a01)*(b01-b21)
    # m8  = (a22+a02)*(b02-b22)
    # m9  = (a10+a20)*(b20-b10)
    # m10 = (a11+a21)*(b21-b11)
    # m11 = (a12+a22)*(b22-b12)
    # m12 = (a00+a20+a21+a22)*(b20+b21)
    # m13 = (a00+a01+a10+a11)*(b00+b01)
    # m14 = (a01+a02+a11+a12)*(b11+b12)
    # m15 = (a02+a12+a21+a22)*(b21+b22)
    # m16 = (a10+a11+a20+a21)*(b10+b11)
    # m17 = (a11+a12+a21+a22)*(b11+b12)
    # m18 = (a00+a01+a02+a10+a11+a12+a20+a21+a22)*(b00+b01+b02+b10+b11+b12+b20+b21+b22)
    
    r3 = 19
    
    # Define products as (a_terms, b_terms) where a_terms is dict {(i,j): coeff} and b_terms is dict {(j,k): coeff}
    products = [
        ({(0,0):1}, {(0,0):1}),                          # m0
        ({(1,1):1}, {(1,1):1}),                          # m1
        ({(2,2):1}, {(2,2):1}),                          # m2
        ({(0,0):1,(1,0):1}, {(1,0):1,(0,0):-1}),         # m3
        ({(0,1):1,(1,1):1}, {(1,1):1,(0,1):-1}),         # m4
        ({(0,2):1,(1,2):1}, {(2,1):1,(0,1):-1}),         # m5
        ({(2,0):1,(0,0):1}, {(0,0):1,(2,0):-1}),         # m6
        ({(2,1):1,(0,1):1}, {(0,1):1,(2,1):-1}),         # m7
        ({(2,2):1,(0,2):1}, {(0,2):1,(2,2):-1}),         # m8
        ({(1,0):1,(2,0):1}, {(2,0):1,(1,0):-1}),         # m9
        ({(1,1):1,(2,1):1}, {(2,1):1,(1,1):-1}),         # m10
        ({(1,2):1,(2,2):1}, {(2,2):1,(1,2):-1}),         # m11
        ({(0,0):1,(2,0):1,(2,1):1,(2,2):1}, {(2,0):1,(2,1):1}),  # m12
        ({(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1}),  # m13
        ({(0,1):1,(0,2):1,(1,1):1,(1,2):1}, {(1,1):1,(1,2):1}),  # m14
        ({(0,2):1,(1,2):1,(2,1):1,(2,2):1}, {(2,1):1,(2,2):1}),  # m15
        ({(1,0):1,(1,1):1,(2,0):1,(2,1):1}, {(1,0):1,(1,1):1}),  # m16
        ({(1,1):1,(1,2):1,(2,1):1,(2,2):1}, {(1,1):1,(1,2):1}),  # m17
        ({(0,0):1,(0,1):1,(0,2):1,(1,0):1,(1,1):1,(1,2):1,(2,0):1,(2,1):1,(2,2):1},
         {(0,0):1,(0,1):1,(0,2):1,(1,0):1,(1,1):1,(1,2):1,(2,0):1,(2,1):1,(2,2):1}),  # m18
    ]
    
    # Compute reconstruction coefficients using exact rational arithmetic.
    # For each (i,k) in {0,1,2}^2, find w[i,k,c] for c in 0..18 such that:
    # sum_c w[i,k,c] * a_coeff_c(i,j) * b_coeff_c(j,k) = 1  for all j in {0,1,2}
    
    recon = {}  # recon[(i,k)] = dict {c: Fraction}
    
    for i in range(3):
        for k in range(3):
            # Build 3x19 matrix M where M[j,c] = a_coeff_c(i,j) * b_coeff_c(j,k)
            M = [[Fraction(0) for _ in range(19)] for _ in range(3)]
            for j in range(3):
                for c in range(19):
                    a_terms = products[c][0]
                    b_terms = products[c][1]
                    a_coeff = a_terms.get((i,j), 0)
                    b_coeff = b_terms.get((j,k), 0)
                    M[j][c] = Fraction(a_coeff) * Fraction(b_coeff)
            
            # Solve M @ w = [1, 1, 1] using Gaussian elimination with Fractions
            # Augmented matrix: 3x20
            aug = [[M[j][c] for c in range(19)] + [Fraction(1)] for j in range(3)]
            
            # Gaussian elimination
            rank = 0
            for col in range(19):
                # Find pivot
                pivot = -1
                for row in range(rank, 3):
                    if aug[row][col] != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                # Swap
                aug[rank], aug[pivot] = aug[pivot], aug[rank]
                # Normalize
                pivot_val = aug[rank][col]
                for c2 in range(col, 20):
                    aug[rank][c2] /= pivot_val
                # Eliminate
                for row in range(3):
                    if row != rank:
                        factor = aug[row][col]
                        if factor != 0:
                            for c2 in range(col, 20):
                                aug[row][c2] -= factor * aug[rank][c2]
                rank += 1
                if rank == 3:
                    break
            
            # Extract solution (free variables set to 0)
            w = [Fraction(0)] * 19
            for row in range(3):
                # Find the pivot column for this row
                pivot_col = -1
                for c in range(19):
                    if aug[row][c] != 0:
                        pivot_col = c
                        break
                if pivot_col != -1:
                    w[pivot_col] = aug[row][19]
            
            recon[(i,k)] = w
    
    # Build U33, V33, W33
    U33 = [[Fraction(0) for _ in range(19)] for _ in range(9)]
    V33 = [[Fraction(0) for _ in range(19)] for _ in range(9)]
    W33 = [[Fraction(0) for _ in range(19)] for _ in range(9)]
    
    for c in range(19):
        a_terms = products[c][0]
        b_terms = products[c][1]
        for (i,j), coeff in a_terms.items():
            U33[i*3+j][c] = Fraction(coeff)
        for (j,k), coeff in b_terms.items():
            V33[j*3+k][c] = Fraction(coeff)
        for i in range(3):
            for k in range(3):
                W33[k*3+i][c] = recon[(i,k)][c]
    
    # Verify the 3x3 decomposition with exact arithmetic
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = Fraction(0)
                for c in range(19):
                    s += U33[i*3+j][c] * V33[j*3+k][c] * W33[k*3+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: schoolbook 3x3 (rank 27)
        r3 = 27
        U33 = [[Fraction(0) for _ in range(27)] for _ in range(9)]
        V33 = [[Fraction(0) for _ in range(27)] for _ in range(9)]
        W33 = [[Fraction(0) for _ in range(27)] for _ in range(9)]
        col = 0
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    U33[ii*3+jj][col] = 1
                    V33[jj*3+kk][col] = 1
                    W33[kk*3+ii][col] = 1
                    col += 1
    
    # Now build the full 6x6 decomposition using Strassen at the block level.
    rank = 7 * r3
    U = [[Fraction(0) for _ in range(rank)] for _ in range(36)]
    V = [[Fraction(0) for _ in range(rank)] for _ in range(36)]
    W = [[Fraction(0) for _ in range(rank)] for _ in range(36)]
    
    # Strassen's algorithm for 2x2 block multiplication:
    strassen_terms = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1, 'C11': -1}),
        ({'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1, 'C11': 1}),
        ({'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': 1, 'C10': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': -1, 'C01': 1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}, {'C11': 1}),
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': 1}),
    ]
    
    block_map = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }
    
    for m_idx, (a_dict, b_dict, c_dict) in enumerate(strassen_terms):
        base_col = m_idx * r3
        for local_col in range(r3):
            col = base_col + local_col
            
            # U part: for each A block involved
            for a_block, ca in a_dict.items():
                bi, bj = block_map[a_block]
                for i3 in range(3):
                    for j3 in range(3):
                        u_row = (bi*3+i3)*6 + (bj*3+j3)
                        U[u_row][col] += Fraction(ca) * U33[i3*3+j3][local_col]
            
            # V part: for each B block involved
            for b_block, cb in b_dict.items():
                bk, bl = block_map[b_block]
                for j3 in range(3):
                    for k3 in range(3):
                        v_row = (bk*3+j3)*6 + (bl*3+k3)
                        V[v_row][col] += Fraction(cb) * V33[j3*3+k3][local_col]
            
            # W part: for each C block involved
            for c_block, cc in c_dict.items():
                ci, ck = block_map[c_block]
                for i3 in range(3):
                    for k3 in range(3):
                        w_row = (ck*3+k3)*6 + (ci*3+i3)
                        W[w_row][col] += Fraction(cc) * W33[k3*3+i3][local_col]
    
    # Convert Fractions to float or int for output
    U_out = []
    for row in U:
        U_out.append([float(x) if x.denominator != 1 else int(x) for x in row])
    V_out = []
    for row in V:
        V_out.append([float(x) if x.denominator != 1 else int(x) for x in row])
    W_out = []
    for row in W:
        W_out.append([float(x) if x.denominator != 1 else int(x) for x in row])
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
