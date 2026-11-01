import time
import numpy as np

def construct():
    # Rank 28 decomposition using Strassen's algorithm on 2x2 blocks.
    # 2x4 * 4x4: split into four 2x2 blocks:
    # A = [A11 A12], B = [B11 B12] where each is 2x2
    #     [A21 A22]       [B21 B22]
    # C = [C11 C12] = [A11*B11 + A12*B21,  A11*B12 + A12*B22]
    #     [C21 C22]   [A21*B11 + A22*B21,  A21*B12 + A22*B22]
    #
    # Each 2x2 * 2x2 multiplication can be done with 7 multiplications (Strassen).
    # We have 4 such block multiplications, so naively 4*7 = 28.
    #
    # Indexing:
    # n=2, m=4, p=4
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (16, rank)
    # W has shape (p*n, rank) = (8, rank)
    #
    # U[i*m+j, r] = U[i*4+j, r]
    # V[j*p+k, r] = V[j*4+k, r]
    # W[k*n+i, r] = W[k*2+i, r]
    #
    # Block A11: rows 0,1 of A; cols 0,1 of A -> i in {0,1}, j in {0,1}
    # Block A12: rows 0,1 of A; cols 2,3 of A -> i in {0,1}, j in {2,3}
    # Block A21: rows 2,3 of A; cols 0,1 of A -> i in {2,3}, j in {0,1}
    # Block A22: rows 2,3 of A; cols 2,3 of A -> i in {2,3}, j in {2,3}
    #
    # Block B11: rows 0,1 of B; cols 0,1 of B -> j in {0,1}, k in {0,1}
    # Block B12: rows 0,1 of B; cols 2,3 of B -> j in {0,1}, k in {2,3}
    # Block B21: rows 2,3 of B; cols 0,1 of B -> j in {2,3}, k in {0,1}
    # Block B22: rows 2,3 of B; cols 2,3 of B -> j in {2,3}, k in {2,3}
    #
    # C11 = A11*B11: output at C[i, k] for i in {0,1}, k in {0,1} -> W[k*2+i]
    # C12 = A11*B12: output at C[i, k] for i in {0,1}, k in {2,3} -> W[k*2+i]
    # C21 = A21*B11: output at C[i, k] for i in {2,3}, k in {0,1} -> W[k*2+i]
    # C22 = A21*B12: output at C[i, k] for i in {2,3}, k in {2,3} -> W[k*2+i]
    #
    # Strassen for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For each 2x2x2x2 sub-problem, we need to express the 7 Strassen terms.
    # Each term is a product of a linear combination of A entries and a linear combination of B entries.
    #
    # Let me define the 7 terms for a generic 2x2*2x2 where:
    # A has entries indexed by (i,j) for i,j in {0,1}
    # B has entries indexed by (j,k) for j,k in {0,1}
    # C has entries indexed by (i,k) for i,k in {0,1}
    #
    # The 7 bilinear terms:
    # M1: (A00+A11)*(B00+B11) -> U: A00=1, A11=1; V: B00=1, B11=1
    # M2: (A10+A11)*B00 -> U: A10=1, A11=1; V: B00=1
    # M3: A00*(B01-B11) -> U: A00=1; V: B01=1, B11=-1
    # M4: A11*(B10-B00) -> U: A11=1; V: B10=1, B00=-1
    # M5: (A00+A01)*B11 -> U: A00=1, A01=1; V: B11=1
    # M6: (A10-A00)*(B00+B01) -> U: A10=1, A00=-1; V: B00=1, B01=1
    # M7: (A01-A11)*(B10+B11) -> U: A01=1, A11=-1; V: B10=1, B11=1
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    rank = 28
    U = [[0] * rank for _ in range(8)]   # n*m = 8
    V = [[0] * rank for _ in range(16)]  # m*p = 16
    W = [[0] * rank for _ in range(8)]   # p*n = 8
    
    # For each block multiplication, we define the mapping from local (i,j,k) to global indices
    # and set up the 7 Strassen terms.
    
    # Block (block_i, block_j, block_k) where:
    # block_i in {0,1}: which row block of A (0=rows 0-1, 1=rows 2-3)
    # block_j in {0,1}: which col block of A / row block of B (0=cols 0-1, 1=cols 2-3)
    # block_k in {0,1}: which col block of B (0=cols 0-1, 1=cols 2-3)
    
    # For block (bi, bj, bk):
    # A entries: A[bi*2+li, bj*2+lj] for li,lj in {0,1}
    # B entries: B[bj*2+lj, bk*2+lk] for lj,lk in {0,1}
    # C entries: C[bk*2+lk, bi*2+li] for lk,li in {0,1}
    
    # The 7 Strassen terms for local indices (li, lj, lk):
    # Each term t in {0,...,6} has:
    #   A coefficients: for each (li, lj), a coefficient a_li_lj
    #   B coefficients: for each (lj, lk), a coefficient b_lj_lk
    #   C contributions: for each (li, lk), a coefficient c_li_lk
    
    # M1: (A00+A11)*(B00+B11)
    # M2: (A10+A11)*B00
    # M3: A00*(B01-B11)
    # M4: A11*(B10-B00)
    # M5: (A00+A01)*B11
    # M6: (A10-A00)*(B00+B01)
    # M7: (A01-A11)*(B10+B11)
    
    # C contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 terms: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict mapping (li,lj) -> coeff
    # B_coeffs: dict mapping (lj,lk) -> coeff
    # C_coeffs: dict mapping (li,lk) -> coeff
    
    terms = []
    # M1
    terms.append({
        'A': {(0,0): 1, (1,1): 1},
        'B': {(0,0): 1, (1,1): 1},
        'C': {(0,0): 1, (1,1): 1}  # contributes to C00 and C11
    })
    # M2
    terms.append({
        'A': {(1,0): 1, (1,1): 1},
        'B': {(0,0): 1},
        'C': {(1,0): 1, (1,1): -1}  # contributes to C10 and C11
    })
    # M3
    terms.append({
        'A': {(0,0): 1},
        'B': {(0,1): 1, (1,1): -1},
        'C': {(0,1): 1, (1,1): 1}  # contributes to C01 and C11
    })
    # M4
    terms.append({
        'A': {(1,1): 1},
        'B': {(1,0): 1, (0,0): -1},
        'C': {(0,0): 1, (1,0): 1}  # contributes to C00 and C10
    })
    # M5
    terms.append({
        'A': {(0,0): 1, (0,1): 1},
        'B': {(1,1): 1},
        'C': {(0,0): -1, (0,1): 1}  # contributes to C00 and C01
    })
    # M6
    terms.append({
        'A': {(1,0): 1, (0,0): -1},
        'B': {(0,0): 1, (0,1): 1},
        'C': {(1,1): 1}  # contributes to C11
    })
    # M7
    terms.append({
        'A': {(0,1): 1, (1,1): -1},
        'B': {(1,0): 1, (1,1): 1},
        'C': {(0,0): 1}  # contributes to C00
    })
    
    # Now fill in U, V, W for each block and each Strassen term
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for t in range(7):
                    # Set U entries
                    for (li, lj), coeff in terms[t]['A'].items():
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        U[gi * 4 + gj][col] = coeff
                    
                    # Set V entries
                    for (lj, lk), coeff in terms[t]['B'].items():
                        gj = bj * 2 + lj
                        gk = bk * 2 + lk
                        V[gj * 4 + gk][col] = coeff
                    
                    # Set W entries
                    for (li, lk), coeff in terms[t]['C'].items():
                        gi = bi * 2 + li
                        gk = bk * 2 + lk
                        W[gk * 2 + gi][col] = coeff
                    
                    col += 1
    
    # Verify
    valid = True
    for i in range(2):
        for j in range(4):
            for k in range(4):
                s = 0
                for r in range(28):
                    s += U[i * 4 + j][r] * V[j * 4 + k][r] * W[k * 2 + i][r]
                if abs(s - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = 32
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]
    column = 0
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][column] = 1
                V[j * 4 + k][column] = 1
                W[k * 2 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
