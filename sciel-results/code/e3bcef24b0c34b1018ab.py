import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 99
    rank = 99
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Define the 6 Strassen blocks
    # i-pair: {0, 1}
    # j-pairs: {0, 1}, {2, 3}
    # k-pairs: {0, 1}, {2, 3}, {4, 5}
    
    i_pair = [0, 1]
    j_pairs = [[0, 1], [2, 3]]
    k_pairs = [[0, 1], [2, 3], [4, 5]]
    
    # Strassen terms definition
    # Each term is a list of (local_a, local_b, coeff_u), (local_b, local_c, coeff_v), and contributions to W
    # W contributions are (local_c, local_a, coeff_w)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A01+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # W formulas:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M4
    # C10 = M3 + M7
    # C11 = M1 - M3 + M6 + M7
    
    strassen_terms = [
        {
            "u": [(0,0,1), (1,1,1)],
            "v": [(0,0,1), (1,1,1)],
            "w": [(0,0,1), (1,1,1)] # M1 contributes to C00 and C11
        },
        {
            "u": [(0,1,1), (1,1,1)],
            "v": [(0,0,1)],
            "w": [(0,1,1)] # M2 contributes to C01
        },
        {
            "u": [(0,0,1)],
            "v": [(1,0,1), (1,1,-1)],
            "w": [(1,0,1)] # M3 contributes to C10
        },
        {
            "u": [(1,1,1)],
            "v": [(0,1,1), (0,0,-1)],
            "w": [(0,0,-1), (0,1,1)] # M4 contributes to -C00 and C01
        },
        {
            "u": [(0,0,1), (0,1,1)],
            "v": [(1,1,1)],
            "w": [(0,0,1)] # M5 contributes to C00
        },
        {
            "u": [(1,0,1), (0,0,-1)],
            "v": [(0,0,1), (1,0,1)],
            "w": [(1,1,1)] # M6 contributes to C11
        },
        {
            "u": [(0,1,1), (1,1,-1)],
            "v": [(1,1,1), (0,1,1)],
            "w": [(0,0,-1), (1,0,1), (1,1,1)] # M7 contributes to -C00, C10, C11
        }
    ]
    
    for j_pair in j_pairs:
        for k_pair in k_pairs:
            # For each of the 7 Strassen terms
            for term in strassen_terms:
                # Add U contributions
                for (la, lb, coeff) in term["u"]:
                    gi = i_pair[la]
                    gj = j_pair[lb]
                    U[gi * m + gj, col] += coeff
                
                # Add V contributions
                for (lb, lc, coeff) in term["v"]:
                    gj = j_pair[lb]
                    gk = k_pair[lc]
                    V[gj * p + gk, col] += coeff
                
                # Add W contributions
                for (lc, la, coeff) in term["w"]:
                    gk = k_pair[lc]
                    gi = i_pair[la]
                    W[gk * n + gi, col] += coeff
                
                col += 1
    
    # Add schoolbook terms for remaining indices
    # Indices not covered by Strassen blocks:
    # i=2 (any j, k)
    # j=4 (any i, k)
    # k=6 (any i, j)
    # But we must avoid double counting.
    # The Strassen blocks cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    # So any index with i=2 OR j=4 OR k=6 is NOT covered.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this index is covered by a Strassen block
                covered = False
                if i in i_pair and j in j_pairs[0] + j_pairs[1] and k in k_pairs[0] + k_pairs[1] + k_pairs[2]:
                    # It is in the bounding box of the Strassen blocks
                    # But we need to check if it's actually in one of the 6 specific blocks
                    # The 6 blocks are the Cartesian product of the pairs.
                    # So if i is in i_pair, j is in one of j_pairs, k is in one of k_pairs, it IS covered.
                    covered = True
                
                if not covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
