import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    # Target: 2x3 * 3x6 -> 2x6
    # Decomposition via Kronecker: M_{2,3,6} ~ M_{2,3,3} x M_{1,1,2}
    # Rank(M_{1,1,2}) = 1
    # Rank(M_{2,3,3}) = 18 (schoolbook)
    # Total Rank = 18 * 1 = 18
    
    n, m, p = DIMENSIONS
    # Sub-problem 1: 2x3 * 3x3
    n1, m1, p1 = 2, 3, 3
    # Sub-problem 2: 1x1 * 1x2
    n2, m2, p2 = 1, 1, 2
    
    rank1 = n1 * m1 * p1
    rank2 = 1
    
    rank = rank1 * rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over the combined rank index
    # c = c1 * rank2 + c2
    # Since rank2 = 1, c = c1
    
    for c1 in range(rank1):
        c = c1 * rank2 + 0 # c2 is always 0
        
        # U part
        # U_prod[i, c] = U1[i1, c1] * U2[i2, c2]
        # i = i1 * n2 + i2 = i1 * 1 + 0 = i1
        # i1 ranges 0..n1-1 (0..1)
        # j1 ranges 0..m1-1 (0..2)
        # U1 index is i1 * m1 + j1
        for i1 in range(n1):
            for j1 in range(m1):
                # Schoolbook U1: U1[i1*m1+j1, c1] = 1 if (i1,j1) matches c1
                # c1 corresponds to (i1, j1, k1) in schoolbook
                # We need to extract i1, j1 from c1 for U1
                # In schoolbook, c1 = i1 * (m1 * p1) + j1 * p1 + k1
                i1_c = c1 // (m1 * p1)
                j1_c = (c1 // p1) % m1
                k1_c = c1 % p1
                
                if i1 == i1_c and j1 == j1_c:
                    U[i1 * m + j1][c] = 1 # U2[0,0] is 1
                    
        # V part
        # V_prod[j, k, c] = V1[j1, k1, c1] * V2[j2, k2, c2]
        # j = j1 * m2 + j2 = j1
        # k = k1 * p2 + k2
        # V1 index is j1 * p1 + k1
        # V2 index is j2 * p2 + k2 = 0 * 2 + k2 = k2
        for j1 in range(m1):
            for k1 in range(p1):
                # Schoolbook V1: V1[j1*p1+k1, c1] = 1 if (j1,k1) matches c1
                j1_c = c1 // p1 % m1
                k1_c = c1 % p1
                
                if j1 == j1_c and k1 == k1_c:
                    # V1 is 1
                    # V2[k2, 0] is 1 for all k2
                    # k = k1 * 2 + k2
                    for k2 in range(p2):
                        k = k1 * p2 + k2
                        V[j1 * p + k][c] = 1 # V2[k2,0] is 1
                        
        # W part
        # W_prod[k, i, c] = W1[k1, i1, c1] * W2[k2, i2, c2]
        # i = i1 * n2 + i2 = i1
        # k = k1 * p2 + k2
        # W1 index is k1 * n1 + i1
        # W2 index is k2 * n2 + i2 = k2
        for k1 in range(p1):
            for i1 in range(n1):
                # Schoolbook W1: W1[k1*n1+i1, c1] = 1 if (k1,i1) matches c1
                k1_c = c1 // n1
                i1_c = c1 % n1
                
                if k1 == k1_c and i1 == i1_c:
                    # W1 is 1
                    # W2[k2, 0] is 1 for all k2
                    for k2 in range(p2):
                        k = k1 * p2 + k2
                        W[k * n + i1][c] = 1 # W2[k2,0] is 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
