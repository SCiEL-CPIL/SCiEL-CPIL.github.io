import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # We will construct a rank 58 decomposition using two Strassen blocks.
    # Block 1: i in {0, 1}, j in {0, 1}, k in {0, 1}
    # Block 2: i in {0, 1}, j in {2, 3}, k in {2, 3}
    # The rest of the terms are handled by schoolbook.
    
    # Total rank = (60 - 8 - 8) + 7 + 7 = 58
    rank = 58
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    # Helper to add a Strassen block for i_indices, j_indices, k_indices
    # Indices must be size 2
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # Strassen 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Map local (0,0) to global (i0,j0), (1,0) to (i1,j0), etc.
        # A local 00 -> A[i0, j0]
        # A local 10 -> A[i1, j0]
        # A local 01 -> A[i0, j1]
        # A local 11 -> A[i1, j1]
        
        # B local 00 -> B[j0, k0]
        # B local 10 -> B[j1, k0]
        # B local 01 -> B[j0, k1]
        # B local 11 -> B[j1, k1]
        
        # W local 00 -> W[k0, i0]
        # W local 10 -> W[k0, i1]
        # W local 01 -> W[k1, i0]
        # W local 11 -> W[k1, i1]
        
        # M1
        U[i0 * m + j0][col] += 1
        U[i1 * m + j1][col] += 1
        V[j0 * p + k0][col] += 1
        V[j1 * p + k1][col] += 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i1][col] += 1
        W[k1 * n + i0][col] += 1
        W[k1 * n + i1][col] += 1
        col += 1
        
        # M2
        U[i1 * m + j0][col] += 1
        U[i1 * m + j1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i1][col] += 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j1 * p + k0][col] += 1
        V[j1 * p + k1][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k1 * n + i0][col] -= 1
        col += 1
        
        # M4
        U[i1 * m + j1][col] += 1
        V[j0 * p + k1][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k1 * n + i1][col] += 1
        W[k0 * n + i1][col] -= 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j1][col] += 1
        V[j1 * p + k1][col] += 1
        W[k1 * n + i0][col] += 1
        col += 1
        
        # M6
        U[i1 * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k1][col] += 1
        W[k0 * n + i1][col] += 1
        W[k1 * n + i1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j1][col] += 1
        U[i1 * m + j1][col] -= 1
        V[j1 * p + k0][col] += 1
        V[j1 * p + k1][col] += 1
        W[k0 * n + i0][col] += 1
        W[k1 * n + i0][col] += 1
        col += 1

    # Add Strassen Block 1: i in {0, 1}, j in {0, 1}, k in {0, 1}
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Add Strassen Block 2: i in {0, 1}, j in {2, 3}, k in {2, 3}
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term was covered by Strassen
                if (i in [0, 1] and j in [0, 1] and k in [0, 1]):
                    continue
                if (i in [0, 1] and j in [2, 3] and k in [2, 3]):
                    continue
                add_schoolbook(i, j, k)
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
