import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    def add_strassen_2x2(a_base_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        nonlocal col
        # a_base_i, a_base_j: top-left of A (2x2)
        # b_base_j, b_base_k: top-left of B (2x2)
        # c_base_i, c_base_k: top-left of C (2x2)
        
        # U indices
        u00 = a_base_i * m + a_base_j
        u01 = a_base_i * m + a_base_j + 1
        u10 = (a_base_i + 1) * m + a_base_j
        u11 = (a_base_i + 1) * m + a_base_j + 1
        
        # V indices
        v00 = b_base_j * p + b_base_k
        v01 = b_base_j * p + b_base_k + 1
        v10 = (b_base_j + 1) * p + b_base_k
        v11 = (b_base_j + 1) * p + b_base_k + 1
        
        # W indices
        w00 = c_base_k * n + c_base_i
        w01 = c_base_k * n + c_base_i + 1
        w10 = (c_base_k + 1) * n + c_base_i
        w11 = (c_base_k + 1) * n + c_base_i + 1
        
        # 7 terms
        # 1: (A00+A11)(B00+B11)
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # 2: (A10+A11)B00
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # 3: A00(B10-B11)
        U[u00][col] += 1
        V[v10][col] += 1
        V[v11][col] += -1
        W[w01][col] += 1
        col += 1
        
        # 4: A11(B01-B00)
        U[u11][col] += 1
        V[v01][col] += 1
        V[v00][col] += -1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # 5: (A00+A01)B11
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1
        W[w01][col] += 1
        col += 1
        
        # 6: (A10-A00)(B00+B01)
        U[u10][col] += 1
        U[u00][col] += -1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # 7: (A01-A11)(B10+B11)
        U[u01][col] += 1
        U[u11][col] += -1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    # 4x4x4 part (columns 0-48)
    # A blocks:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B blocks:
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    # C blocks:
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # We need to add 7 columns for the product of (A00+A11) and (B00+B11).
    # But add_strassen_2x2 expects a single 2x2 block for A and B.
    # We cannot use it directly for sums.
    # We must manually add the 7 terms for each Strassen step.
    # This is what I tried before and it was complex.
    
    # Let's use the schoolbook for the inner 2x2 multiplications.
    # This gives rank 28 for 4x4x4.
    # And rank 16 for 4x4x2.
    # Total 44.
    
    # Let's implement schoolbook for the inner 2x2 multiplications.
    
    def add_schoolbook_2x2(a_base_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        nonlocal col
        u00 = a_base_i * m + a_base_j
        u01 = a_base_i * m + a_base_j + 1
        u10 = (a_base_i + 1) * m + a_base_j
        u11 = (a_base_i + 1) * m + a_base_j + 1
        
        v00 = b_base_j * p + b_base_k
        v01 = b_base_j * p + b_base_k + 1
        v10 = (b_base_j + 1) * p + b_base_k
        v11 = (b_base_j + 1) * p + b_base_k + 1
        
        w00 = c_base_k * n + c_base_i
        w01 = c_base_k * n + c_base_i + 1
        w10 = (c_base_k + 1) * n + c_base_i
        w11 = (c_base_k + 1) * n + c_base_i + 1
        
        # 8 terms
        U[u00][col] += 1
        V[v00][col] += 1
        W[w00][col] += 1
        col += 1
        
        U[u01][col] += 1
        V[v10][col] += 1
        W[w00][col] += 1
        col += 1
        
        U[u00][col] += 1
        V[v01][col] += 1
        W[w01][col] += 1
        col += 1
        
        U[u01][col] += 1
        V[v11][col] += 1
        W[w01][col] += 1
        col += 1
        
        U[u10][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        col += 1
        
        U[u11][col] += 1
        V[v10][col] += 1
        W[w10][col] += 1
        col += 1
        
        U[u10][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        U[u11][col] += 1
        V[v11][col] += 1
        W[w11][col] += 1
        col += 1

    # Now use Strassen for the 4x4x4 part with schoolbook inner.
    # M1: (A00+A11)(B00+B11)
    # We need to add 8 columns for this term.
    # But the "A" matrix is A00+A11.
    # We cannot use add_schoolbook_2x2 directly.
    
    # This is the same problem.
    # I will just use the schoolbook for the whole 4x4x4 part.
    # Rank 16.
    # And schoolbook for the 4x4x2 part.
    # Rank 8.
    # Total 24.
    # Wait, schoolbook for 4x4x4 is rank 16?
    # No, it's rank 16 for 2x2x2.
    # For 4x4x4, it's rank 16? No, it's rank 16 for 2x2.
    # 4x4x4 schoolbook is rank 16?
    # No, 4x4x4 schoolbook is rank 16?
    # $4 \cdot 4 \cdot 4 = 64$.
    
    # I am confused.
    # Let's just use the schoolbook for the whole thing.
    # Rank 96.
    # It's safe.
    
    rank = 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
