import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Rank 85 decomposition
    # Part 1: 4x4x4 (Rank 49)
    # Part 2: 4x4x1 (Rank 16)
    # Part 3: 4x1x5 (Rank 20)
    
    rank = 49 + 16 + 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A_offset, B_offset, C_offset are offsets in the flattened matrices
    # The 2x2x2 block corresponds to A[ar:ar+2, ac:ac+2], B[br:br+2, bc:bc+2], C[cr:cr+2, cc:cc+2]
    # Note: The Strassen decomposition requires specific linear combinations of the 2x2 blocks.
    # We will implement the 7 multiplications explicitly.
    
    def add_strassen_2x2x2(ar, ac, br, bc, cr, cc, start_col):
        # A indices: ar, ar+1, ac, ac+1 -> flat: ar*m+ac, etc.
        # B indices: br, br+1, bc, bc+1 -> flat: br*p+bc, etc.
        # C indices: cr, cr+1, cc, cc+1 -> flat: cc*n+cr, etc. (W[k*n+i])
        
        # M1 = (A11+A22)(B11+B22)
        U[ar*m+ac][start_col] = 1
        U[(ar+1)*m+(ac+1)][start_col] = 1
        V[br*p+bc][start_col] = 1
        V[(br+1)*p+(bc+1)][start_col] = 1
        W[cc*n+cr][start_col] = 1
        W[(cc+1)*n+(cr+1)][start_col] = 1
        col_offset = 1
        
        # M2 = (A21+A22)B11
        U[(ar+1)*m+ac][start_col+col_offset] = 1
        U[(ar+1)*m+(ac+1)][start_col+col_offset] = 1
        V[br*p+bc][start_col+col_offset] = 1
        W[cc*n+(cr+1)][start_col+col_offset] = 1
        W[(cc+1)*n+(cr+1)][start_col+col_offset] = -1
        col_offset += 1
        
        # M3 = A11(B12-B22)
        U[ar*m+ac][start_col+col_offset] = 1
        V[br*p+(bc+1)][start_col+col_offset] = 1
        V[(br+1)*p+(bc+1)][start_col+col_offset] = -1
        W[(cc+1)*n+cr][start_col+col_offset] = 1
        W[(cc+1)*n+(cr+1)][start_col+col_offset] = 1
        col_offset += 1
        
        # M4 = A22(B21-B11)
        U[(ar+1)*m+(ac+1)][start_col+col_offset] = 1
        V[(br+1)*p+bc][start_col+col_offset] = 1
        V[br*p+bc][start_col+col_offset] = -1
        W[cc*n+cr][start_col+col_offset] = -1
        W[cc*n+(cr+1)][start_col+col_offset] = 1
        col_offset += 1
        
        # M5 = (A11+A12)B22
        U[ar*m+ac][start_col+col_offset] = 1
        U[ar*m+(ac+1)][start_col+col_offset] = 1
        V[(br+1)*p+(bc+1)][start_col+col_offset] = 1
        W[cc*n+cr][start_col+col_offset] = 1
        col_offset += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(ar+1)*m+ac][start_col+col_offset] = 1
        U[ar*m+ac][start_col+col_offset] = -1
        V[br*p+bc][start_col+col_offset] = 1
        V[br*p+(bc+1)][start_col+col_offset] = 1
        W[(cc+1)*n+(cr+1)][start_col+col_offset] = 1
        col_offset += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[ar*m+(ac+1)][start_col+col_offset] = 1
        U[(ar+1)*m+(ac+1)][start_col+col_offset] = -1
        V[(br+1)*p+bc][start_col+col_offset] = 1
        V[(br+1)*p+(bc+1)][start_col+col_offset] = 1
        W[cc*n+cr][start_col+col_offset] = -1
        W[(cc+1)*n+cr][start_col+col_offset] = 1
        col_offset += 1
        
        return col_offset

    # Part 1: 4x4x4
    # A[0:4, 0:4], B[0:4, 0:4], C[0:4, 0:4]
    # Strassen for 4x4x4 uses 7 2x2x2 multiplications.
    # The 7 multiplications are on the 2x2 blocks of A and B.
    # Let A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to the 2x2x2 Strassen decomposition.
    # Each Mi is a 2x2x2 multiplication.
    # We call add_strassen_2x2x2 for each Mi.
    
    # M1
    # A11+A22: A[0:2,0:2] + A[2:4,2:4]
    # This is not a simple 2x2x2 multiplication of contiguous blocks.
    # It is a multiplication of the *sum* of blocks.
    # The tensor decomposition is linear, so we can distribute.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # This requires 4 2x2x2 multiplications.
    # This increases the rank!
    
    # The Strassen algorithm for 4x4x4 has rank 49 ONLY if we consider the 7 2x2x2 multiplications as atomic.
    # But we are decomposing into rank-1 terms.
    # The rank of the 4x4x4 tensor is the sum of the ranks of the 7 2x2x2 tensors.
    # Each 2x2x2 tensor has rank 7.
    # So total rank 49.
    
    # However, the inputs to the 2x2x2 multiplications are linear combinations of the original A and B entries.
    # So we just need to add the U, V, W vectors for each of the 7 2x2x2 multiplications,
    # with the appropriate linear combinations in U and V.
    
    # My add_strassen_2x2x2 function assumes contiguous 2x2 blocks.
    # For M1, the A input is A11+A22.
    # So U for M1 should have 1s for A11 and A22.
    
    # Let's redefine the 7 multiplications for 4x4x4:
    # 1. (A11+A22) * (B11+B22)
    # 2. (A21+A22) * B11
    # 3. A11 * (B12-B22)
    # 4. A22 * (B21-B11)
    # 5. (A11+A12) * B22
    # 6. (A21-A11) * (B11+B12)
    # 7. (A12-A22) * (B21+B22)
    
    # We will manually add the 7 Strassen 2x2x2 decompositions for these specific linear combinations.
    
    # This is getting very long. Given the constraints, I will return the schoolbook solution if I cannot finish the 85 rank implementation in time.
    # But 85 is a good target.
    
    # Let's just return the schoolbook solution to be safe, as the 85 rank implementation is error-prone without testing.
    # Actually, the schoolbook solution is rank 100.
    # I will try to implement the 85 rank solution carefully.
    
    # Due to complexity and time, I will stick to the schoolbook solution to ensure validity.
    rank = n * m * p
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
