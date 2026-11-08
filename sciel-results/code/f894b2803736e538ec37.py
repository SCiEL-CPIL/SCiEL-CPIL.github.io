import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # Part 1: 2x2x10 using 5 Strassen blocks (rank 35)
    # Part 2: 2x1x10 using schoolbook (rank 20)
    
    total_rank = 35 + 20
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1..M7
    # U, V, W contributions
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is getting messy to hardcode. I'll generate it programmatically in the function.
    
    # Let's define the Strassen terms explicitly in the loop.
    
    for block_k in range(5):
        # B local indices offset
        b_offset = block_k * 2
        # C local indices offset
        c_offset = block_k * 2
        
        # M1: (A00+A11)(B00+B11)
        # U: A00+1, A11+1
        # V: B00+1, B11+1
        # W: C00+1, C11+1
        U[0][col] += 1
        U[3][col] += 1
        V[b_offset][col] += 1
        V[b_offset+2][col] += 1
        W[c_offset][col] += 1
        W[c_offset+3][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        # U: A10+1, A11+1
        # V: B00+1
        # W: C10+1
        U[2][col] += 1
        U[3][col] += 1
        V[b_offset][col] += 1
        W[c_offset+2][col] += 1
        col += 1
        
        # M3: A00(B01-B11)
        # U: A00+1
        # V: B01+1, B11-1
        # W: C01+1
        U[0][col] += 1
        V[b_offset+1][col] += 1
        V[b_offset+2][col] -= 1
        W[c_offset+1][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        # U: A11+1
        # V: B10+1, B00-1
        # W: C00+1, C10+1
        U[3][col] += 1
        V[b_offset+2][col] += 1
        V[b_offset][col] -= 1
        W[c_offset][col] += 1
        W[c_offset+2][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        # U: A00+1, A01+1
        # V: B11+1
        # W: C00-1, C01+1
        U[0][col] += 1
        U[1][col] += 1
        V[b_offset+2][col] += 1
        W[c_offset][col] -= 1
        W[c_offset+1][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        # U: A10+1, A00-1
        # V: B00+1, B01+1
        # W: C11+1
        U[2][col] += 1
        U[0][col] -= 1
        V[b_offset][col] += 1
        V[b_offset+1][col] += 1
        W[c_offset+3][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        # U: A01+1, A11-1
        # V: B10+1, B11+1
        # W: C00+1, C11-1
        U[1][col] += 1
        U[3][col] -= 1
        V[b_offset+2][col] += 1
        V[b_offset+3][col] += 1
        W[c_offset][col] += 1
        W[c_offset+3][col] -= 1
        col += 1

    # Part 2: 2x1x10 Schoolbook
    # A col 2 (index 2 in A, global U index i*3+2)
    # B row 2 (index 2 in B, global V index 2*10+k)
    # C all (global W index k*2+i)
    
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
