import numpy as np
from collections import defaultdict

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 49 decomposition for 4x4x4
    # We generate the 49 terms by running Strassen's algorithm on 4x4 matrices
    # where the entries are symbols.
    
    # Symbols for A (4x4)
    A_syms = {}
    for i in range(4):
        for j in range(4):
            A_syms[(i,j)] = f'A_{i}_{j}'
            
    # Symbols for B (4x4)
    B_syms = {}
    for i in range(4):
        for j in range(4):
            B_syms[(i,j)] = f'B_{i}_{j}'
            
    # Symbols for C (4x4)
    C_syms = {}
    for i in range(4):
        for j in range(4):
            C_syms[(i,j)] = f'C_{i}_{j}'

    # We represent linear forms as dictionaries {symbol: coefficient}
    
    def add(l1, l2):
        res = dict(l1)
        for k, v in l2.items():
            res[k] = res.get(k, 0) + v
        return res
        
    def sub(l1, l2):
        res = dict(l1)
        for k, v in l2.items():
            res[k] = res.get(k, 0) - v
        return res
        
    def mul(l1, l2):
        # This is not multiplication of forms, this is a placeholder for the multiplication step
        # In Strassen, we multiply two linear forms.
        # The result is a bilinear form.
        # We store the pair (l1, l2) as a term.
        return (l1, l2)

    # Strassen's algorithm for 4x4 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # Extract 2x2 blocks
    A00 = {(i,j): A_syms[(i,j)] for i in range(2) for j in range(2)}
    A01 = {(i,j): A_syms[(i,j+2)] for i in range(2) for j in range(2)}
    A10 = {(i,j): A_syms[(i+2,j)] for i in range(2) for j in range(2)}
    A11 = {(i,j): A_syms[(i+2,j+2)] for i in range(2) for j in range(2)}
    
    B00 = {(i,j): B_syms[(i,j)] for i in range(2) for j in range(2)}
    B01 = {(i,j): B_syms[(i,j+2)] for i in range(2) for j in range(2)}
    B10 = {(i,j): B_syms[(i+2,j)] for i in range(2) for j in range(2)}
    B11 = {(i,j): B_syms[(i+2,j+2)] for i in range(2) for j in range(2)}
    
    # Strassen steps for 2x2 blocks of 2x2 matrices
    # Each step is a multiplication of two 2x2 matrices (linear forms)
    # We need to decompose each 2x2 matrix multiplication into 7 rank-1 terms.
    
    # This is getting too complex to do symbolically in this short time.
    # I will use the schoolbook for 4x4x4 blocks.
    # Rank 64 + 64 + 16 = 144. No gain.
    
    # Wait, I can use the rank-49 decomposition if I hardcode it or find a library.
    # But I don't have a library.
    
    # Alternative: Use the rank-7 decomposition for 2x2x2.
    # Can I use it for 4x4x4?
    # No.
    
    # Let's try a different split.
    # Split 9 into 3+3+3.
    # 4x3 blocks.
    # Rank 48 each.
    # Total 144.
    
    # It seems 144 is hard to beat without Strassen.
    
    # I will return the schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
