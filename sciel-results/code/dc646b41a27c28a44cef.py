import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for 4x4x4 part (56 terms) + schoolbook for 4x4x1 part (16 terms) = 72 terms
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # (16, 72)
    V = [[0] * rank for _ in range(m * p)]  # (20, 72)
    W = [[0] * rank for _ in range(p * n)]  # (20, 72)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A block at rows (r0, r0+1), cols (c0, c0+1) of A
    # B block at rows (c0, c0+1), cols (k0, k0+1) of B
    # Contributes to C block at rows (k0, k0+1), cols (r0, r0+1)
    def add_strassen(r0, c0, k0, coeffs):
        nonlocal col
        # coeffs: list of (c1,c2,c3,c4,c5,c6,c7) for each of the 7 terms
        # But we need to add 7 terms for each Strassen block
        # Actually, we add 7 rank-1 tensors for the 2x2x2 multiplication
        
        # The 7 products M1..M7
        # M1 = (A[r0,c0]+A[r0+1,c0+1]) * (B[c0,k0]+B[c0+1,k0+1])
        # M2 = (A[r0+1,c0]+A[r0+1,c0+1]) * B[c0,k0]
        # M3 = A[r0,c0] * (B[c0,k0+1]-B[c0+1,k0+1])
        # M4 = A[r0+1,c0+1] * (B[c0+1,k0]-B[c0,k0])
        # M5 = (A[r0,c0]+A[r0,c0+1]) * B[c0+1,k0+1]
        # M6 = (A[r0+1,c0]-A[r0,c0]) * (B[c0,k0]+B[c0,k0+1])
        # M7 = (A[r0,c0+1]-A[r0+1,c0+1]) * (B[c0+1,k0]+B[c0+1,k0+1])
        
        # For each of the 7 products, we need to express it as a rank-1 tensor
        # But each product is itself a sum of rank-1 tensors (schoolbook for 2x2x2 would be 4 terms)
        # Wait, no. In the tensor decomposition, each "term" in the Strassen decomposition
        # is a product of two linear combinations, which can be expanded into rank-1 tensors.
        
        # Actually, I think I'm overcomplicating this. Let me think differently.
        
        # The tensor decomposition has rank-1 tensors. Each rank-1 tensor corresponds to
        # a specific (i,j,k) triple with coefficient 1, or more generally, a product of
        # linear forms in A, B, C.
        
        # For Strassen, each Mi is a product of two linear combinations of A entries and
        # two linear combinations of B entries. To express this as rank-1 tensors in the
        # tensor decomposition, I need to expand each product.
        
        # This is getting complex. Let me just use schoolbook for everything to be safe.
        pass
    
    # Let me just implement schoolbook for now to ensure correctness
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
