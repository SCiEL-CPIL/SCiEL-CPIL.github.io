import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for the block i in {0,1}, j in {0,1}, k in {0,1}
    # and schoolbook for the rest.
    
    # Schoolbook rank for the rest
    # Total schoolbook is 75.
    # The block 2x2x2 has 8 terms.
    # We replace it with 7 terms.
    # So we have 67 schoolbook terms + 7 Strassen terms = 74 terms.
    
    rank = 74
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen block: i in {0,1}, j in {0,1}, k in {0,1}
    # Indices in U: i*m + j -> 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    # Indices in V: j*p + k -> 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    # Indices in W: k*n + i -> 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    
    # Strassen Factors
    # M1: (A00+A11)(B00+B11)
    # U: [1, 1] for i=0,1? No.
    # Let's use the known rank 7 decomposition.
    
    # U is 2x7 (local)
    # V is 2x7 (local)
    # W is 4x7 (local)
    
    # Local U
    # r=0: M1. A00+A11. U[0]=1, U[1]=1.
    # r=1: M2. A10+A11. U[0]=0, U[1]=1. (Wait, A10 is i=1, j=0. A11 is i=1, j=1.
    #      So U[1] should be 1 for both? No.
    #      M2 = (A10+A11)B00.
    #      This requires A10 and A11.
    #      U[1, r] must be 1. U[0, r] must be 0.
    #      V[0, r] must be 1 (for B00). V[1, r] must be 0.
    #      So U=[0,1], V=[1,0].
    #      Term: U[1]V[0]B[0,k] = 1*1*B[0,k] = B[0,k].
    #      Multiplied by U[1]=1? No.
    #      The term is U[i,r] V[j,r] B[j,k].
    #      For i=1, j=0: U[1]V[0]B[0,k] = 1*1*B[0,k].
    #      For i=1, j=1: U[1]V[1]B[1,k] = 1*0*B[1,k] = 0.
    #      This gives B[0,k].
    #      But we want (A10+A11)B00.
    #      This implies A10 and A11 are summed.
    #      But U[i,r] is just a scalar.
    #      It seems I cannot represent (A10+A11) with a single U[i,r] V[j,r] product.
    
    # I will fall back to schoolbook for all blocks to ensure correctness, 
    # but I will try to implement the 1-block Strassen if I can find the factors.
    # Since I am unsure of the factors, I will use the 74 rank strategy 
    # by manually constructing the 74 terms.
    
    # Actually, the safest bet is to output the schoolbook 75 rank, 
    # as I am not confident in the Strassen factor implementation.
    # But the prompt says "score 1.0 equals the seed program" (75).
    # And "Knowledge" says 74 was measured.
    # I will try to implement the 74 rank decomposition.
    
    # Let's just return schoolbook 75 for now to be safe, 
    # or try to find a simple improvement.
    
    # Wait, I can just return the schoolbook.
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
