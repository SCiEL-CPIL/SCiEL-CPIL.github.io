DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = 3, 5, 6
    
    # For 3x5x6 matrix multiplication, we need to decompose the tensor
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The trivial rank is n*m*p = 90
    # We can try to use Strassen-like methods for sub-blocks
    #
    # Let's decompose by splitting the middle dimension m=5 into 2+3
    # Or split n=3 into 1+2, or p=6 into 2+4, etc.
    
    # Actually, let's try a simpler approach first: use the fact that
    # we can group columns of A and rows of B.
    
    # For now, let's implement a rank-90 solution but structured clearly
    # and see if we can improve.
    
    # One approach: split m into m1=2 and m2=3
    # C = A1*B1 + A2*B2 where A1 is 3x2, A2 is 3x3, B1 is 2x6, B2 is 3x6
    
    # For 3x2x6: rank at most 18 (trivial)
    # For 3x3x6: rank at most 54 (trivial)
    # Total: 72
    
    # But we can do better with Strassen on the 3x3x6 part?
    # 3x3 can be done with rank 19 (or 23 with Strassen variant)
    # Actually 3x3x3 has rank 19 known.
    # So 3x3x6 could be done with rank 19*2 = 38? No, that's not right.
    
    # Let me think differently. The tensor is T_{i,j,k} = delta_{i,j} for matrix mult
    # where i in 0..2, j in 0..4, k in 0..5
    
    # Let's just use the trivial decomposition for now but ensure correctness.
    # Rank = 90
    
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
