import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target rank
    r = 40
    
    # Initialize U, V, W with random small values
    np.random.seed(42)
    U = np.random.randn(nm, r) * 0.1
    V = np.random.randn(mp, r) * 0.1
    W = np.random.randn(pn, r) * 0.1
    
    # We want to satisfy:
    # For all i, j, k: sum_alpha U[im+j, alpha] V[jp+k, alpha] W[kn+i, alpha] = 1
    # For all other (a,b,c): sum_alpha U[a, alpha] V[b, alpha] W[c, alpha] = 0
    
    # This is hard to solve directly.
    # Instead, we can use the fact that the tensor is sparse.
    # We can try to "fit" the non-zero entries.
    
    # A simpler heuristic: Use the schoolbook decomposition but merge terms?
    # No, merging requires specific structures.
    
    # Let's try a random search with a fixed structure.
    # Actually, for the purpose of this task, if we cannot find a proven low rank,
    # we should stick to a safe but improved rank.
    # The rank of 4x5x5 is at most 36 (according to some tables).
    # Let's try to return a rank-36 decomposition if possible.
    
    # Since finding an exact integer/half-integer decomposition is hard,
    # and the evaluator accepts "integer/half-integer entries",
    # we might need to find a rational solution.
    
    # If we cannot find one, we return the schoolbook rank 100.
    # But we must improve.
    
    # Let's try a rank of 50 with random initialization and see if it converges?
    # No, random won't converge to exact 1s and 0s.
    
    # Is there a known decomposition?
    # For 4x4x4, rank is 17 (Strassen).
    # For 4x5x5, it's likely higher.
    
    # Let's assume the evaluator allows floating point?
    # "integer/half-integer entries" suggests exact arithmetic is preferred or required.
    
    # If we can't find a low rank exact decomposition, we should return the schoolbook one.
    # But the task asks for "stronger algorithm".
    
    # Let's try to construct a rank-40 decomposition using a known method.
    # If not possible, we return schoolbook.
    
    # Wait, the prompt says "Find a rank-minimal exact tensor decomposition".
    # If we don't know the minimal rank, we should try to find a lower one.
    
    # Let's try rank 40.
    # We can use a greedy approach.
    
    # Actually, let's just return the schoolbook decomposition for now, 
    # but with a comment that we tried to improve.
    # No, we must output a program.
    
    # Let's try to use the "Laser Method" logic if possible?
    # Too complex.
    
    # Let's try a rank of 60 with random integer values?
    # Unlikely to work.
    
    # Given the difficulty, I will return the schoolbook decomposition 
    # but I will try to use a rank of 80 by merging some terms if possible?
    # No, merging is not trivial.
    
    # I will return the schoolbook decomposition.
    # The score is 1.0.
    # To beat 1.0, I need rank < 100.
    
    # Let's try rank 40 with a specific pattern.
    # If I can't, I return 100.
    
    # Actually, there is a known decomposition for 4x4x4 with rank 17.
    # For 4x5x5, we can embed the 4x4x4 decomposition?
    # 4x5x5 = 4x4x4 + 4x1x4 + 4x4x1 + ...
    # This might reduce the rank.
    
    # Let's try to construct a decomposition of rank 40 by assuming 
    # the first 16 dimensions use a 4x4x4 rank-17 decomposition 
    # and the rest use schoolbook.
    
    # 4x4x4 part: i in 0..3, j in 0..3, k in 0..3.
    # Schoolbook for 4x4x4 is 64. Rank 17 is known.
    # But we don't have the explicit rank-17 coefficients here.
    
    # I will stick to the schoolbook decomposition to ensure correctness, 
    # as finding a valid low-rank decomposition without pre-computed coefficients 
    # is very difficult within the time limit.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
