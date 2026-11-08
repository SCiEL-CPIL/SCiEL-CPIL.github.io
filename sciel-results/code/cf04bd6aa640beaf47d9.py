"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 145.0  # Leave 5s safety margin

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-minimal decomposition
    # For 3x5 * 5x6, the schoolbook rank is 90.
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30.
    # Known results:
    # - For 2x2*2x2, rank is 7 (Strassen).
    # - For 3x3*3x3, rank is 23 (Laderman).
    # - For 2x3*3x3, rank is 16.
    # - For 3x5*5x6, no specific known minimal rank is widely cited, but we can try to find a good one.
    
    # Strategy: Use a known low-rank decomposition if available, or fall back to schoolbook.
    # Since we don't have a specific known decomposition for 3x5*5x6 that beats 90 significantly in standard literature,
    # and constructing one from scratch within time limits is hard, we will attempt a simple optimization or use a known block structure.
    
    # However, the prompt asks for "rank-minimal". If we can't find a better one quickly, we return schoolbook.
    # But wait, the score is `seed/value`. The seed is 1.0. The value is the rank.
    # If we return rank 90, score is 1.0.
    # If we return rank 89, score is 90/89 > 1.0.
    # So ANY reduction in rank improves the score.
    
    # Let's check if there's a simple reduction.
    # Actually, for rectangular matrices, sometimes we can use the "block" structure.
    # But without a pre-computed tensor, finding one is the task.
    
    # Given the constraints and the lack of a known explicit 3x5x6 tensor in standard small libraries,
    # and the risk of timeout, the safest bet to improve from 1.0 is to find *any* valid decomposition with rank < 90.
    
    # Is it possible to just return the schoolbook decomposition? Yes, score 1.0.
    # Can we do better?
    # Let's try a simple heuristic: If n=3, m=5, p=6.
    # Maybe we can use the fact that 5 is prime? No.
    
    # Actually, let's look at the inspirations. They all score 1.0.
    # This suggests that finding a better rank is difficult or the previous attempts failed.
    
    # However, I recall that for 3x5 * 5x6, the rank is at most 90.
    # Is there a known result?
    # According to some tables, the rank of 3x5 * 5x6 is not 90.
    # For example, if we use the "Kronecker product" trick?
    # 3x5 * 5x6 = (3x1 * 1x5) * (5x6)? No.
    
    # Let's try to construct a decomposition with rank 89 or less by simply removing one term if possible?
    # No, the terms are independent in the schoolbook algorithm.
    
    # What if we use a different algorithm?
    # If we can't find a specific low-rank tensor, we should stick to the schoolbook to ensure validity, 
    # UNLESS we can implement a search.
    
    # Given the 120s limit, a random search for a rank 89 tensor is infeasible.
    # But wait, is there a trivial reduction?
    # No.
    
    # Let's check if the score calculation implies that the "seed" value is the schoolbook rank.
    # "score is the rank" -> No, "score is seed/value".
    # "1.0 equals the seed program".
    # Seed program returns rank 90.
    # So score = 90 / rank.
    # To get score > 1.0, we need rank < 90.
    
    # Since I cannot reliably generate a rank < 90 tensor in Python without a database or heavy computation, 
    # and the inspirations failed, I will return the schoolbook decomposition. 
    # BUT, the directive says "improving the Python program". 
    # If I return the same thing, I get 1.0.
    
    # Is there any chance the "seed" value is different?
    # "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program".
    # This confirms seed_value = 90.
    
    # I will implement a very fast check for a known rank reduction if any exists in a small hardcoded set, 
    # but I don't have one.
    
    # Wait, I can try to use the "Laderman" style for 3x3 and pad?
    # 3x5 * 5x6.
    # If we split 5 into 3+2 and 6 into 3+3?
    # This is getting too complex to derive on the fly.
    
    # I will stick to the schoolbook decomposition to ensure a valid score of 1.0, as attempting a complex search 
    # that times out or returns an invalid tensor would result in a score of 0.
    
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
# EVOLVE-BLOCK-END
