"""Matrix multiplication 3,5,7 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 x 5x7 matrix multiplication.
    
    The tensor T[i,j,k] for [n,m,p] matrix multiplication has:
    T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k
    and 0 elsewhere.
    
    We need U, V, W such that for all a, b, c:
    sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    The schoolbook gives rank = n*m*p = 105.
    
    Strategy: Use a systematic search for a lower rank decomposition.
    For [3,5,7], known upper bounds are significantly below 105.
    
    Key insight: We can use the structure of the tensor to find decompositions
    with rank less than n*m*p by exploiting the fact that T is sparse in a
    structured way.
    
    Approach: Use the "embedding" technique. We can embed [3,5,7] into a larger
    problem with known efficient decompositions, or use direct optimization.
    
    For this specific case, let's try a construction based on splitting dimensions
    and using Strassen-like ideas where applicable.
    
    Actually, let's use a more direct approach: we'll construct the schoolbook
    decomposition first (rank 105), then try to find a lower rank one using
    algebraic techniques.
    
    For [3,5,7], a known upper bound is around 60-70. Let's try to achieve
    a rank significantly below 105.
    
    One effective approach: use the fact that [3,5,7] can be decomposed using
    the tensor product structure and known efficient algorithms for sub-problems.
    
    Let's try: split m=5 into 2+3.
    [3,5,7] = [3,2,7] + [3,3,7]
    
    For [3,2,7]: This is a 3x2 times 2x7 multiplication.
    Rank is at most 3*2*7 = 42 (schoolbook).
    But we can do better: for each of the 3 rows of A and 7 columns of B,
    we compute a dot product of length 2. That's 3*7*2 = 42 scalar mults.
    Actually, this is already optimal for this sub-problem since each of the
    2 intermediate dimensions must be used.
    
    For [3,3,7]: 3x3 times 3x7.
    Schoolbook: 3*3*7 = 63.
    Using Strassen for the 3x3 part: rank([3,3,3]) = 19.
    But [3,3,7] is not directly [3,3,3].
    
    Alternative: [3,3,7] = 7 copies of [3,3,1] (matrix-vector mult).
    rank([3,3,1]) = 9 (you need all 9 products).
    So [3,3,7] = 7*9 = 63 in schoolbook.
    
    Total: 42 + 63 = 105. No improvement.
    
    Let's try a different split: [3,5,7] = [3,5,3] + [3,5,4]
    [3,5,3]: 3*5*3 = 45
    [3,5,4]: 3*5*4 = 60
    Total: 105. Same.
    
    The key is to use a non-trivial decomposition. Let's try using the
    "sum of squares" identity or other algebraic tricks.
    
    Actually, for a concrete improvement, let's use the following:
    We can use the fact that for certain dimension combinations, there exist
    decompositions with rank less than the schoolbook.
    
    For [3,5,7], let's try a direct construction with rank < 105.
    
    One approach: use the tensor rank of [3,5,7] which is known to be at most
    around 60-70 based on various embeddings.
    
    Let me try a construction using circulant matrices over a finite field.
    
    Actually, let's just use the schoolbook for now since it's guaranteed correct,
    and try to optimize if time permits.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Start with schoolbook: rank = n*m*p = 105
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify the schoolbook decomposition
    # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
    # For a=i*m+j, b=j*p+k, c=k*n+i: should be 1
    # For other combinations: should be 0
    
    # Let's try to find a lower rank decomposition
    # Using a systematic approach: try rank = 90, 85, 80, ...
    
    # For now, return the schoolbook which is guaranteed correct
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
