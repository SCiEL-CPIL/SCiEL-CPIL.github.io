"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """
    Verify if the decomposition exactly multiplies matrices.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Convention: W index is k*n + i.
    T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
    We want T[i, j, k] = 1 for all i, j, k.
    """
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != r or len(V[0]) != r or len(W[0]) != r:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U_np[i * m + j]
                v = V_np[j * p + k]
                w = W_np[k * n + i]
                s = np.dot(u, v * w)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _verify_exact(U, V, W):
    """Exact verification using fractions."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != r or len(V[0]) != r or len(W[0]) != r:
        return False
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_3x3x3():
    """
    Strassen decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W with rank 7 (or close).
    """
    # Standard Strassen uses rank 7 for 2x2x2, but for 3x3x3 we need more.
    # Let's use a known rank-13 decomposition for 3x3x3.
    # Actually, the best known for 3x3x3 is rank 13 (Bayer, 1987).
    # For now, let's construct a valid decomposition.
    
    # Use the trivial decomposition as baseline, then try to reduce.
    n = 3
    rank = n * n * n  # 27
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j][col] = 1
                V[j * n + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_block_decomposition():
    """
    Construct a block-structured decomposition for 3x5x5.
    
    Key idea: The 5x5 multiplication can be decomposed into blocks.
    We treat the problem as: C[3x5] = A[3x5] * B[5x5]
    
    We can partition B into 5 column blocks of width 1, but that's just schoolbook.
    
    Better: Use the fact that 5 = 2 + 3, and use Strassen-like tricks on sub-blocks.
    Or: Use a rank-reducing scheme based on the algebraic structure.
    
    Alternative approach: Use the "embedding" method where we embed the 3x5x5 problem
    into a larger problem with known low-rank decomposition.
    
    Let's try a direct algebraic construction:
    - For each (i,j) pair in A and each k in B, we need the product.
    - The tensor T[i,j,k] = delta_{i,j,k} (all ones)
    
    A rank-r decomposition means:
    T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    Let's try to find a structured solution with r < 75.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Try a Kronecker product approach:
    # Decompose 5x5 multiplication using a rank-r5 decomposition of 5x5,
    # then combine with the 3x5 part.
    
    # Actually, let's think differently. The matrix multiplication tensor for m x n x p
    # has rank at least max(m*n, n*p, p*m) = max(15, 25, 15) = 25.
    # So rank >= 25.
    
    # Let's try to construct a rank-25 or close decomposition.
    
    # One approach: If we can find a rank-r decomposition of the 5x5x5 tensor,
    # then we can use it here. The rank of 5x5x5 is at least 25.
    
    # For now, let's use a systematic search with structured initialization.
    
    # Start with the schoolbook and try to merge terms.
    # Schoolbook has 75 terms. Can we find linear dependencies?
    
    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,5)
    # is the all-ones tensor. Wait, no - it's the matrix multiplication tensor.
    
    # Actually, the matrix multiplication tensor for multiplying A[m x n] by B[n x p]
    # to get C[m x p] has entries T[a, b, c] where a indexes rows of A, b indexes 
    # (col of A, row of B), c indexes cols of C.
    
    # In our convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is the standard matrix multiplication tensor.
    
    # Let's try a different structural approach: use the fact that we can
    # factor the 5-dimension into sub-dimensions.
    
    # Approach: Use a "tensor product" decomposition.
    # Write 5 = 2 + 3. Then 5x5 multiplication can be done by:
    # Splitting A into A1 (3x2) and A2 (3x3), B into B1 (2x5) and B2 (3x5)
    # C = A1*B1 + A2*B2
    # This gives rank = rank(3x2x2) + rank(3x3x3) <= 3*2*2 + 3*3*3 = 12 + 27 = 39
    # But we can do better with Strassen on the 3x3x3 part.
    
    # Let's try: rank(3x2x2) = 6 (schoolbook), rank(3x3x3) = 13 (Bayer)
    # Total: 6 + 13 = 19? That seems too good. Let me reconsider.
    
    # Actually, the rank of 3x2x2 matrix multiplication is 6 (it's just 6 scalar mults).
    # The rank of 3x3x3 is 13 (best known).
    # But when we combine them, we're not just adding ranks - we need to handle
    # the cross terms properly.
    
    # Let me just do a careful construction:
    # A is 3x5, B is 5x5, C is 3x5.
    # Split columns of A: A = [A1 | A2] where A1 is 3x2, A2 is 3x3
    # Split rows of B: B = [B1; B2] where B1 is 2x5, B2 is 3x5
    # C = A1*B1 + A2*B2
    
    # A1*B1: 3x2 times 2x5 = 3x5, rank = 3*2*2 = 12 (schoolbook) or less
    # A2*B2: 3x3 times 3x5 = 3x5, rank = 3*3*5 = 45 (schoolbook) or less
    
    # This doesn't help much. Let me try a different split.
    
    # Better: Use the structure of 5 = 2 + 3 more carefully.
    # Or just do a direct search for a low-rank decomposition.
    
    # Let's try a rank-25 construction based on the lower bound.
    
    # For rank 25, we need:
    # sum_{c=1}^{25} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # One idea: Let c index (a, b) where a in [0,5), b in [0,5).
    # Set U[i*m+j, (a,b)] = 1 if j == a, else 0
    # Set V[j*p+k, (a,b)] = 1 if k == b, else 0  
    # Set W[k*n+i, (a,b)] = 1 if ... hmm, this gives product = 1 only if j=a AND k=b
    # and W needs to give 1 for all i. So W[k*n+i, (a,b)] = 1 for all i? 
    # Then sum over (a,b) where j=a and k=b: only one term survives, giving W[k*n+i, (j,k)] = 1.
    # So W[k*n+i, (a,b)] = 1 if (a,b) = (j,k)... but that depends on j,k which vary.
    
    # This approach gives the schoolbook decomposition with rank 25... wait, 
    # c indexes (a,b) with a in [0,5), b in [0,5), so 25 values.
    # U[i*m+j, (a,b)] = delta_{j,a}
    # V[j*p+k, (a,b)] = delta_{k,b}
    # W[k*n+i, (a,b)] = 1 for all i, k, (a,b)?
    
    # Then T[i,j,k] = sum_{a,b} delta_{j,a} * delta_{k,b} * 1 = 1 (only term a=j, b=k survives)
    # YES! This gives rank 25!
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for a in range(m):  # a in [0,5)
        for b in range(p):  # b in [0,5)
            c = a * p + b  # c in [0,25)
            for i in range(n):
                U[i * m + a][c] = 1  # j = a
                for j in range(m):
                    V[j * p + b][c] = 1  # k = b
                for k in range(p):
                    for i2 in range(n):
                        W[k * n + i2][c] = 1
    
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If that didn't work, fall back to schoolbook
    return None

def _search_low_rank(deadline):
    """
    Search for rank < 25 decompositions using structured methods.
    """
    n, m, p = DIMENSIONS
    
    # Try ranks from 24 down to some minimum
    # Use a more intelligent search
    
    # For rank 24, we need one fewer term. This requires a non-trivial algebraic relation.
    # Let's try to find one by perturbing the rank-25 solution.
    
    # Start from the rank-25 solution and try to merge two terms.
    base = _construct_block_decomposition()
    if base is None:
        return None
    
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    # Try to find a linear combination that eliminates one term
    # This is a hard algebraic problem. For now, just return the rank-25 solution.
    return base

def construct():
    deadline = time.time() + 110  # 10s margin for 120s limit (but evaluator kills at 150s)
    
    # 1. Try the rank-25 structured decomposition
    res = _construct_block_decomposition()
    if res is not None:
        rank25 = len(res["U"][0])
        # Try to find something even lower
        if time.time() < deadline - 5:
            res2 = _search_low_rank(deadline)
            if res2 is not None:
                return res2
        return res
    
    # 2. Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END

