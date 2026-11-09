"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    def _time_remaining():
        return deadline - time.time()
    
    def _verify(U, V, W):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    def _make_schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Start with schoolbook as fallback
    best = _make_schoolbook()
    best_rank = n * m * p
    
    # Strategy: Use Kronecker product structure
    # m = 4 = 2 * 2, so we can view the middle dimension as a 2x2 block
    # j = j1 * 2 + j2 where j1, j2 in {0, 1}
    # 
    # The tensor T[i, j1, j2, k] needs to be decomposed.
    # We can try to use the fact that for fixed (i, k), the sum over j1, j2
    # of A[i, j1*2+j2] * B[j1*2+j2, k] must work out.
    #
    # Key idea: Try to find a decomposition where each term r has a block structure
    # in the j dimension. Specifically, for each term, the V row for j = j1*2+j2
    # should be of the form V[j1*2+j2, r] = a_r(j1) * b_r(j2) for some factors.
    #
    # This reduces the search space significantly.
    
    # Approach: Enumerate possible structured decompositions
    # For each term r, specify:
    #   - U values: U[i*m+j, r] for i in {0,1}, j in {0,1,2,3}
    #   - V values: V[j*p+k, r] for j in {0,1,2,3}, k in {0,1,2,3,4}
    #   - W values: W[k*n+i, r] for k in {0,1,2,3,4}, i in {0,1}
    #
    # With the Kronecker structure, we can parameterize V[j*p+k, r] more compactly.
    
    # Let's try a direct search over small integer values with the block structure.
    # For each term r, let:
    #   u_r[i, j1, j2] = U[i*4 + j1*2 + j2, r]  (2x2x2 array)
    #   v_r[j1, j2, k] = V[(j1*2+j2)*5 + k, r]  (2x2x5 array)
    #   w_r[k, i] = W[k*2 + i, r]  (5x2 array)
    #
    # The constraint is: for all i, j1, j2, k:
    #   sum_r u_r[i,j1,j2] * v_r[j1,j2,k] * w_r[k,i] = 1
    
    # This is still a large search space, but we can use the structure to prune.
    
    # Alternative: Try known low-rank decompositions from literature.
    # For 2x2x2, the minimum rank is 7 (Strassen).
    # For 2x4x5, we might be able to achieve rank < 40.
    
    # Let's try a more systematic approach: use the fact that the tensor is
    # a sum of outer products, and search for a sparse solution.
    
    # First, let's try to see if rank 39 is achievable with a structured search.
    # We'll use the Kronecker structure to reduce the effective dimensionality.
    
    # For now, let's try a few structured patterns:
    
    # Pattern 1: Try to merge some schoolbook terms
    # The schoolbook has 40 terms, one for each (i,j,k).
    # Can we merge two terms if they share structure?
    
    # Let's try: for each pair of (i,j,k) triples, see if we can replace
    # two schoolbook terms with one combined term.
    
    # Actually, let's try a different approach: use integer linear programming
    # style search with the block structure.
    
    # Given the time constraints, let's try a focused search:
    # 1. Try rank from 30 to 39 with structured random search
    # 2. Use the Kronecker structure to guide the search
    
    for target_rank in range(30, min(40, best_rank)):
        if _time_remaining() < 5.0:
            break
        
        found = _try_structured_search(target_rank, n, m, p, nm, mp, pn, time_limit=3.0)
        if found is not None:
            U, V, W = found
            if _verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found a valid decomposition, try lower ranks
    
    # If we didn't find anything better, return the schoolbook
    return best


def _try_structured_search(target_rank, n, m, p, nm, mp, pn, time_limit=3.0):
    """
    Try to find a decomposition using the Kronecker product structure.
    m = 4 = 2 * 2, so j = j1*2 + j2.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Small integer values to try
    values = [-2, -1, 0, 1, 2]
    
    for it in range(2000):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random U, V, W with small integer entries
        # Use the block structure to guide generation
        U = [[0] * target_rank for _ in range(nm)]
        V = [[0] * target_rank for _ in range(mp)]
        W = [[0] * target_rank for _ in range(pn)]
        
        for r in range(target_rank):
            # U: 2x4 matrix (i, j)
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = rng.choice(values)
            
            # V: 4x5 matrix (j, k) - use block structure
            for j in range(m):
                for k in range(p):
                    V[j*p+k][r] = rng.choice(values)
            
            # W: 5x2 matrix (k, i)
            for k in range(p):
                for i in range(n):
                    W[k*n+i][r] = rng.choice(values)
        
        # Verify
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        valid = False
                        break
        
        if valid:
            return U, V, W
    
    return None
# EVOLVE-BLOCK-END
