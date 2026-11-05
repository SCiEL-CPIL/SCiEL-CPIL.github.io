"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105  # Safety margin before 120s limit

    def time_remaining():
        return deadline - time.time()

    # Build schoolbook decomposition as fallback (rank 72)
    schoolbook_rank = n * m * p  # 72
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1 if (j == j2 and k == k2 and i == i2) else 0
                    if val != expected:
                        return False
        return True

    # Strategy: Use structured algebraic approach
    # For 3x3 * 3x8, we can decompose using the following insight:
    # The tensor M_{3,3,8} can be viewed as having a "core" 3x3x3 part 
    # replicated/extended for the 8 columns.
    #
    # Key idea: Use a rank-r decomposition of the 3x3x3 tensor, then
    # extend it to handle the 8 columns.
    #
    # For the 3x3x3 tensor, we know rank is at most 9 (schoolbook).
    # We'll try to find a lower rank decomposition.
    #
    # Approach 1: Try known low-rank decompositions for small matrix multiplication
    # For 2x2 * 2x2, Strassen gives rank 7.
    # For 3x3 * 3x3, the best known is around 19-20, but let's try to find one.
    #
    # Actually, let's try a different approach: use the fact that p=8 is a power of 2
    # and use a recursive/block decomposition.
    #
    # Better approach: Directly search for low-rank decompositions using
    # algebraic identities and local optimization.
    #
    # Let's try a "peeling" approach: start with schoolbook and try to
    # eliminate terms by finding linear dependencies.

    # Method: Try to construct a decomposition using integer/half-integer coefficients
    # by solving a system of equations.
    #
    # For rank r, we need to find U (9 x r), V (24 x r), W (24 x r) such that
    # for all (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    #
    # This is a system of 9*24*24 = 5184 equations in 9*r + 24*r + 24*r = 57*r unknowns.
    # For r=24: 5184 equations, 1368 unknowns - overdetermined
    # For r=72: 5184 equations, 4104 unknowns - underdetermined
    #
    # Let's try a greedy/iterative approach to find a valid decomposition.

    # Try structured decomposition using block structure
    # Split the 8 columns into groups and use block matrix multiplication ideas

    # Approach: Use a rank-16 decomposition based on the following:
    # For each of the 3 middle indices j, we have a 3x8 outer product.
    # We can try to combine these using algebraic identities.

    # Let's try a direct construction:
    # The tensor entry T[(i,j),(j,k),(k,i)] = 1
    # We want: T = sum_r U_r ⊗ V_r ⊗ W_r
    
    # Try rank 32, 40, 48, 56, 64, 72 with random + local search

    def random_decomposition(target_rank, seed=None):
        """Generate a random sparse decomposition and try to fix it."""
        rng = random.Random(seed)
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Initialize with schoolbook-like structure but shuffled
        # Each term r should contribute to exactly one (i,j,k) triple
        # But we can use linear combinations to reduce rank
        
        # Start with a random assignment
        for r in range(target_rank):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
        
        return U, V, W

    def local_search(U, V, W, max_iter=1000, seed=None):
        """Try to reduce rank by finding and removing dependent terms."""
        rng = random.Random(seed)
        rank = len(U[0])
        
        # Try to zero out terms that are linearly dependent
        # This is a heuristic approach
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Try removing a random term and checking if the decomposition still works
            # after adjusting other terms
            r_remove = rng.randint(0, rank - 1)
            
            # Check if term r_remove is "redundant" - i.e., can be expressed
            # as a linear combination of other terms
            # This is complex, so we use a simpler heuristic:
            # Try setting the term to zero and see if we can compensate
            
            # Skip for now - just verify current state
            if verify_decomposition(U, V, W):
                return U, V, W, rank
        
        return U, V, W, rank

    # Try a more systematic approach: use the known structure
    # For 3x3 * 3x8, we can use the following decomposition:
    # 
    # Let A be 3x3, B be 3x8, C = A*B be 3x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this using 5 "Strassen-like" terms for the 3x3 core,
    # but that's not straightforward for non-square matrices.
    #
    # Instead, let's try a direct algebraic construction.
    #
    # Key insight: The tensor M_{3,3,8} has a symmetry that we can exploit.
    # The 8 columns of B can be grouped, and we can use linearity.

    # Try ranks from 24 to 72 with a smarter search
    # Use the fact that we need integer/half-integer coefficients

    # Approach: Use a "basis" of rank-1 tensors and find a subset that spans
    # the multiplication tensor.

    # For now, let's try a few specific constructions:
    
    # Construction 1: Use the 3x3x3 tensor decomposition (if we can find one)
    # and extend it to 8 columns.
    
    # The 3x3x3 tensor has rank at most 9. If we can find a rank-9 decomposition,
    # then for 3x3x8 we might be able to use rank 9 * ceil(8/something) or similar.
    
    # Actually, the correct way to extend: if we have a rank-r decomposition of M_{3,3,3},
    # then M_{3,3,8} can be decomposed by "padding" the V and W factors.
    # But this doesn't directly reduce the rank.
    
    # Let's try a completely different approach: use the fact that the problem
    # is small enough to do an exhaustive search over small coefficient sets.

    # For rank r, each entry of U, V, W is in {-1, 0, 1} or {-2,-1,0,1,2}
    # The number of possibilities is huge, so we need a smarter search.

    # Use a "meet-in-the-middle" or "constraint satisfaction" approach.
    
    # For now, let's try a simple hill-climbing approach:
    # 1. Start with schoolbook (rank 72)
    # 2. Try to merge pairs of terms
    # 3. Verify after each merge

    def try_merge_terms(U, V, W, r1, r2, seed=None):
        """Try to merge terms r1 and r2 into a single term."""
        rng = random.Random(seed)
        rank = len(U[0])
        
        # The merged term would have:
        # U_new[a] = U[r1][a] + U[r2][a] for all a
        # V_new[b] = V[r1][b] + V[r2][b] for all b  
        # W_new[c] = W[r1][c] + W[r2][c] for all c
        # But this doesn't work because (u1+v1)*(v1+v2)*(w1+w2) != u1*v1*w1 + u2*v2*w2
        
        # Instead, we need to find coefficients alpha, beta such that:
        # alpha * term1 + beta * term2 can be replaced by a single term
        # This is generally not possible, so this approach is limited.
        
        return None

    # Let's try a different strategy: use known results from tensor rank theory.
    # The rank of M_{3,3,8} is at most 72 (schoolbook).
    # Lower bound: max(9, 24, 24) = 24 (from the slice rank lower bound)
    # Actually, the slice rank lower bound for M_{n,m,p} is max(n*m, m*p, p*n) = max(9,24,24) = 24.
    # But this is a lower bound on the slice rank, not necessarily the tensor rank.
    # The tensor rank can be lower than the slice rank.
    
    # Let's try to find a decomposition with rank between 24 and 72.
    # Use a random search with verification, but be smarter about it.

    # Improved random search:
    # 1. Generate random sparse decompositions
    # 2. Use local search to improve them
    # 3. Try multiple random seeds

    best_rank_found = schoolbook_rank
    best_decomp = (U_school, V_school, W_school)

    # Try ranks from 40 down to 24 (more likely to find something)
    for target_rank in [64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24]:
        if time_remaining() < 10:
            break
        
        found = False
        for attempt in range(200):
            if time_remaining() < 5:
                break
            
            seed = attempt * 1000 + target_rank
            U, V, W = random_decomposition(target_rank, seed)
            
            # Quick verification
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
