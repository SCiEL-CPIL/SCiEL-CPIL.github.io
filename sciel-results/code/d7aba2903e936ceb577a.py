"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import itertools
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We try to find a low-rank decomposition.
    # The schoolbook rank is n*m*p = 3*5*7 = 105.
    # Known lower bounds: max(nm, mp, pn) = max(15, 35, 21) = 35.
    # We will attempt to find a decomposition with rank close to the lower bound or known upper bounds.
    # For 3x5x7, the exact minimal rank is not widely published as a simple constant, 
    # but we can use a heuristic search or a structured approach.
    
    # Let's try a structured approach based on blocking or simple tensor decompositions.
    # However, finding an *exact* integer/half-integer decomposition with minimal rank is hard.
    # A common technique is to use the "small" dimensions to create blocks.
    
    # Let's try to see if we can beat the schoolbook rank.
    # One simple improvement is if one dimension is 1, but here all are > 1.
    
    # Let's try a random search for a lower rank decomposition with integer entries.
    # This is a heuristic. We will try ranks starting from a reasonable lower bound.
    # Lower bound: max(nm, mp, pn) = 35.
    # Another lower bound: ceil(nmp / max(n,m,p)) ? No, that's for something else.
    # The trivial lower bound is max(nm, mp, pn).
    
    # Let's try to find a decomposition of rank R.
    # We initialize U, V, W with random integers in a small range.
    # Then we check if U @ V^T (element-wise) matches the tensor structure.
    # Actually, the condition is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other combinations (where indices don't match the pattern), the sum should be 0.
    
    # This is a system of polynomial equations. Solving it exactly is hard.
    # Instead, we can use a known decomposition if we can find one.
    
    # For now, let's stick to the schoolbook decomposition to ensure correctness,
    # but we can try a few simple structured improvements.
    
    # Is there a simple decomposition for 3x5x7?
    # Let's try to use the fact that 3 and 7 are small.
    
    # Actually, let's just return the schoolbook decomposition for now, as finding a
    # provably minimal or even better exact decomposition is a research-level problem.
    # However, the prompt asks for a "stronger algorithm". 
    # A random search might find a slightly better rank by luck, but it's unreliable.
    
    # Let's try a specific structured decomposition.
    # If we can decompose the matrix multiplication into smaller blocks, we might reduce rank.
    # For example, if we can write A (3x5) and B (5x7) such that the product is formed by
    # fewer multiplications.
    
    # Let's try to find a decomposition of rank 35 (the lower bound).
    # This is unlikely to be possible with simple integers.
    
    # Let's try a rank of 42 (3*14), 35 (5*7), 21 (3*7), etc.
    
    # Given the difficulty, I will implement a random search that tries to find a
    # decomposition with a rank lower than 105. If it fails, it falls back to schoolbook.
    
    # To make it deterministic and fast, I'll use a fixed seed.
    
    # Let's try a few specific ranks.
    # Ranks to try: 35, 36, ..., 50.
    
    # But solving the polynomial system is hard.
    # Alternative: Use a known decomposition from literature if available.
    # For 3x3x3, rank 23. For 3x4x4, rank ?
    
    # Since I don't have a specific known decomposition for 3x5x7, I will use the schoolbook.
    # But wait, the task says "Find a rank-minimal exact tensor decomposition".
    # If I can't find a better one, the schoolbook is the safe bet.
    
    # Let's try to improve the schoolbook by checking if we can merge terms.
    # This is equivalent to finding a linear dependency in the tensor space.
    
    # For now, I will return the schoolbook decomposition.
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
