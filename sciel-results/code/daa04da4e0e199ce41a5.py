"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import itertools

def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # First, compute the catalecticant ranks to understand the structure
    # This guides our search for a lower-rank decomposition
    
    # Try to find a decomposition with rank less than schoolbook (n*m*p = 80)
    # Known bounds: for 4x4 x 4x5, rank is at most 80 (schoolbook)
    # We'll try structured approaches to reduce rank
    
    # Strategy 1: Try a block-structured decomposition
    # Group rows/columns to exploit structure
    
    # For now, let's try a systematic search for small rank decompositions
    # using the fact that the tensor has specific zero patterns
    
    # First, verify schoolbook works
    schoolbook = _schoolbook_decomposition(n, m, p)
    if _verify_decomposition(schoolbook, n, m, p, num_checks=50):
        # Schoolbook is valid with rank n*m*p = 80
        # Try to find something better
        pass
    
    # Try structured decomposition based on Winograd-like ideas
    # For 4x4 x 4x5, we can try:
    # - Split the 4x5 matrix into 4x4 + 4x1
    # - Use a rank-20 decomposition for 4x4 x 4x4 (if exists)
    # - Add 4 more terms for the extra column
    
    # Actually, let's try a direct approach:
    # Construct a decomposition using the structure of matrix multiplication
    
    # Approach: Use the fact that M_{ijk} = delta_{j,k} (in appropriate indices)
    # We can try to find a sparse decomposition
    
    # Try rank 24 first (known achievable for some 4x4 x 4x4 cases)
    # Then extend to handle the 5th column
    
    # For simplicity and correctness, let's use a verified approach:
    # 1. Start with schoolbook
    # 2. Try to compress using linear dependencies
    
    # Actually, the most reliable approach within time constraints:
    # Return a valid decomposition. Let's use a structured one.
    
    # Try a rank-32 decomposition using block structure
    decomp = _structured_decomposition(n, m, p)
    if decomp is not None and _verify_decomposition(decomp, n, m, p, num_checks=100):
        return decomp
    
    # Fallback to schoolbook
    return schoolbook


def _structured_decomposition(n, m, p):
    """
    Try to construct a structured decomposition with rank < n*m*p.
    
    For 4x4 x 4x5, we use the following approach:
    - The multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A 
      times the (j,k) entry of B contributes to the (i,k) entry of C
    - We can exploit the structure by grouping indices
    
    Strategy: Use a Kronecker-like product structure
    """
    n, m, p = 4, 4, 5
    
    # For 4x4 x 4x4, we can use a rank-16 decomposition based on:
    # Split into 2x2 blocks and use 2x2 matrix multiplication (rank 7)
    # But 2x2 x 2x2 has rank 7, so 4x4 x 4x4 might have rank 16 or less
    
    # Actually, let's use a simpler structured approach:
    # Group the m=4 dimension into pairs: (0,1) and (2,3)
    # For each pair, we have a 4x2 x 4x5 sub-problem
    
    # Let's try a direct construction:
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (20, rank)  
    # W has shape (p*n, rank) = (20, rank)
    
    # For rank 32: use the structure that each (i,k) pair in the output
    # depends on all j, but we can share some structure
    
    # Try a decomposition where we group by (i mod 2, k mod 2)
    # This gives 4 groups, each handling a 2x2 x 2x5 sub-problem
    
    # Actually, let's just return None and use schoolbook for now
    # The structured approach needs more careful design
    
    return None


def _schoolbook_decomposition(n, m, p):
    """Create a schoolbook (trivial) tensor decomposition for n x m by m x p matrix multiplication."""
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


def _verify_decomposition(decomp, n, m, p, num_checks=200):
    """Verify the decomposition by checking random entries."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check some invalid positions give 0
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        # Check if (a,b,c) corresponds to a valid (i,j,k)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue  # Valid position, skip
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True
# EVOLVE-BLOCK-END
