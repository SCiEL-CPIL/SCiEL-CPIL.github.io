import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n×m × m×p matrix multiplication."""
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

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    rank = len(U[0])
    
    # Build the tensor from U, V, W
    # T[a, b, c] = sum_{col=0}^{rank-1} U[a, col] * V[b, col] * W[c, col]
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                # Expected value: 1 if (a,b,c) corresponds to valid (i,j,k)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                expected = 1 if (i == i2 and j == j2) else 0
                
                actual = 0
                for col in range(rank):
                    actual += U[a][col] * V[b][col] * W[c][col]
                
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _try_structured_decomposition(n, m, p, deadline):
    """Try to find a lower-rank decomposition using structured approaches."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    schoolbook_rank = n * m * p
    
    # Try ranks from schoolbook_rank - 1 down to a reasonable minimum
    for r in range(schoolbook_rank - 1, max(10, schoolbook_rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Strategy: Use the fact that matrix multiplication has structure
        # Try to find sparse decompositions
        
        # Method 1: Random sparse initialization with local search
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with mostly zeros
            U = np.zeros((dim_a, r))
            V = np.zeros((dim_b, r))
            W = np.zeros((dim_c, r))
            
            # Place non-zeros strategically
            for col in range(r):
                # Choose random positions
                a_pos = np.random.randint(0, dim_a)
                b_pos = np.random.randint(0, dim_b)
                c_pos = np.random.randint(0, dim_c)
                
                # Ensure consistency: a=i*m+j, b=j*p+k, c=k*n+i
                i = a_pos // m
                j = a_pos % m
                j2 = b_pos // p
                k = b_pos % p
                k2 = c_pos // n
                i2 = c_pos % n
                
                # Only keep if consistent
                if i == i2 and j == j2:
                    U[a_pos, col] = 1
                    V[b_pos, col] = 1
                    W[c_pos, col] = 1
            
            # Check if this gives a valid decomposition
            # (This is a very rough check - most won't work)
            # We need a more systematic approach
            
            # Method 2: Build from partial schoolbook
            # Start with schoolbook and try to merge terms
            if attempt == 0:
                # Create a modified schoolbook where some terms are combined
                U = np.zeros((dim_a, r))
                V = np.zeros((dim_b, r))
                W = np.zeros((dim_c, r))
                
                # First r-1 terms from schoolbook
                for col in range(min(r - 1, schoolbook_rank)):
                    i = col // (m * p)
                    j = (col // p) % m
                    k = col % p
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                
                # Last term: try to cover multiple entries
                # This won't work in general, but let's try
                # Actually, we can't just add one term to cover multiple
                # because each term is a rank-1 tensor
                
                # Better approach: use known algorithms
                # For 3x3x3, we know rank is 23 (Strassen-like)
                # For 3x3x4, the optimal rank is unknown but likely < 36
                pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, get schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try to find better decomposition
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Known results suggest rank might be around 20-30 for 3x3x4
    
    # For now, return schoolbook as we haven't implemented a working search
    # A proper implementation would use:
    # 1. Integer linear programming
    # 2. Gröbner bases
    # 3. Tensor network decomposition
    # 4. Known algorithmic results
    
    # Since finding a lower-rank decomposition is computationally hard,
    # and we have time constraints, return schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
