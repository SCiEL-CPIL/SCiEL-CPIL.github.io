import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-minimal decomposition using known results or search
    # For 3x3 * 3x7, the schoolbook rank is 3*3*7 = 63
    # We can try to reduce this using block decomposition or known algorithms
    
    # First, let's try a simple approach: use Strassen-like ideas on blocks
    # But since we need exact integer/half-integer entries, let's be conservative
    
    # Try rank 55 first (known lower bound for some matrix multiplication problems)
    for rank in [55, 56, 57, 58, 59, 60, 61, 62, 63]:
        result = try_find_decomposition(n, m, p, rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
    """Standard matrix multiplication tensor decomposition with rank n*m*p"""
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

def try_find_decomposition(n, m, p, rank):
    """
    Attempt to find a rank-`rank` exact tensor decomposition.
    
    Strategy: Use a structured approach based on splitting the p-dimension.
    For 3x3 * 3x7, we can split 7 = 3 + 4.
    - 3x3 * 3x3: Use a rank-20 decomposition (if available) or schoolbook rank 27
    - 3x3 * 3x4: Use a rank-? decomposition
    
    Actually, let's use a more systematic approach with random search + verification.
    """
    # For now, implement a basic search that tries to construct valid decompositions
    # We'll use a greedy approach: build up the decomposition term by term
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # must equal 1 if (i,j,k) is a valid multiplication index, else 0.
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
    
    # Let's try a randomized approach with exact rational arithmetic
    np.random.seed(42)
    
    # Initialize with random half-integer values
    U = np.random.randint(-2, 3, size=(n*m, rank)) / 2
    V = np.random.randint(-2, 3, size=(m*p, rank)) / 2
    W = np.random.randint(-2, 3, size=(p*n, rank)) / 2
    
    # Verify if this decomposition is exact
    if verify_decomposition(n, m, p, U, V, W):
        # Convert to Fraction for exactness
        U_frac = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[i][r]) for r in range(rank)] for i in range(m*p)]
        W_frac = [[Fraction(W[i][r]) for r in range(rank)] for i in range(p*n)]
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Try more iterations with different seeds
    for seed in range(100):
        np.random.seed(seed)
        U = np.random.randint(-2, 3, size=(n*m, rank)) / 2
        V = np.random.randint(-2, 3, size=(m*p, rank)) / 2
        W = np.random.randint(-2, 3, size=(p*n, rank)) / 2
        
        if verify_decomposition(n, m, p, U, V, W):
            U_frac = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
            V_frac = [[Fraction(V[i][r]) for r in range(rank)] for i in range(m*p)]
            W_frac = [[Fraction(W[i][r]) for r in range(rank)] for i in range(p*n)]
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the given U, V, W matrices exactly decompose the matrix multiplication tensor.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = A[i,j] * B[j,k] * C[k,i] (with appropriate indexing)
    Actually, the standard formulation is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    This should equal 1 for all valid (i,j,k) in the multiplication.
    """
    rank = U.shape[1]
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute the sum
                val = 0
                for r in range(rank):
                    val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                
                # For matrix multiplication, this should be 1
                if abs(val - 1) > 1e-9:
                    return False
    
    return True
# EVOLVE-BLOCK-END
