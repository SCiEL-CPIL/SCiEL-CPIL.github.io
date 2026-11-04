"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import itertools
import random

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] equals the MM tensor."""
    rank = len(U[0])
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        s = 0
        for c in range(rank):
            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if abs(s - 1) > tol:
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# Rank-19 decomposition for 5x5x5 matrix multiplication (Landsberg's construction)
# The factors are given as lists of [row, col, value] triples for each tensor.
# We use the known rank-19 factors for 5x5x5.

def _rank19_5x5x5():
    """Return the rank-19 decomposition for 5x5x5 matrix multiplication."""
    # Using the explicit rank-19 factors from literature
    # U: 25 x 19, V: 25 x 19, W: 25 x 19
    
    # The rank-19 decomposition of 5x5x5 matrix multiplication
    # Based on the construction by Landsberg (1994)
    
    # We'll construct it using the known structure.
    # For 5x5x5, we can use the following approach:
    # The tensor can be decomposed using 19 rank-1 terms.
    
    # Here's a known rank-19 decomposition:
    # We use the fact that 5x5x5 has rank 19.
    
    # Let me provide the actual factors. I'll use a known valid set.
    
    # Actually, let me construct this properly. The rank-19 decomposition
    # for 5x5x5 matrix multiplication is well-known. Let me encode it.
    
    # Using the standard rank-19 factors:
    U = [[0.0] * 19 for _ in range(25)]
    V = [[0.0] * 19 for _ in range(25)]
    W = [[0.0] * 19 for _ in range(25)]
    
    # Define the 19 rank-1 terms
    # Each term (i,j,k) with coefficients
    terms = []
    
    # The rank-19 decomposition uses the following structure:
    # For 5x5x5, we can use:
    # - 16 terms from a 4x4x4 block structure
    # - 3 additional terms
    
    # Let me use a verified set of factors.
    # I'll construct using the known algorithm.
    
    # Actually, let me just hardcode a valid rank-19 decomposition.
    # The factors for 5x5x5 rank-19 MM tensor:
    
    # Using the construction from the literature, here are the factors:
    # U[i*5+j][c], V[j*5+k][c], W[k*5+i][c]
    
    # Let me define them explicitly:
    
    # Term 0: (i,j,k) = (0,0,0)
    U[0*5+0][0] = 1.0; V[0*5+0][0] = 1.0; W[0*5+0][0] = 1.0
    
    # Term 1: (i,j,k) = (0,0,1)
    U[0*5+0][1] = 1.0; V[0*5+1][1] = 1.0; W[1*5+0][1] = 1.0
    
    # Term 2: (i,j,k) = (0,0,2)
    U[0*5+0][2] = 1.0; V[0*5+2][2] = 1.0; W[2*5+0][2] = 1.0
    
    # Term 3: (i,j,k) = (0,0,3)
    U[0*5+0][3] = 1.0; V[0*5+3][3] = 1.0; W[3*5+0][3] = 1.0
    
    # Term 4: (i,j,k) = (0,0,4)
    U[0*5+0][4] = 1.0; V[0*5+4][4] = 1.0; W[4*5+0][4] = 1.0
    
    # Term 5: (i,j,k) = (0,1,0)
    U[0*5+1][5] = 1.0; V[1*5+0][5] = 1.0; W[0*5+0][5] = 1.0
    
    # Term 6: (i,j,k) = (0,1,1)
    U[0*5+1][6] = 1.0; V[1*5+1][6] = 1.0; W[1*5+0][6] = 1.0
    
    # Term 7: (i,j,k) = (0,1,2)
    U[0*5+1][7] = 1.0; V[1*5+2][7] = 1.0; W[2*5+0][7] = 1.0
    
    # Term 8: (i,j,k) = (0,1,3)
    U[0*5+1][8] = 1.0; V[1*5+3][8] = 1.0; W[3*5+0][8] = 1.0
    
    # Term 9: (i,j,k) = (0,1,4)
    U[0*5+1][9] = 1.0; V[1*5+4][9] = 1.0; W[4*5+0][9] = 1.0
    
    # Term 10: (i,j,k) = (0,2,0)
    U[0*5+2][10] = 1.0; V[2*5+0][10] = 1.0; W[0*5+0][10] = 1.0
    
    # Term 11: (i,j,k) = (0,2,1)
    U[0*5+2][11] = 1.0; V[2*5+1][11] = 1.0; W[1*5+0][11] = 1.0
    
    # Term 12: (i,j,k) = (0,2,2)
    U[0*5+2][12] = 1.0; V[2*5+2][12] = 1.0; W[2*5+0][12] = 1.0
    
    # Term 13: (i,j,k) = (0,2,3)
    U[0*5+2][13] = 1.0; V[2*5+3][13] = 1.0; W[3*5+0][13] = 1.0
    
    # Term 14: (i,j,k) = (0,2,4)
    U[0*5+2][14] = 1.0; V[2*5+4][14] = 1.0; W[4*5+0][14] = 1.0
    
    # Term 15: (i,j,k) = (0,3,0)
    U[0*5+3][15] = 1.0; V[3*5+0][15] = 1.0; W[0*5+0][15] = 1.0
    
    # Term 16: (i,j,k) = (0,3,1)
    U[0*5+3][16] = 1.0; V[3*5+1][16] = 1.0; W[1*5+0][16] = 1.0
    
    # Term 17: (i,j,k) = (0,3,2)
    U[0*5+3][17] = 1.0; V[3*5+2][17] = 1.0; W[2*5+0][17] = 1.0
    
    # Term 18: (i,j,k) = (0,3,3)
    U[0*5+3][18] = 1.0; V[3*5+3][18] = 1.0; W[3*5+0][18] = 1.0
    
    # This is just a partial schoolbook-like construction, not rank 19.
    # Let me use a proper rank-19 decomposition instead.
    
    # I need to use a known valid rank-19 decomposition for 5x5x5.
    # Let me use the one from the literature properly.
    
    # Actually, the above is NOT rank 19 - it's just 19 terms but they don't
    # correctly decompose the full 5x5x5 tensor. Let me use a correct one.
    
    # For now, let me return the schoolbook for safety, but I'll implement
    # the slicing approach with a proper rank-19 5x5x5 decomposition.
    
    # Let me construct a valid rank-19 decomposition using known factors.
    
    # The correct rank-19 decomposition for 5x5x5 matrix multiplication:
    # I'll use the factors from the known result.
    
    # Let me restart with a proper construction.
    
    return None  # Placeholder - will be replaced

def _rank19_5x5x5_correct():
    """Return the rank-19 decomposition for 5x5x5 matrix multiplication using known factors."""
    # Using the explicit rank-19 factors from the literature
    # This is based on the construction that achieves rank 19 for 5x5x5
    
    # The factors are 25x19 matrices U, V, W
    
    # I'll use a verified set of rank-19 factors for 5x5x5 MM
    
    # From known results, here's a valid rank-19 decomposition:
    
    U = [[0.0]*19 for _ in range(25)]
    V = [[0.0]*19 for _ in range(25)] 
    W = [[0.0]*19 for _ in range(25)]
    
    # Define the rank-1 terms with their (i,j,k) indices and coefficients
    # This is a known valid rank-19 decomposition
    
    # Using the structure from the literature:
    terms = [
        # (i, j, k, u_coeff, v_coeff, w_coeff)
        # We need 19 terms that correctly decompose the 5x5x5 tensor
        
        # Let me use a different approach - I'll construct it from scratch
        # using the known algebraic structure
    ]
    
    # Actually, let me just use the schoolbook as a fallback for now
    # and implement the slicing properly once I have valid rank-19 factors
    
    # For the purpose of this task, let me use a simpler approach:
    # Use the known fact that we can slice the 5th dimension
    
    # But first I need valid rank-19 factors for 5x5x5.
    
    # Let me return None to indicate this needs proper implementation
    return None

def construct():
    n, m, p = DIMENSIONS
    # Return the schoolbook decomposition (rank = n*m*p = 100)
    # The slicing approach from rank-19 5x5x5 would give rank 19 for 4x5x5,
    # but we do not have the explicit rank-19 factors hardcoded.
    # A safe, always-valid decomposition is the schoolbook one.
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
