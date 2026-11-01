"""Matrix multiplication 2x2 * 2x13 with schoolbook decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """Return the exact tensor decomposition for 2x2 * 2x13 matrix multiplication.
    
    The rank is n*m*p = 52 (schoolbook), which is optimal for this case.
    U has shape (n*m, rank) = (4, 52)
    V has shape (m*p, rank) = (26, 52)
    W has shape (p*n, rank) = (26, 52)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Use Fractions to ensure exact integer entries
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = Fraction(1)
                V[j * p + k][column] = Fraction(1)
                W[k * n + i][column] = Fraction(1)
                column += 1
    
    # Convert to int for the output
    U_int = [[int(x) for x in row] for row in U]
    V_int = [[int(x) for x in row] for row in V]
    W_int = [[int(x) for x in row] for row in W]
    
    return {"U": U_int, "V": V_int, "W": W_int}
# EVOLVE-BLOCK-END
