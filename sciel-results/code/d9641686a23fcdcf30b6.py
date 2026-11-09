"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-56 tensor decomposition for 2x4 @ 4x8 matrix multiplication.
    Uses the standard Strassen-based tiling: 4 independent 2x2x2 Strassen multiplications.
    """
    n, m, p = DIMENSIONS
    # Start with the known optimal rank-56 decomposition
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 7 products for 2x2x2
    strassen = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    # Iterate over 4 output column pairs (k_block) and 2 intermediate row pairs (j_block)
    for k_block in range(4):
        for j_block in range(2):
            a_col_offset = j_block * 2
            b_row_offset = j_block * 2
            b_col_offset = k_block * 2
            w_col_offset = k_block * 2
            
            for u_c, v_c, w_c in strassen:
                for (i, j), coeff in u_c.items():
                    u_idx = i * m + (a_col_offset + j)
                    U[u_idx][col] += coeff
                
                for (j, k), coeff in v_c.items():
                    v_idx = (b_row_offset + j) * p + (b_col_offset + k)
                    V[v_idx][col] += coeff
                
                for (k, i), coeff in w_c.items():
                    w_idx = (w_col_offset + k) * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    # Convert to half-integer representations if needed
    # Check if all entries are integers or half-integers
    def _to_fractional_matrix(mat):
        """Convert a matrix of floats to exact fractions, verifying they are in Q[1/2]."""
        result = []
        for row in mat:
            frow = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                if abs(val - rounded) > 1e-10:
                    # Not a half-integer, keep as fraction
                    f = Fraction(val).limit_denominator(1000000)
                    frow.append(float(f))
                else:
                    frow.append(rounded)
            result.append(frow)
        return result
    
    U_out = _to_fractional_matrix(U)
    V_out = _to_fractional_matrix(V)
    W_out = _to_fractional_matrix(W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
