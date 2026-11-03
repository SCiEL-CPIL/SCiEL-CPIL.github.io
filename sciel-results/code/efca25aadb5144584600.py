import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _time_ok(deadline):
    return time.time() < deadline

def _to_fraction(x):
    """Convert a float to Fraction if it's close to a rational with small denominator."""
    f = float(x)
    # Check if integer
    if abs(f - round(f)) < 1e-10:
        return Fraction(int(round(f)))
    # Check if half-integer
    half = round(f * 2) / 2
    if abs(f - half) < 1e-10:
        return Fraction(half).limit_denominator(1000000)
    # Try to find a fraction with small denominator
    for d in range(1, 1000):
        num = round(f * d)
        if abs(f - num/d) < 1e-10:
            return Fraction(num, d)
    return Fraction(f)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Strassen components for 2x2x2
    # local_a: 0=A11, 1=A12, 2=A21, 3=A22
    # local_b: 0=B11, 1=B12, 2=B21, 3=B22
    # local_w: 0=C11, 1=C12, 2=C21, 3=C22
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]
    
    # Blocks: (a_rows, a_cols, b_rows, b_cols)
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    rank = len(blocks) * len(strassen)  # 6 * 7 = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for a_rows, a_cols, b_rows, b_cols in blocks:
        if not _time_ok(deadline):
            break
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            if not _time_ok(deadline):
                break
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx], col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx], col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx], col] += coeff
            col += 1
    
    # Try to find a rank-minimal decomposition using integer/half-integer entries
    # by searching for a smaller rank that still works
    # The theoretical lower bound is n*p = 12, but we need to find the exact minimum
    
    # Let's try to find a decomposition with smaller rank by using a different approach
    # For 2x4 by 4x6, we can try to use the fact that 4 = 2*2 and 6 = 3*2
    # and use a tensor decomposition approach
    
    # Actually, let's just return the current best (rank 42) since finding a smaller
    # exact decomposition is non-trivial
    
    # Convert to lists with exact fractions where possible
    U_list = [[_to_fraction(U[i, j]).numerator / _to_fraction(U[i, j]).denominator 
               if _to_fraction(U[i, j]).denominator > 1 
               else int(_to_fraction(U[i, j]).numerator) 
               for j in range(rank)] for i in range(n * m)]
    
    V_list = [[_to_fraction(V[i, j]).numerator / _to_fraction(V[i, j]).denominator 
               if _to_fraction(V[i, j]).denominator > 1 
               else int(_to_fraction(V[i, j]).numerator) 
               for j in range(rank)] for i in range(m * p)]
    
    W_list = [[_to_fraction(W[i, j]).numerator / _to_fraction(W[i, j]).denominator 
               if _to_fraction(W[i, j]).denominator > 1 
               else int(_to_fraction(W[i, j]).numerator) 
               for j in range(rank)] for i in range(p * n)]
    
    # Simplify: just return floats since the values are already integers or half-integers
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
