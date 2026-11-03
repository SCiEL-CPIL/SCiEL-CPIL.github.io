"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]
from fractions import Fraction
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Candidate values for entries: 0, 1, -1, 1/2, -1/2
    candidate_vals = [0.0, 1.0, -1.0, 0.5, -0.5]
    
    for rank in range(30, 43):
        found = False
        # Try 200 random configurations per rank
        for trial in range(200):
            # Generate random U and V
            U = np.random.choice(candidate_vals, size=(nm, rank))
            V = np.random.choice(candidate_vals, size=(mp, rank))
            
            # Build A and b
            # A shape: (n*m*p, p*n*rank)
            # b shape: (n*m*p,)
            
            A = np.zeros((n * m * p, pn * rank))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_row_base = k * n + i
                        for c in range(rank):
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            if coeff != 0:
                                flat_idx = w_row_base * rank + c
                                A[eq_idx, flat_idx] = coeff
                        eq_idx += 1
            
            # Solve using least squares
            # Since it's underdetermined, lstsq gives min norm solution
            try:
                W_flat, residuals, rank_np, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Round W_flat to nearest 0.5
            W_rounded = np.round(W_flat * 2) / 2.0
            
            # Reshape W
            W = W_rounded.reshape((pn, rank))
            
            # Verify with Fractions
            # Convert U, V, W to Fraction lists for exact check
            # To save time, we can check with floats first, then exact
            # Float check
            valid_float = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(rank):
                            total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid_float = False
                            break
                    if not valid_float:
                        break
                if not valid_float:
                    break
            
            if not valid_float:
                continue
            
            # Exact verification
            U_frac = [[Fraction(int(x*2), 2) for x in row] for row in U.tolist()]
            V_frac = [[Fraction(int(x*2), 2) for x in row] for row in V.tolist()]
            W_frac = [[Fraction(int(x*2), 2) for x in row] for row in W.tolist()]
            
            valid_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = Fraction(0)
                        for c in range(rank):
                            total += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                        if total != 1:
                            valid_exact = False
                            break
                    if not valid_exact:
                        break
                if not valid_exact:
                    break
            
            if valid_exact:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        pass
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
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
