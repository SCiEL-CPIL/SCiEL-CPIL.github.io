import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    r = 25  # target rank
    
    # U: (n*m, r) = (10, 25)
    # V: (m*p, r) = (25, 25)
    # W: (p*n, r) = (10, 25)
    
    # We need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy:
    # 1. Randomly initialize U and V with small integers.
    # 2. Compute B_{(i,k),c} = sum_j U[i*m+j, c] * V[j*p+k, c]
    # 3. For each (i,k), check if there exists c such that B_{(i,k),c} is in {1, -1, 2, -2}.
    #    If so, we can set W[k*n+i, c] = 1/B_{(i,k),c} which is a half-integer.
    # 4. If all (i,k) pairs are satisfied, construct W and verify.
    
    # To ensure we find a solution, we try many random configurations.
    # We use a fixed seed for reproducibility in the "search" but maybe vary it if it fails?
    # The task says "Each evaluation must finish within 120 s".
    # 1000 attempts is very fast. 10000 is also fast.
    
    for attempt in range(5000):
        # Use a different seed for each attempt to explore space
        random.seed(attempt)
        
        # Generate U and V with entries in {-1, 0, 1}
        # Sparsity helps keep B small
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
        
        # Compute B_{(i,k),c}
        # B[i][k][c]
        B = [[[0]*r for _ in range(p)] for _ in range(n)]
        
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    s = 0
                    for j in range(m):
                        s += U[i*m+j][c] * V[j*p+k][c]
                    B[i][k][c] = s
        
        # Check if we can solve for W with integer/half-integer entries
        W = [[0]*r for _ in range(p*n)]
        valid = True
        for i in range(n):
            for k in range(p):
                row_idx = k*n + i
                b_row = B[i][k]
                # Find c where b_row[c] is in {1, -1, 2, -2}
                found_c = -1
                found_val = 0
                for c in range(r):
                    if b_row[c] in (1, -1, 2, -2):
                        found_c = c
                        found_val = b_row[c]
                        break
                
                if found_c == -1:
                    valid = False
                    break
                
                # Set W[row_idx, found_c] = 1 / found_val
                # Since found_val is +/-1 or +/-2, 1/found_val is +/-1 or +/-0.5
                # This is a half-integer.
                W[row_idx][found_c] = Fraction(1, found_val)
            
            if not valid:
                break
        
        if valid:
            # Verify the full decomposition
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * W[k*n+i][c]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Convert fractions to float for output
                # The problem accepts integer/half-integer entries.
                # Returning floats is standard for these interfaces if they are exact.
                U_out = [[float(x) for x in row] for row in U]
                V_out = [[float(x) for x in row] for row in V]
                W_out = [[float(x) for x in row] for row in W]
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook
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
