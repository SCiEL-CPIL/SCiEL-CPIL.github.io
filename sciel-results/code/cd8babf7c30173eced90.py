import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = nm * p # 45
    
    # Fallback: Schoolbook
    def get_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
        return U, V, W

    # Attempt: Laderman (3x3x3) + Schoolbook (3x3x2)
    # Rank 23 + 18 = 41
    def get_laderman_schoolbook():
        # Laderman coefficients for 3x3x3
        # U_l: 9x23, V_l: 9x23, W_l: 9x23
        # Indices: 0..8 for 3x3 matrices
        # This is a known set of Laderman coefficients.
        # If incorrect, verification will fail.
        
        U_l = [
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0: (0,0)
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1: (0,1)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2: (0,2)
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 3: (1,0)
            [0, 1, 0, 1, 1, 1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], # 4: (1,1)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # 5: (1,2)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # 6: (2,0)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], # 7: (2,1)
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 8: (2,2)
        ]
        
        # Note: The above U_l is a guess/placeholders. 
        # I need the actual Laderman coefficients.
        # Since I don't have them verified, I will use a different strategy.
        
        # Strategy: Use the schoolbook for 45.
        # Is there a way to get 44?
        # No obvious way.
        
        return None

    U, V, W = get_schoolbook()
    return {"U": U, "V": V, "W": W}
