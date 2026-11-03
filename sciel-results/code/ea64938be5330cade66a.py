"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen's algorithm for 2x2x2 blocks
    # Partition m=4 into two blocks of 2
    # Partition p=6 into three blocks of 2
    # Total 6 blocks of 2x2x2
    # Each 2x2x2 has rank 7
    # Total rank 6 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # t: term index 0..6
    # a,b,c: local indices 0..1
    # U[a, b, t], V[b, c, t], W[c, a, t]
    # Note: W index is k*n+i, so c is k_local, a is i_local.
    # In our global mapping, W index is 2*k_global + i_global.
    # k_global = k_base + c, i_global = a.
    # So W[2*(k_base+c) + a, r]
    
    # U: i*m + j. i=a, j=j_base+b.
    # V: j*p + k. j=j_base+b, k=k_base+c.
    
    # Define the 7 terms
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Coefficients stored as list of (index, value) for U, V, W
    # U indices: a*2 + b (local 2x2) -> mapped to global
    # V indices: b*2 + c (local 2x2) -> mapped to global
    # W indices: c*2 + a (local 2x2, k*n+i) -> mapped to global
    
    strassen_terms = [
        { # M1
            'U': [(0, 1), (3, 1)], # a=0,b=0; a=1,b=1
            'V': [(0, 1), (3, 1)], # b=0,c=0; b=1,c=1
            'W': [(0, 1), (3, 1)]  # c=0,a=0; c=1,a=1
        },
        { # M2
            'U': [(2, 1), (3, 1)], # a=1,b=0; a=1,b=1
            'V': [(0, 1)],         # b=0,c=0
            'W': [(2, 1), (3, -1)] # c=0,a=1; c=1,a=1
        },
        { # M3
            'U': [(0, 1)],         # a=0,b=0
            'V': [(1, 1), (3, -1)],# b=0,c=1; b=1,c=1
            'W': [(1, 1), (3, 1)]  # c=1,a=0; c=1,a=1
        },
        { # M4
            'U': [(3, 1)],         # a=1,b=1
            'V': [(2, 1), (0, -1)],# b=1,c=0; b=0,c=0
            'W': [(2, 1), (3, 1)]  # c=0,a=1; c=1,a=1
        },
        { # M5
            'U': [(0, 1), (1, 1)], # a=0,b=0; a=0,b=1
            'V': [(3, 1)],         # b=1,c=1
            'W': [(1, 1), (3, -1)] # c=1,a=0; c=1,a=1
        },
        { # M6
            'U': [(2, 1), (0, -1)],# a=1,b=0; a=0,b=0
            'V': [(0, 1), (1, 1)], # b=0,c=0; b=0,c=1
            'W': [(3, 1)]          # c=1,a=1
        },
        { # M7
            'U': [(1, 1), (3, -1)],# a=0,b=1; a=1,b=1
            'V': [(2, 1), (3, 1)], # b=1,c=0; b=1,c=1
            'W': [(0, 1)]          # c=0,a=0
        }
    ]
    
    for s in range(6):
        j_group = s // 3
        k_group = s % 3
        j_base = 2 * j_group
        k_base = 2 * k_group
        
        for t in range(7):
            r = s * 7 + t
            term = strassen_terms[t]
            
            for idx, val in term['U']:
                a = idx // 2
                b = idx % 2
                i = a
                j = j_base + b
                u_idx = i * m + j
                U[u_idx][r] += val
            
            for idx, val in term['V']:
                b = idx // 2
                c = idx % 2
                j = j_base + b
                k = k_base + c
                v_idx = j * p + k
                V[v_idx][r] += val
            
            for idx, val in term['W']:
                c = idx // 2
                a = idx % 2
                k = k_base + c
                i = a
                w_idx = k * n + i
                W[w_idx][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
