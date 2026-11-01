"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen's algorithm for 4 blocks of 2x2
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b10 - b11)
    # m4 = a11(b01 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Coefficients for A (u)
    u_coeffs = [
        [1, 0, 0, 1], # m1: a00, a01, a10, a11 -> 1, 0, 0, 1
        [0, 0, 1, 1], # m2: 0, 0, 1, 1
        [1, 0, 0, 0], # m3: 1, 0, 0, 0
        [0, 0, 0, 1], # m4: 0, 0, 0, 1
        [1, 1, 0, 0], # m5: 1, 1, 0, 0
        [-1, 0, 1, 0],# m6: -1, 0, 1, 0
        [0, 1, 0, -1] # m7: 0, 1, 0, -1
    ]
    
    # Coefficients for B (v) for a 2x2 block
    # b00, b01, b10, b11
    v_coeffs = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 0, 1, -1],# m3
        [0, 1, 0, -1],# m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    
    # Coefficients for C (w)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    w_coeffs = [
        [1, 0, 0, 0], # m1 -> C00
        [0, 0, 0, 0], # m2
        [0, 0, 0, 0], # m3
        [0, 0, 0, 0], # m4
        [0, 0, 0, 0], # m5
        [0, 0, 0, 0], # m6
        [0, 0, 0, 0]  # m7
    ]
    # This is not how it works. w_coeffs should map each m_s to C entries.
    # C_{i'k'} = sum_s w_{i'k',s} m_s
    # So for each s, we add w_{i'k',s} to W.
    
    # Let's define w_contrib[s][i][k]
    w_contrib = [
        [[1, 0], [0, 0]], # m1: C00 += 1
        [[0, 0], [1, 0]], # m2: C10 += 1
        [[0, 1], [0, 0]], # m3: C01 += 1
        [[1, 0], [0, 1]], # m4: C00 += 1, C11 += 1
        [[-1, 0], [0, 0]],# m5: C00 -= 1
        [[0, 0], [0, 1]], # m6: C11 += 1
        [[1, 0], [0, 0]]  # m7: C00 += 1
    ]
    # Check C11: m1 - m2 + m3 + m6?
    # My w_contrib for m2 is C10.
    # Standard Strassen:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Let's re-verify w_contrib based on this.
    # m1: C00 += 1, C11 += 1
    # m2: C10 += 1, C11 -= 1
    # m3: C01 += 1, C11 += 1
    # m4: C00 += 1, C10 += 1
    # m5: C00 -= 1, C01 += 1
    # m6: C11 += 1
    # m7: C00 += 1
    
    w_contrib = [
        [[1, 0], [1, 0]], # m1: C00+=1, C10=0, C11=0? No.
        # w_contrib[s][i][k]
        # i=0, k=0 -> C00
        # i=0, k=1 -> C01
        # i=1, k=0 -> C10
        # i=1, k=1 -> C11
        
        [[1, 0], [0, 1]], # m1: C00+=1, C11+=1
        [[0, 0], [1, -1]],# m2: C10+=1, C11-=1
        [[0, 1], [0, 1]], # m3: C01+=1, C11+=1
        [[1, 0], [1, 0]], # m4: C00+=1, C10+=1
        [[-1, 1], [0, 0]],# m5: C00-=1, C01+=1
        [[0, 0], [0, 1]], # m6: C11+=1
        [[1, 0], [0, 0]]  # m7: C00+=1
    ]
    
    # 4 blocks of 2 columns each
    for b in range(4):
        for s in range(7):
            t = b * 7 + s
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][t] = u_coeffs[s][i * 2 + j]
            
            # Fill V
            # Block b covers columns 2*b and 2*b+1
            # V index j*p + k
            for j in range(2):
                for k_local in range(2):
                    k = 2 * b + k_local
                    V[j * p + k][t] = v_coeffs[s][j * 2 + k_local]
            
            # Fill W
            # W index k*n + i
            for i in range(2):
                for k_local in range(2):
                    k = 2 * b + k_local
                    W[k * n + i][t] = w_contrib[s][i][k_local]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
