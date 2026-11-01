import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # L1: a+d -> [1, 0, 0, 1]
    # L2: c+d -> [0, 0, 1, 1]
    # L3: a   -> [1, 0, 0, 0]
    # L4: d   -> [0, 0, 0, 1]
    # L5: a+b -> [1, 1, 0, 0]
    # L6: c-a -> [-1, 0, 1, 0]
    # L7: b-d -> [0, 1, 0, -1]
    
    U_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # For each block b (0, 1, 2) and each Strassen term s (1..7)
    for b in range(3):
        for s in range(1, 8):
            r = b * 7 + (s - 1)
            
            # Set U column r
            for idx, val in enumerate(U_forms[s-1]):
                U[idx][r] = val
                
            # Set V column r
            # V is 12x21. Index j*p + k. j in {0,1}, k in {0..5}.
            # Block b uses k in {2b, 2b+1}.
            # Local k0 = 2b, k1 = 2b+1.
            
            if s == 1:
                # L1(B) = B_{0, k0} + B_{0, k1}
                V[0 * p + 2*b][r] = 1
                V[0 * p + 2*b + 1][r] = 1
            elif s == 2:
                # L2(B) = B_{0, k0}
                V[0 * p + 2*b][r] = 1
            elif s == 3:
                # L3(B) = B_{0, k1} - B_{0, k0}
                V[0 * p + 2*b][r] = -1
                V[0 * p + 2*b + 1][r] = 1
            elif s == 4:
                # L4(B) = B_{1, k0} - B_{0, k0}
                V[0 * p + 2*b][r] = -1
                V[1 * p + 2*b][r] = 1
            elif s == 5:
                # L5(B) = B_{1, k1}
                V[1 * p + 2*b + 1][r] = 1
            elif s == 6:
                # L6(B) = B_{0, k0} + B_{0, k1}
                V[0 * p + 2*b][r] = 1
                V[0 * p + 2*b + 1][r] = 1
            elif s == 7:
                # L7(B) = B_{1, k0} + B_{1, k1}
                V[1 * p + 2*b][r] = 1
                V[1 * p + 2*b + 1][r] = 1
            
            # Set W column r
            # W is 12x21. Index k*n + i. k in {0..5}, i in {0,1}.
            # W[k*n + i, r] is coeff of M_r in C[i, k].
            
            # C[0, k0] = M1 + M4 - M5 + M7
            # C[1, k0] = M2 + M4
            # C[0, k1] = M3 + M5
            # C[1, k1] = M1 - M2 + M3 + M6
            
            if s == 1:
                W[(2*b) * n + 0][r] = 1
                W[(2*b+1) * n + 1][r] = 1
            elif s == 2:
                W[(2*b) * n + 1][r] = 1
                W[(2*b+1) * n + 1][r] = -1
            elif s == 3:
                W[(2*b+1) * n + 0][r] = 1
                W[(2*b+1) * n + 1][r] = 1
            elif s == 4:
                W[(2*b) * n + 0][r] = 1
                W[(2*b) * n + 1][r] = 1
            elif s == 5:
                W[(2*b) * n + 0][r] = -1
                W[(2*b+1) * n + 0][r] = 1
            elif s == 6:
                W[(2*b+1) * n + 1][r] = 1
            elif s == 7:
                W[(2*b) * n + 0][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
