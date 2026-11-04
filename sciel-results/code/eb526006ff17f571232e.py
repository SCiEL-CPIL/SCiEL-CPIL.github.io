"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    
    # Initialize U, V, W as 16x49 matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition vectors
    # u2[t] is vector of size 4 for left input
    # v2[t] is vector of size 4 for right input
    # w2[t] is vector of size 4 for output
    # t = 0..6 corresponding to P1..P7
    
    u2 = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    v2 = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, 0, 1, -1],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    w2 = [
        [1, 0, 0, 1], # P1 -> C00, C11
        [0, 0, 1, 0], # P2 -> C10
        [0, 1, 0, 1], # P3 -> C01, C11
        [1, 0, 1, 0], # P4 -> C00, C10
        [-1, 1, 0, 0],# P5 -> C00(-), C01(+)
        [0, 0, 0, 1], # P6 -> C11
        [1, 0, 0, 0]  # P7 -> C00
    ]
    
    # Define the 7 Strassen steps for 4x4
    # Each step S has:
    # L_blocks: dict mapping block index (0..3) to coefficient
    # R_blocks: dict mapping block index (0..3) to coefficient
    # C_blocks: dict mapping block index (0..3) to coefficient
    
    # Block indices: 00->0, 01->1, 10->2, 11->3
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    steps = []
    
    # S1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
    steps.append({
        'L': {0: 1, 3: 1},
        'R': {0: 1, 3: 1},
        'C': {0: 1, 3: 1}
    })
    
    # S2: (A10+A11)B00 -> C10 +=
    steps.append({
        'L': {2: 1, 3: 1},
        'R': {0: 1},
        'C': {2: 1}
    })
    
    # S3: A00(B01-B11) -> C01 +=, C11 +=
    steps.append({
        'L': {0: 1},
        'R': {1: 1, 3: -1},
        'C': {1: 1, 3: 1}
    })
    
    # S4: A11(B10-B00) -> C00 +=, C10 +=
    steps.append({
        'L': {3: 1},
        'R': {2: 1, 0: -1},
        'C': {0: 1, 2: 1}
    })
    
    # S5: (A00+A01)B11 -> C00 -=, C01 +=
    steps.append({
        'L': {0: 1, 1: 1},
        'R': {3: 1},
        'C': {0: -1, 1: 1}
    })
    
    # S6: (A10-A00)(B00+B01) -> C11 +=
    steps.append({
        'L': {2: 1, 0: -1},
        'R': {0: 1, 1: 1},
        'C': {3: 1}
    })
    
    # S7: (A01-A11)(B10+B11) -> C00 +=
    steps.append({
        'L': {1: 1, 3: -1},
        'R': {2: 1, 3: 1},
        'C': {0: 1}
    })
    
    col = 0
    for s_idx, step in enumerate(steps):
        for t_idx in range(7):
            # Fill U
            for block_a, coeff_a in step['L'].items():
                # block_a is 0..3
                # local indices in block: 0..3
                for local_idx in range(4):
                    val = coeff_a * u2[t_idx][local_idx]
                    if val != 0:
                        # Global index in U: block_a * 4 + local_idx
                        U[block_a * 4 + local_idx][col] += val
            
            # Fill V
            for block_b, coeff_b in step['R'].items():
                for local_idx in range(4):
                    val = coeff_b * v2[t_idx][local_idx]
                    if val != 0:
                        V[block_b * 4 + local_idx][col] += val
            
            # Fill W
            for block_c, coeff_c in step['C'].items():
                for local_idx in range(4):
                    val = coeff_c * w2[t_idx][local_idx]
                    if val != 0:
                        W[block_c * 4 + local_idx][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
