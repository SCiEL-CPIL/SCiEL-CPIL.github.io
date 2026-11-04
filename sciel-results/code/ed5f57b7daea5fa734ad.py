"""Matrix multiplication 2x4x4: rank-28 Strassen-based decomposition."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28

    # U: (n*m, rank) = (8, 28); V: (m*p, rank) = (16, 28); W: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen M definitions: xcoeff, ycoeff, zcoeff over local indices (00,01,10,11)
    Mx = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, -1, 0, 1], # M7
    ]
    My = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    Mz = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0],  # M7
    ]

    # Block mappings: block index -> (x_offset_a, y_offset_a, y_offset_b, z_offset_b)
    # x: A[i', a] where a = a' + x_off_a
    # y: B[a, b] where a = a' + y_off_a, b = b' + y_off_b
    # z: C[i', b] where b = b' + z_off_b
    blocks = [
        (0, 0, 0, 0),  # Block 0: A1*B11
        (2, 2, 0, 0),  # Block 1: A2*B21
        (0, 0, 0, 2),  # Block 2: A1*B12
        (2, 2, 2, 2),  # Block 3: A2*B22
    ]

    r = 0
    for (x_off_a, y_off_a, y_off_b, z_off_b) in blocks:
        for m_idx in range(7):
            # Fill U: for local (i', a'), global A index (i', a' + x_off_a)
            for i_prime in range(2):
                for a_prime in range(2):
                    l = i_prime * 2 + a_prime
                    a_global = a_prime + x_off_a
                    U[i_prime * m + a_global][r] += Mx[m_idx][l]
            # Fill V: for local (a', b'), global B index (a' + y_off_a, b' + y_off_b)
            for a_prime in range(2):
                for b_prime in range(2):
                    l = a_prime * 2 + b_prime
                    a_global = a_prime + y_off_a
                    b_global = b_prime + y_off_b
                    V[a_global * p + b_global][r] += My[m_idx][l]
            # Fill W: for local (i', b'), global C index (i', b' + z_off_b)
            for i_prime in range(2):
                for b_prime in range(2):
                    l = i_prime * 2 + b_prime
                    b_global = b_prime + z_off_b
                    W[b_global * n + i_prime][r] += Mz[m_idx][l]
            r += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
