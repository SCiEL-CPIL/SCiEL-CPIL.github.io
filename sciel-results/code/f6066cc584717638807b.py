import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen terms for 2x2 * 2x2
    # Each term is a tuple of (x_coeffs, y_coeffs, z_coeffs)
    # x: (i', j') -> coeff
    # y: (j', k') -> coeff
    # z: (i', k') -> coeff
    
    strassen_terms = [
        # t=0: (x00+x11)(y00+y11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # t=1: (x10+x11)y00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # t=2: x00(y01-y11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # t=3: x11(y10-y00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # t=4: (x00+x01)y11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # t=5: (x10-x00)(y00+y01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # t=6: (x01-x11)(y10+y11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    # The problem decomposes into 8 independent 2x2x2x2 blocks.
    # A is 2x4. Split into 2 blocks of 2x2: A_a (a=0,1)
    # B is 4x8. Split into 4 blocks of 2x2 per A block?
    # Actually, B is split into 4 blocks of 2x2 along the row dimension (matching A's cols) and 4 blocks along col dimension?
    # Let's trace the indices carefully.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # i in 0..1, j in 0..3, k in 0..7.
    # Split j into j=2a + j' (a in 0..1, j' in 0..1)
    # Split k into k=2b + k' (b in 0..3, k' in 0..1)
    # A[i, 2a+j'] is the element.
    # B[2a+j', 2b+k'] is the element.
    # C[i, 2b+k'] gets contribution from A[i, 2a+j'] * B[2a+j', 2b+k'].
    # This is a product of A_block_a (2x2) and B_block_{a,b} (2x2).
    # For each a (0,1) and b (0,1,2,3), we have a 2x2x2x2 multiplication.
    # Total 2 * 4 = 8 blocks.
    
    for a in range(2):
        for b in range(4):
            for t in range(7):
                x_c, y_c, z_c = strassen_terms[t]
                
                # Update U
                # U index: i * m + j
                # j = 2a + j'
                for (i_prime, j_prime), coeff in x_c.items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx][col] += coeff
                
                # Update V
                # V index: j * p + k
                # j = 2a + j'
                # k = 2b + k'
                for (j_prime, k_prime), coeff in y_c.items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx][col] += coeff
                
                # Update W
                # W index: k * n + i
                # k = 2b + k'
                # i = i'
                for (i_prime, k_prime), coeff in z_c.items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
