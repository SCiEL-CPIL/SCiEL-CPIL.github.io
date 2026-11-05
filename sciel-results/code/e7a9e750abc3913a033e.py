import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n
    block_size = 3
    num_blocks = 2

    total_rank = 7 * 27

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    # Strassen steps defined as (A_terms, B_terms, C_terms)
    # Each term is (I, J, coeff)
    # A_terms: terms for the left matrix (A block combinations)
    # B_terms: terms for the right matrix (B block combinations)  
    # C_terms: terms for the result matrix (C block combinations)
    steps = [
        # P1 = (A11 + A22) * (B11 + B22)
        # C11 += P1, C22 += P1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # P2 = (A21 + A22) * B11
        # C21 += P2, C22 -= P2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # P3 = A11 * (B12 - B22)
        # C12 += P3, C22 += P3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # P4 = A22 * (B21 - B11)
        # C11 += P4, C21 += P4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # P5 = (A11 + A12) * B22
        # C11 -= P5, C12 += P5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # P6 = (A21 - A11) * (B11 + B12)
        # C22 += P6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7 = (A12 - A22) * (B21 + B22)
        # C11 += P7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    bs = block_size
    col_idx = 0
    
    # Pre-compute all index mappings for speed
    # For each Strassen step and each (r,s,t) in 0..2, compute the exact indices
    
    # Process each Strassen step
    for step_idx, step in enumerate(steps):
        a_terms, b_terms, c_terms = step
        for r, s, t in itertools.product(range(bs), repeat=3):
            col = step_idx * 27 + (r * 9 + s * 3 + t)
            
            # Process U: A part
            # U row index: (I * bs + r) * dim + (J * bs + s)
            for I, J, ca in a_terms:
                u_idx = (I * bs + r) * dim + (J * bs + s)
                U[u_idx, col] += ca

            # Process V: B part
            # V row index: (J * bs + s) * dim + (K * bs + t)
            for J, K, cb in b_terms:
                v_idx = (J * bs + s) * dim + (K * bs + t)
                V[v_idx, col] += cb

            # Process W: C part
            # W row index: (K * bs + t) * dim + (I * bs + r)
            # Note: W[k*n+i, r] where k is column index, i is row index
            # For C block (I, J), local (i, j) maps to global row I*bs+i, col J*bs+j
            # W index = col_global * n + row_global = (J*bs+t)*dim + (I*bs+r)
            for I, K, cc in c_terms:
                w_idx = (K * bs + t) * dim + (I * bs + r)
                W[w_idx, col] += cc

            col_idx = col + 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
