import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 * 2x2
    # x: (i', j') -> coeff
    # y: (j', k') -> coeff
    # z: (i', k') -> coeff
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    col = 0
    for a in range(2):  # A block column: 2a, 2a+1
        for b in range(4):  # B column block: 2b, 2b+1
            for t in range(7):
                term = strassen_terms[t]
                
                # X part: A[i', 2a+j']
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx, col] += coeff
                
                # Y part: B[2a+j', 2b+k']
                # Note: B row index is j'. In the block structure, 
                # A_L corresponds to A rows 0-1, cols 0-1. 
                # The "j'" in Strassen is the inner index of the 2x2 block.
                # For A_L, inner index 0 maps to global col 0, 1 maps to 1.
                # For B, the row index must match.
                # B row 2a+j'? 
                # Wait, A_L is cols 0,1. So j' is 0,1.
                # B rows corresponding to A_L are 0,1.
                # So B row index is 2a + j'? 
                # If a=0, A_L is cols 0,1. B rows 0,1.
                # If a=1, A_R is cols 2,3. B rows 2,3.
                # So B row index is indeed 2a + j'.
                for (j_prime, k_prime), coeff in term['y'].items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx, col] += coeff
                
                # Z part: C[i', 2b+k'] -> W[2b+k', i']
                # Output C is 2x8.
                # Block b corresponds to cols 2b, 2b+1.
                # i' is 0,1.
                for (i_prime, k_prime), coeff in term['z'].items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx, col] += coeff
                
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
