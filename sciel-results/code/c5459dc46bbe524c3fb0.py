import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    rank = 29
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    # Helper to get global indices
    def get_I(i, j):
        return i * m + j
    def get_J(j, k):
        return j * p + k
    def get_K(k, i):
        return k * n + i

    # --- Part 1: 2x2x2 (Strassen) ---
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global I: 0,1,3,4
    # Global J: 0,1,5,6
    # Global K: 0,1,2,3
    
    # Strassen factors for 2x2x2
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # U factors (coeff of A_ij)
    # V factors (coeff of B_jk)
    # W factors (coeff of C_ik)
    
    # Local indices: i in 0,1; j in 0,1; k in 0,1
    # Global mapping:
    I_map = { (0,0): 0, (0,1): 1, (1,0): 3, (1,1): 4 }
    J_map = { (0,0): 0, (0,1): 1, (1,0): 5, (1,1): 6 }
    K_map = { (0,0): 0, (1,0): 1, (0,1): 2, (1,1): 3 } # Note: K = 2k+i. 
    # Check K_map:
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    # Correct.
    
    # Define the 7 terms
    # term: (U_dict, V_dict, W_dict)
    # U_dict: {(i,j): coeff}
    # V_dict: {(j,k): coeff}
    # W_dict: {(i,k): coeff}
    
    strassen_terms = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}), # C11=M1+..., C22=M1-...
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # C21=M2+..., C22=-M2
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # C12=M3+..., C22=M3
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # C11=M4+..., C21=M4
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # C11=-M5, C12=M5
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # C22=M6
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}) # C11=M7, C22=-M7
    ]
    
    # Wait, I need to verify the W coefficients carefully.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-define W dicts based on this.
    # W[ (i,k) ] = coeff for M_c in C_ik
    
    strassen_terms_corrected = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1})
    ]
    
    # Populate U, V, W for Part 1 (cols 0-6)
    for c, (U_dict, V_dict, W_dict) in enumerate(strassen_terms_corrected):
        for (i,j), coeff in U_dict.items():
            I_glob = I_map[(i,j)]
            U[I_glob, c] += coeff
        for (j,k), coeff in V_dict.items():
            J_glob = J_map[(j,k)]
            V[J_glob, c] += coeff
        for (i,k), coeff in W_dict.items():
            K_glob = K_map[(i,k)]
            W[K_glob, c] += coeff

    # --- Part 2: 2x2x3 (Schoolbook) ---
    # Indices: i in {0,1}, j in {0,1}, k in {2,3,4}
    # Rank 12. Cols 7-18.
    # I_map same as Part 1.
    # J_map: j in {0,1}, k in {2,3,4}
    # J = 5j + k
    # (0,2)->2, (0,3)->3, (0,4)->4
    # (1,2)->7, (1,3)->8, (1,4)->9
    J_map2 = {}
    for j in [0,1]:
        for k in [2,3,4]:
            J_map2[(j,k)] = get_J(j,k)
            
    # K_map: k in {2,3,4}, i in {0,1}
    # K = 2k + i
    K_map2 = {}
    for k in [2,3,4]:
        for i in [0,1]:
            K_map2[(k,i)] = get_K(k,i)
            
    col_offset = 7
    for i in [0,1]:
        for j in [0,1]:
            for k in [2,3,4]:
                c = col_offset
                col_offset += 1
                I_glob = get_I(i,j)
                J_glob = get_J(j,k)
                K_glob = get_K(k,i)
                U[I_glob, c] = 1
                V[J_glob, c] = 1
                W[K_glob, c] = 1

    # --- Part 3: 2x1x5 (Schoolbook) ---
    # Indices: i in {0,1}, j in {2}, k in {0,1,2,3,4}
    # Rank 10. Cols 19-28.
    I_map3 = { (0,2): get_I(0,2), (1,2): get_I(1,2) } # I = 3i+2 -> 2, 5
    J_map3 = { (2,k): get_J(2,k) for k in range(5) } # J = 10+k -> 10..14
    K_map3 = { (k,i): get_K(k,i) for k in range(5) for i in [0,1] }
    
    col_offset = 19
    for i in [0,1]:
        for k in range(5):
            c = col_offset
            col_offset += 1
            I_glob = get_I(i, 2)
            J_glob = get_J(2, k)
            K_glob = get_K(k, i)
            U[I_glob, c] = 1
            V[J_glob, c] = 1
            W[K_glob, c] = 1

    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
