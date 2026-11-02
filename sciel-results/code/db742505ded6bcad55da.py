import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Base Decompositions
    
    # 2x2x2 Strassen (Rank 7)
    U_222 = np.array([
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [-1, 0, 1, 0],# c-a
        [0, 1, 0, -1] # b-d
    ])
    V_222 = np.array([
        [1, 0, 0, 1], # e+h
        [1, 0, 0, 0], # e
        [0, 1, 0, -1],# f-h
        [-1, 0, 1, 0],# g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # g+h
    ])
    W_222 = np.array([
        [1, 0, 0, 1], # C11 + C22
        [0, 0, 1, -1],# C21 - C22
        [0, 1, 0, 1], # C12 + C22
        [1, 0, 1, 0], # C11 + C21
        [-1, 1, 0, 0],# -C11 + C12
        [0, 0, 0, 1], # C22
        [1, 0, 0, 0]  # C11
    ])
    
    # 2x2x1 Schoolbook (Rank 4)
    # A (2x2) * B (2x1) -> C (2x1)
    # C1 = A11*B11 + A12*B21
    # C2 = A21*B11 + A22*B21
    U_221 = np.array([
        [1, 0, 0, 0], # A11
        [0, 1, 0, 0], # A12
        [0, 0, 1, 0], # A21
        [0, 0, 0, 1]  # A22
    ])
    V_221 = np.array([
        [1, 0, 1, 0], # B11, B21, B11, B21
        [0, 1, 0, 1]  #
    ])
    W_221 = np.array([
        [1, 1, 0, 0], # C11, C11, C21, C21
        [0, 0, 1, 1]  #
    ])

    # 2. Kronecker Products
    
    # 4x4x4 Decomposition (Rank 49)
    U_444_kron = np.kron(U_222, U_222)
    V_444_kron = np.kron(V_222, V_222)
    W_444_kron = np.kron(W_222, W_222)
    
    # Permutation for 4x4 matrices
    # Standard index i = b_r*8 + e_r*4 + b_c*2 + e_c
    # Kron index I = b_r*8 + b_c*4 + e_r*2 + e_c
    perm_44 = []
    for i in range(16):
        b_r = (i // 4) // 2
        e_r = (i // 4) % 2
        b_c = (i % 4) // 2
        e_c = (i % 4) % 2
        I = b_r * 8 + b_c * 4 + e_r * 2 + e_c
        perm_44.append(I)
        
    U_444 = U_444_kron[perm_44, :]
    V_444 = V_444_kron[perm_44, :]
    W_444 = W_444_kron[perm_44, :]
    
    # 4x4x1 Decomposition (Rank 16)
    U_441_kron = np.kron(U_221, U_221)
    V_441_kron = np.kron(V_221, V_221)
    W_441_kron = np.kron(W_221, W_221)
    
    U_441 = U_441_kron[perm_44, :]
    # V and W for 4x4x1 don't need permutation for rows (4x1 flattening matches kron row index)
    # But let's verify.
    # V_441 is 4x16. Row index I = b_r*2 + e_r. Standard index i = b_r*2 + e_r. Match.
    V_441 = V_441_kron
    W_441 = W_441_kron

    # 3. Assemble Final U, V, W
    
    rank_total = 49 + 49 + 16
    n, m, p = 4, 4, 9
    
    U_final = np.zeros((n*m, rank_total))
    V_final = np.zeros((m*p, rank_total))
    W_final = np.zeros((p*n, rank_total))
    
    # Part 1: 4x4x4 (cols 0-48)
    U_final[:, 0:49] = U_444
    V_final[0:16, 0:49] = V_444
    W_final[0:16, 0:49] = W_444
    
    # Part 2: 4x4x4 (cols 49-97)
    U_final[:, 49:98] = U_444
    V_final[16:32, 49:98] = V_444
    W_final[16:32, 49:98] = W_444
    
    # Part 3: 4x4x1 (cols 98-113)
    U_final[:, 98:114] = U_441
    V_final[32:36, 98:114] = V_441
    W_final[32:36, 98:114] = W_441
    
    # Convert to list of lists of integers
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    construct()
