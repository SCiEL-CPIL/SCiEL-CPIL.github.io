import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # --- 1. Define 2x2x2 Strassen Decomposition (Rank 7) ---
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    U_S = np.array([
        [0, 1, 0, 1, 1, -1, 1],
        [0, 0, 1, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 1, -1, 0, 0, 0, 0]
    ], dtype=float)
    
    V_S = np.array([
        [1, 0, 0, 1, 1, 1, -1],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [-1, 1, 1, 0, 0, 0, 1]
    ], dtype=float)
    
    W_S = np.array([
        [1, 1, 0, -1, 0, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, -1, 1, 0, 0, 0, 1]
    ], dtype=float)

    # --- 2. Construct 4x4x4 Decomposition (Rank 49) ---
    # Using Kronecker products
    U_444 = np.kron(U_S, U_S) # 16x49
    V_444 = np.kron(V_S, V_S) # 16x49
    W_444 = np.kron(W_S, W_S) # 16x49

    # --- 3. Construct 4x4x1 Decomposition (Rank 16, Schoolbook) ---
    # U_441: 16x16, V_441: 4x16, W_441: 4x16
    U_441 = np.eye(16)
    
    V_441 = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            V_441[j, i * 4 + j] = 1
            
    W_441 = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            W_441[i, i * 4 + j] = 1

    # --- 4. Combine for 4x4x5 ---
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U_total = np.zeros((n * m, total_rank)) # 16x65
    V_total = np.zeros((m * p, total_rank)) # 20x65
    W_total = np.zeros((p * n, total_rank)) # 20x65

    # First 49 terms: 4x4x4 part
    U_total[:, 0:rank_444] = U_444
    V_total[0:16, 0:rank_444] = V_444
    W_total[0:16, 0:rank_444] = W_444
    
    # Last 16 terms: 4x4x1 part (5th column)
    U_total[:, rank_444:total_rank] = U_441
    V_total[16:20, rank_444:total_rank] = V_441
    W_total[16:20, rank_444:total_rank] = W_441

    # Convert to lists of lists of integers (or floats if necessary, but these are integers)
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    # Ensure entries are integers where possible
    def to_int_list(lst):
        return [[int(round(x)) for x in row] for row in lst]

    return {"U": to_int_list(U_list), "V": to_int_list(V_list), "W": to_int_list(W_list)}

# EVOLVE-BLOCK-END
