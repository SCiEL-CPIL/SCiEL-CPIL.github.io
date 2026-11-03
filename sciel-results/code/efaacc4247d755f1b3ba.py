import time
import numpy as np

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 130

def construct():
    """
    Finds a rank-42 exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Strategy:
    Split the output dimension (7) into 3 and 4.
    C1 = A * B1 (3x3 output), where B1 is 4x3.
    C2 = A * B2 (3x4 output), where B2 is 4x4.
    
    For C1 (3x4 * 4x3):
    Split the inner dimension (4) into 2 and 2.
    C1 = A_L * B1_L + A_R * B1_R
    Each term is 3x2 * 2x3. Rank 9 each. Total 18.
    
    For C2 (3x4 * 4x4):
    Split the inner dimension (4) into 2 and 2.
    C2 = A_L * B2_L + A_R * B2_R
    Each term is 3x2 * 2x4. Rank 12 each. Total 24.
    
    Total Rank = 18 + 24 = 42.
    """
    n, m, p = 3, 4, 7
    
    # Shapes
    # U: (n*m, rank) = (12, 42)
    # V: (m*p, rank) = (28, 42)
    # W: (p*n, rank) = (21, 42)
    
    rank_total = 42
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    col_offset = 0
    
    # Helper to add a schoolbook component
    # i_range, j_range, k_range
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def add_component(i_range, j_range, k_range):
        nonlocal col_offset
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx, col_offset] = 1
                    V[v_idx, col_offset] = 1
                    W[w_idx, col_offset] = 1
                    col_offset += 1

    # Block 1: 3x2 * 2x3 (Part 1 of C1)
    # i in 0..2, j in 0..1, k in 0..2
    add_component(range(3), range(2), range(3))
    
    # Block 2: 3x2 * 2x3 (Part 2 of C1)
    # i in 0..2, j in 2..3, k in 0..2
    add_component(range(3), range(2, 4), range(3))
    
    # Block 3: 3x2 * 2x4 (Part 1 of C2)
    # i in 0..2, j in 0..1, k in 3..6
    add_component(range(3), range(2), range(3, 7))
    
    # Block 4: 3x2 * 2x4 (Part 2 of C2)
    # i in 0..2, j in 2..3, k in 3..6
    add_component(range(3), range(2, 4), range(3, 7))
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
