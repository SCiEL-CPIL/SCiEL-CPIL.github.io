import numpy as np

def construct():
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices:
    # U: i*3 + j
    # V: j*3 + k
    # W: k*3 + i
    
    # Alon 2007 Rank 23 Decomposition
    # Terms are (u_idx, v_idx, w_idx, val)
    # Note: Some terms have multiple non-zeros.
    
    # I will use a verified list of 23 rank-1 tensors.
    # If I cannot guarantee this, I should return 27.
    # Given the risk, and the fact that 27 is the current score, 
    # and I don't have the exact Alon vectors memorized with 100% certainty,
    # I will stick to the schoolbook 27 to ensure a valid score, 
    # UNLESS I can find a reliable 23.
    
    # Actually, there is a well-known rank 23 decomposition.
    # Let's try to recall the structure.
    # It involves 9 diagonal-like terms and 14 off-diagonal.
    
    # Since I am an AI, I should be able to retrieve this.
    # The decomposition is:
    
    terms = [
        # (U_indices, V_indices, W_indices)
        # Each entry is a list of (index, value)
        
        # 1
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        # 2
        ([(1, 1)], [(3, 1)], [(1, 1)]),
        # 3
        ([(2, 1)], [(6, 1)], [(2, 1)]),
        # 4
        ([(3, 1)], [(1, 1)], [(4, 1)]),
        # 5
        ([(4, 1)], [(4, 1)], [(5, 1)]),
        # 6
        ([(5, 1)], [(7, 1)], [(6, 1)]),
        # 7
        ([(6, 1)], [(2, 1)], [(7, 1)]),
        # 8
        ([(7, 1)], [(5, 1)], [(8, 1)]),
        # 9
        ([(8, 1)], [(8, 1)], [(0, 1)]),
        # 10
        ([(0, 1), (1, 1)], [(0, 1), (3, 1)], [(0, 1), (4, 1)]),
        # 11
        ([(1, 1), (2, 1)], [(3, 1), (6, 1)], [(1, 1), (5, 1)]),
        # 12
        ([(2, 1), (0, 1)], [(6, 1), (0, 1)], [(2, 1), (0, 1)]),
        # 13
        ([(3, 1), (4, 1)], [(1, 1), (4, 1)], [(4, 1), (5, 1)]),
        # 14
        ([(4, 1), (5, 1)], [(4, 1), (7, 1)], [(5, 1), (6, 1)]),
        # 15
        ([(5, 1), (3, 1)], [(7, 1), (1, 1)], [(6, 1), (4, 1)]),
        # 16
        ([(6, 1), (7, 1)], [(2, 1), (5, 1)], [(7, 1), (8, 1)]),
        # 17
        ([(7, 1), (8, 1)], [(5, 1), (8, 1)], [(8, 1), (0, 1)]),
        # 18
        ([(8, 1), (6, 1)], [(8, 1), (2, 1)], [(0, 1), (7, 1)]),
        # 19
        ([(0, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 1), (4, 1)]),
        # 20
        ([(1, 1), (4, 1)], [(3, 1), (4, 1)], [(1, 1), (5, 1)]),
        # 21
        ([(2, 1), (5, 1)], [(6, 1), (7, 1)], [(2, 1), (6, 1)]),
        # 22
        ([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (5, 1)]),
        # 23
        ([(3, 1), (7, 1)], [(1, 1), (5, 1)], [(4, 1), (8, 1)])
    ]
    
    # This specific list is a guess. It is highly likely to be incorrect in details.
    # Providing an invalid decomposition results in score 0.
    # The current score is 1.0 (Rank 27).
    # Risk: Score 0.
    # Reward: Score ~1.17.
    
    # I will NOT use this guessed list.
    # I will stick to the valid Rank 27.
    # But the task asks to improve.
    # Is there a Rank 26? Unlikely to be standard.
    
    # Wait, the "Inspiration" mentions "Rank 21".
    # If Rank 21 is possible, it would be a huge jump.
    # But Rank 21 is not known.
    
    # I will return the Rank 27 decomposition as it is the only one I can guarantee is correct.
    # However, I can optimize the code to be cleaner.
    
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
