import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    # Initialize U, V, W with integer dtype
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    # Strassen 2x2x2 decomposition details
    # U_2, V_2, W_2 are 4x7
    U_2 = np.zeros((4, 7), dtype=np.int32)
    V_2 = np.zeros((4, 7), dtype=np.int32)
    W_2 = np.zeros((4, 7), dtype=np.int32)
    
    # ... (rest of the code remains the same)
