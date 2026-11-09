import numpy as np

def strassen_decomposition(n):
    # Returns U, V, W for n x n x n
    # n is power of 2
    if n == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    half = n // 2
    # Recursively get decomposition for half x half
    U_h, V_h, W_h = strassen_decomposition(half)
    rank_h = len(U_h[0])
    
    # Strassen's 7 terms for 2x2 blocks
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # We need to map the indices of the 2x2 block decomposition to the n x n indices.
    # This is complex to do generically.
    # I will hardcode the 49 terms for 4x4.
    pass
