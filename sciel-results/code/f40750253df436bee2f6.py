import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4 multiplication using Strassen (Rank 49)
    # We need U1, V1, W1 for 4x4 x 4
    # Part 2: 4x1 multiplication using Schoolbook (Rank 16)
    # We need U2, V2, W2 for 4x4 x 1
    
    # I will implement a helper to get Strassen matrices for 4x4
    # This is complex. Let's try to find a simpler decomposition.
    
    # If I can't implement Strassen correctly, I might fail.
    # Let's try to implement the rank 72 decomposition if I can find it.
    # But I don't have it.
    
    # Let's assume the rank 65 is achievable and try to implement it.
    pass
