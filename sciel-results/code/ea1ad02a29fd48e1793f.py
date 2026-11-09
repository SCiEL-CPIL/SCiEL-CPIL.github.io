import time

def construct():
    n = 10
    # Generate L1: standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Generate L2: constructed to be orthogonal to L1
    L2 = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            r2 = (i + j) % 2
            r5 = (i % 5 + 2 * j % 5) % 5
            # Inverse CRT: x = 6*r5 + 5*r2 (mod 10)
            L2[i][j] = (6 * r5 + 5 * r2) % 10
            
    # Verify types are Python int
    # L1 and L2 contain Python ints from arithmetic operations on ints
    
    return [L1, L2]
