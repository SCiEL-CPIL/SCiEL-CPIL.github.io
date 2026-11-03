"""Strassen-based decomposition for 4x4 * 4x8 matrix multiplication, rank 112."""
import time
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = 4, 4, 8
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
    ]
    
    for ci in range(2):
        for cj in range(4):
            for k in range(2):
                a_rb, a_cb = 2 * ci, 2 * k
                b_rb, b_cb = 2 * k, 2 * cj
                c_rb, c_cb = 2 * ci, 2 * cj
                
                for x_c, y_c, z_c in strassen_terms:
                    for (lr, lc), cf in x_c.items():
                        U[(a_rb + lr) * m + (a_cb + lc)][col] += cf
                    for (lr, lc), cf in y_c.items():
                        V[(b_rb + lr) * p + (b_cb + lc)][col] += cf
                    for (lr, lc), cf in z_c.items():
                        W[(c_cb + lc) * n + (c_rb + lr)][col] += cf
                    col += 1
    
    assert col == rank
    
    if time.time() - start_time < 100:
        _verify(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, trials=10):
    rng = random.Random(42)
    for _ in range(trials):
        A = [[rng.randint(-9, 9) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-9, 9) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C2 = [[0] * p for _ in range(n)]
        for c in range(len(U[0])):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C2[i][k] += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        for i in range(n):
            for k in range(p):
                if C[i][k] != C2[i][k]:
                    raise ValueError(f"Mismatch at ({i},{k}): {C[i][k]} vs {C2[i][k]}")

# EVOLVE-BLOCK-END
