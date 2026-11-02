import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    
    # We use the block Strassen approach.
    # Split B (2x12) into 6 blocks of size 2x2.
    # For each block, use Strassen's algorithm (rank 7).
    # Total rank = 6 * 7 = 42.
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # The tensor for 2x2x2 multiplication has rank 7.
    # We need to construct U, V, W matrices for this.
    
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (24, 42)
    # W shape: (p*n, rank) = (24, 42)
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each of the 6 blocks (corresponding to columns 2*b to 2*b+1 of B),
    # we apply Strassen's decomposition.
    
    # Strassen's 7 terms for one 2x2x2 block:
    # Term 1: (a+d)(e+h)
    # Term 2: (c+d)e
    # Term 3: a(f-h)
    # Term 4: d(g-e)
    # Term 5: (a+b)h
    # Term 6: (c-a)(e+f)
    # Term 7: (b-d)(g+h)
    
    # Indices for A (2x2): a=0, b=1, c=2, d=3
    # Indices for B block (2x2): e=0, f=1, g=2, h=3 (local indices)
    # Indices for C (2x2): c11=0, c12=1, c21=2, c22=3
    
    # For block b (0-indexed), the global indices for B are:
    # e -> 2*b, f -> 2*b+1, g -> 2*b+2, h -> 2*b+3
    # The global indices for C are:
    # c11 -> 2*b, c12 -> 2*b+1, c21 -> 2*b+2, c22 -> 2*b+3
    
    for b in range(p // 2):
        base = b * 7
        # Term 1: (a+d)(e+h)
        # U: a=0, d=3 -> U[0, base+0] += 1, U[3, base+0] += 1
        # V: e=2*b, h=2*b+3 -> V[2*b, base+0] += 1, V[2*b+3, base+0] += 1
        # W: c11=2*b, c22=2*b+3 -> W[2*b, base+0] += 1, W[2*b+3, base+0] += 1
        U[0, base + 0] += 1
        U[3, base + 0] += 1
        V[2 * b, base + 0] += 1
        V[2 * b + 3, base + 0] += 1
        W[2 * b, base + 0] += 1
        W[2 * b + 3, base + 0] += 1
        
        # Term 2: (c+d)e
        # U: c=2, d=3 -> U[2, base+1] += 1, U[3, base+1] += 1
        # V: e=2*b -> V[2*b, base+1] += 1
        # W: c21=2*b+2 -> W[2*b+2, base+1] += 1
        U[2, base + 1] += 1
        U[3, base + 1] += 1
        V[2 * b, base + 1] += 1
        W[2 * b + 2, base + 1] += 1
        
        # Term 3: a(f-h)
        # U: a=0 -> U[0, base+2] += 1
        # V: f=2*b+1, h=2*b+3 -> V[2*b+1, base+2] += 1, V[2*b+3, base+2] -= 1
        # W: c12=2*b+1 -> W[2*b+1, base+2] += 1, c22=2*b+3 -> W[2*b+3, base+2] += 1
        U[0, base + 2] += 1
        V[2 * b + 1, base + 2] += 1
        V[2 * b + 3, base + 2] -= 1
        W[2 * b + 1, base + 2] += 1
        W[2 * b + 3, base + 2] += 1
        
        # Term 4: d(g-e)
        # U: d=3 -> U[3, base+3] += 1
        # V: g=2*b+2, e=2*b -> V[2*b+2, base+3] += 1, V[2*b, base+3] -= 1
        # W: c11=2*b -> W[2*b, base+3] += 1, c21=2*b+2 -> W[2*b+2, base+3] += 1
        U[3, base + 3] += 1
        V[2 * b + 2, base + 3] += 1
        V[2 * b, base + 3] -= 1
        W[2 * b, base + 3] += 1
        W[2 * b + 2, base + 3] += 1
        
        # Term 5: (a+b)h
        # U: a=0, b=1 -> U[0, base+4] += 1, U[1, base+4] += 1
        # V: h=2*b+3 -> V[2*b+3, base+4] += 1
        # W: c11=2*b -> W[2*b, base+4] -= 1, c12=2*b+1 -> W[2*b+1, base+4] += 1
        U[0, base + 4] += 1
        U[1, base + 4] += 1
        V[2 * b + 3, base + 4] += 1
        W[2 * b, base + 4] -= 1
        W[2 * b + 1, base + 4] += 1
        
        # Term 6: (c-a)(e+f)
        # U: c=2, a=0 -> U[2, base+5] += 1, U[0, base+5] -= 1
        # V: e=2*b, f=2*b+1 -> V[2*b, base+5] += 1, V[2*b+1, base+5] += 1
        # W: c22=2*b+3 -> W[2*b+3, base+5] += 1
        U[2, base + 5] += 1
        U[0, base + 5] -= 1
        V[2 * b, base + 5] += 1
        V[2 * b + 1, base + 5] += 1
        W[2 * b + 3, base + 5] += 1
        
        # Term 7: (b-d)(g+h)
        # U: b=1, d=3 -> U[1, base+6] += 1, U[3, base+6] -= 1
        # V: g=2*b+2, h=2*b+3 -> V[2*b+2, base+6] += 1, V[2*b+3, base+6] += 1
        # W: c11=2*b -> W[2*b, base+6] += 1
        U[1, base + 6] += 1
        U[3, base + 6] -= 1
        V[2 * b + 2, base + 6] += 1
        V[2 * b + 3, base + 6] += 1
        W[2 * b, base + 6] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
