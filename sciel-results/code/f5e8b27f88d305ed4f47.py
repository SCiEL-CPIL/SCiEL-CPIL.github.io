"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Rank 20 decomposition:
    # Split A (2x4) into A1 (2x2), A2 (2x1), A3 (2x1)
    # Split B (4x5) into B1 (2x5), B2 (1x5), B3 (1x5)
    # C = A1*B1 + A2*B2 + A3*B3
    
    # Rank(A1*B1) = Rank(2x2x5) = 18
    #   Split B1 (2x5) into B1a (2x2), B1b (2x2), B1c (2x1)
    #   Rank(2x2x2) = 7 (Strassen)
    #   Rank(2x2x1) = 4 (Naive)
    #   Total = 7 + 7 + 4 = 18
    # Rank(A2*B2) = Rank(2x1x5) = 1 (Outer product)
    # Rank(A3*B3) = Rank(2x1x5) = 1 (Outer product)
    # Total Rank = 18 + 1 + 1 = 20
    
    rank = 20
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    deadline = time.time() + 140.0

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def strassen_2x2x2(y00, y01, y10, y11, z00, z01, z10, z11, w00, w01, w10, w11):
        nonlocal col
        if col >= rank or time.time() > deadline:
            return
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        if col >= rank or time.time() > deadline:
            return
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1
        
        if col >= rank or time.time() > deadline:
            return
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1
        V[v_idx(*z11)][col] -= 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        if col >= rank or time.time() > deadline:
            return
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z00)][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        if col >= rank or time.time() > deadline:
            return
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1
        
        if col >= rank or time.time() > deadline:
            return
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z01)][col] += 1
        W[w11][col] += 1
        col += 1
        
        if col >= rank or time.time() > deadline:
            return
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] += 1
        U[u_idx(*y11)][col] -= 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w00][col] += 1
        col += 1

    def naive_2x2x1(y00, y01, y10, y11, z0, z1, w0, w1):
        nonlocal col
        if col >= rank or time.time() > deadline:
            return
        # M1 = y00*z0
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z0)][col] += 1
        W[w0][col] += 1
        col += 1
        
        if col >= rank or time.time() > deadline:
            return
        # M2 = y01*z1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z1)][col] += 1
        W[w0][col] += 1
        col += 1
        
        if col >= rank or time.time() > deadline:
            return
        # M3 = y10*z0
        U[u_idx(*y10)][col] += 1
        V[v_idx(*z0)][col] += 1
        W[w1][col] += 1
        col += 1
        
        if col >= rank or time.time() > deadline:
            return
        # M4 = y11*z1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z1)][col] += 1
        W[w1][col] += 1
        col += 1

    def outer_2x1x5(y0, y1, z0, z1, z2, z3, z4):
        nonlocal col
        if col >= rank or time.time() > deadline:
            return
        # Rank 1: u=[1,1], v=[1,1,1,1,1], w=ones(2,5)
        U[u_idx(0, y0)][col] += 1
        U[u_idx(1, y0)][col] += 1
        V[v_idx(z0, 0)][col] += 1
        V[v_idx(z1, 0)][col] += 1
        V[v_idx(z2, 0)][col] += 1
        V[v_idx(z3, 0)][col] += 1
        V[v_idx(z4, 0)][col] += 1
        
        # W is ones for all k in 0..4, i in 0..1
        for k in range(5):
            for i in range(2):
                W[w_idx(k, i)][col] += 1
        col += 1

    # Block 1: A[:, 0:2] * B[0:2, :]
    # A indices: (0,0),(0,1),(1,0),(1,1)
    
    # 1a: B[0:2, 0:2]
    strassen_2x2x2(
        (0,0), (0,1), (1,0), (1,1),
        (0,0), (0,1), (1,0), (1,1),
        w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)
    )
    
    # 1b: B[0:2, 2:4]
    strassen_2x2x2(
        (0,0), (0,1), (1,0), (1,1),
        (0,2), (0,3), (1,2), (1,3),
        w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)
    )
    
    # 1c: B[0:2, 4:5]
    naive_2x2x1(
        (0,0), (0,1), (1,0), (1,1),
        (0,4), (1,4),
        w_idx(4,0), w_idx(4,1)
    )
    
    # Block 2: A[:, 2] * B[2, :]
    # A index: 2
    # B row: 2
    outer_2x1x5(2, 2, 2, 2, 2, 2, 2)
    
    # Block 3: A[:, 3] * B[3, :]
    # A index: 3
    # B row: 3
    outer_2x1x5(3, 3, 3, 3, 3, 3, 3)
    
    # Trim to actual rank used
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
