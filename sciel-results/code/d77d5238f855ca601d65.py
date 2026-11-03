"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Precompute target tensor non-zeros
    # T[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i match indices
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def check_decomposition(U, V, W):
        r = len(U[0])
        # Verify all non-zero entries of target
        for (a, b, c), val in target.items():
            s = 0
            for r_idx in range(r):
                s += U[a][r_idx] * V[b][r_idx] * W[c][r_idx]
            if s != val:
                return False
        # Verify all zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target:
                        s = 0
                        for r_idx in range(r):
                            s += U[a][r_idx] * V[b][r_idx] * W[c][r_idx]
                        if s != 0:
                            return False
        return True

    # Strassen-based decomposition for 3x4x7
    # Use Strassen rank-7 for 3x3x3 core, schoolbook for remainder
    def strassen_347():
        rank = 64
        U = [[0] * rank for _ in range(n * m)]   # 12 x 64
        V = [[0] * rank for _ in range(m * p)]   # 28 x 64
        W = [[0] * rank for _ in range(p * n)]   # 21 x 64

        col = 0

        # Strassen's rank-7 decomposition for 3x3x3
        # M1 = (A00+A22)*(B00+B22)
        # M2 = (A11+A22)*(B11+B22)
        # M3 = (A00+A11)*(B00+B11)
        # M4 = (A22+A01)*(B22+B11)
        # M5 = (A11+A01)*(B11+B02)
        # M6 = (A22+A12)*(B22+B01)
        # M7 = (A00+A12)*(B00+B12)
        
        # C[0,0]=M1-M2+M3
        # C[0,1]=M2-M3+M5
        # C[0,2]=M4-M6+M7
        # C[1,0]=M1-M4+M6
        # C[1,1]=M1-M2+M5
        # C[1,2]=M3-M5+M7
        # C[2,0]=M2-M3+M4
        # C[2,1]=M1-M4+M5
        # C[2,2]=M1-M2+M6

        # U factors for M1..M7 (for i,j in 0..2, index i*3+j):
        strassen_U = [
            [1,0,1,0,0,0,1],  # (0,0)
            [0,0,0,1,1,0,0],  # (0,1)
            [0,0,0,0,0,0,0],  # (0,2)
            [0,0,0,0,0,0,0],  # (1,0)
            [0,1,1,0,1,0,0],  # (1,1)
            [0,0,0,0,0,1,1],  # (1,2)
            [0,0,0,0,0,0,0],  # (2,0)
            [0,0,0,0,0,0,0],  # (2,1)
            [1,1,0,1,0,1,0],  # (2,2)
        ]

        # V factors:
        strassen_V = [
            [1,0,1,0,0,0,1],  # (0,0)
            [0,0,0,0,0,1,0],  # (0,1)
            [0,0,0,0,1,0,0],  # (0,2)
            [0,0,0,1,0,0,0],  # (1,0)
            [0,1,1,1,1,0,0],  # (1,1)
            [0,0,0,0,0,0,1],  # (1,2)
            [1,0,0,0,0,0,0],  # (2,0)
            [0,0,0,0,0,0,0],  # (2,1)
            [1,1,0,1,0,1,0],  # (2,2)
        ]

        # W factors (for k,i in 0..2, W[k*3+i]):
        strassen_W = [
            [1,-1,1,0,0,0,0],  # (0,0)
            [0,1,-1,0,1,0,0],  # (0,1)
            [0,0,0,1,0,-1,1],  # (0,2)
            [1,0,0,-1,0,1,0],  # (1,0)
            [1,-1,0,0,1,0,0],  # (1,1)
            [0,0,1,0,-1,0,1],  # (1,2)
            [0,1,-1,1,0,0,0],  # (2,0)
            [1,0,0,-1,1,0,0],  # (2,1)
            [1,-1,0,0,0,1,0],  # (2,2)
        ]

        # Place Strassen factors into U, V, W for i,j,k in 0..2
        for i in range(3):
            for j in range(3):
                for r_idx in range(7):
                    U[i * m + j][col + r_idx] = strassen_U[i * 3 + j][r_idx]
                    V[j * p + k][col + r_idx] = strassen_V[j * 3 + k][r_idx] if False else 0
                    # Actually need to be careful with V indexing
        # Let me redo this more carefully
        # Reset
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Strassen part: i in 0..2, j in 0..2, k in 0..2
        for i in range(3):
            for j in range(3):
                for r_idx in range(7):
                    U[i * m + j][col + r_idx] = strassen_U[i * 3 + j][r_idx]
        for j in range(3):
            for k in range(3):
                for r_idx in range(7):
                    V[j * p + k][col + r_idx] = strassen_V[j * 3 + k][r_idx]
        for k in range(3):
            for i in range(3):
                for r_idx in range(7):
                    W[k * n + i][col + r_idx] = strassen_W[k * 3 + i][r_idx]
        col += 7

        # Schoolbook remainder
        # Terms where j=3: i in 0..2, k in 0..6 -> 3*7 = 21 terms
        for i in range(3):
            for k in range(7):
                U[i * m + 3][col] = 1
                V[3 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        # Terms where j in 0..2, k in 3..6: i in 0..2 -> 3*4*3 = 36 terms
        for i in range(3):
            for j in range(3):
                for k in range(3, 7):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

        assert col == rank, f"col={col}, rank={rank}"
        return {"U": U, "V": V, "W": W}

    # Try the Strassen-based decomposition first
    result = strassen_347()
    if check_decomposition(result["U"], result["V"], result["W"]):
        return result

    # Fallback: try random search for lower rank
    best_result = None
    for rank in range(40, 65):
        if time.time() > deadline:
            break
            
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if check_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}

    # Final fallback to schoolbook
    return schoolbook(n, m, p)

# EVOLVE-BLOCK-END
