"""Rewritten program exploiting block-Strassen composition for 4x4 x 4x8 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

def construct():
    # Deadline guard: finish well before the 150s wall-clock kill.
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    # We will construct a decomposition of rank 98 (7 outer Strassen terms x 7 inner Strassen terms x 2 halves of B)
    # A is 4x4, B is 4x8, C = A*B is 4x8
    
    # Index conventions:
    # A indices: i in [0,4), j in [0,4)  -> A[i*m + j] = A[i*4 + j]
    # B indices: j in [0,4), k in [0,8)  -> B[j*p + k] = B[j*8 + k]
    # C indices: k in [0,8), i in [0,4)  -> C[k*n + i] = C[k*4 + i]
    
    # We decompose B into two halves: B_left (4x4) and B_right (4x4)
    # C_left = A * B_left (4x4), C_right = A * B_right (4x4)
    # For each half, we use the 4x4x4 Strassen decomposition of rank 49.
    
    # Inner Strassen (2x2x2) decomposition, rank 7
    # For multiplying 2x2 matrix X by 2x2 matrix Y to get Z:
    # M1 = (x11 + x22)(y11 + y22)
    # M2 = (x21 + x22) y11
    # M3 = x11 (y12 - y22)
    # M4 = x22 (y21 - y11)
    # M5 = (x11 + x12) y22
    # M6 = (x21 - x11)(y11 + y12)
    # M7 = (x12 - x22)(y21 + y22)
    # Z11 = M1 - M4 + M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M3 + M6 + M7
    
    # For the inner 2x2x2, the U vector (for X flattened as [x11, x12, x21, x22]) and V vector (for Y) are:
    # term 1: u=[1,0,0,1], v=[1,0,0,1], w: contributes +1 to Z11, +1 to Z22
    # term 2: u=[0,0,1,1], v=[1,0,0,0], w: contributes +1 to Z21
    # term 3: u=[1,0,0,0], v=[0,1,0,-1], w: contributes +1 to Z12, -1 to Z22
    # term 4: u=[0,0,0,1], v=[0,-1,1,0], w: contributes -1 to Z11, +1 to Z21
    # term 5: u=[1,1,0,0], v=[0,0,0,1], w: contributes +1 to Z11, +1 to Z12
    # term 6: u=[-1,0,1,0], v=[1,1,0,0], w: contributes +1 to Z22
    # term 7: u=[0,1,0,-1], v=[0,0,1,1], w: contributes +1 to Z11, +1 to Z22
    
    # Let's define the inner Strassen terms as (u, v, w) where u, v are 4-dim vectors, w is 4-dim (for Z flattened)
    # u index: x11=0, x12=1, x21=2, x22=3
    # v index: y11=0, y12=1, y21=2, y22=3
    # w index: z11=0, z12=1, z21=2, z22=3
    
    inner_u = [
        [1, 0, 0, 1],   # term 1
        [0, 0, 1, 1],   # term 2
        [1, 0, 0, 0],   # term 3
        [0, 0, 0, 1],   # term 4
        [1, 1, 0, 0],   # term 5
        [-1, 0, 1, 0],  # term 6
        [0, 1, 0, -1],  # term 7
    ]
    inner_v = [
        [1, 0, 0, 1],   # term 1
        [1, 0, 0, 0],   # term 2
        [0, 1, 0, -1],  # term 3
        [0, -1, 1, 0],  # term 4
        [0, 0, 0, 1],   # term 5
        [1, 1, 0, 0],   # term 6
        [0, 0, 1, 1],   # term 7
    ]
    inner_w = [
        [1, 0, 0, 1],   # term 1: Z11+=1, Z22+=1
        [0, 0, 1, 0],   # term 2: Z21+=1
        [0, 1, 0, -1],  # term 3: Z12+=1, Z22-=1
        [-1, 0, 1, 0],  # term 4: Z11-=1, Z21+=1
        [1, 1, 0, 0],   # term 5: Z11+=1, Z12+=1
        [0, 0, 0, 1],   # term 6: Z22+=1
        [1, 0, 0, 1],   # term 7: Z11+=1, Z22+=1
    ]
    
    # Now, for the outer 4x4x4 Strassen:
    # A is 4x4, partitioned into 2x2 blocks: A11, A12, A21, A22
    # B is 4x4, partitioned into 2x2 blocks: B11, B12, B21, B22
    # C is 4x4, partitioned into 2x2 blocks: C11, C12, C21, C22
    
    # Outer Strassen terms (7 terms):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # For each outer term k, we form modified 2x2 blocks of A and B, then apply inner Strassen.
    # The coefficients for A blocks in each outer term:
    outer_A_coeffs = [
        [[1, 0], [0, 1]],   # M1: A11 + A22 -> blocks (11:1, 12:0, 21:0, 22:1)
        [[0, 0], [1, 1]],   # M2: A21 + A22
        [[1, 0], [0, 0]],   # M3: A11
        [[0, 0], [0, 1]],   # M4: A22
        [[1, 1], [0, 0]],   # M5: A11 + A12
        [[-1, 0], [1, 0]],  # M6: A21 - A11
        [[0, 1], [0, -1]],  # M7: A12 - A22
    ]
    outer_B_coeffs = [
        [[1, 0], [0, 1]],   # M1: B11 + B22
        [[1, 0], [0, 0]],   # M2: B11
        [[0, 1], [0, -1]],  # M3: B12 - B22
        [[0, -1], [1, 0]],  # M4: B21 - B11
        [[0, 0], [0, 1]],   # M5: B22
        [[1, 1], [0, 0]],   # M6: B11 + B12
        [[0, 0], [1, 1]],   # M7: B21 + B22
    ]
    
    # For the full 4x4x4, we have 7*7=49 terms.
    # For the 4x4x8 problem, we have 7*7*2=98 terms (duplicating for B_left and B_right).
    
    rank = 98
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 16 x 98
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 32 x 98
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 32 x 98
    
    # Helper: get index in A for block (bi, bj) and position (pi, pj) within block
    # bi, bj in {0,1} (block row, block col), pi, pj in {0,1} (position within 2x2 block)
    def a_idx(bi, bj, pi, pj):
        return (bi*2 + pi) * m + (bj*2 + pj)
    
    def b_idx(bi, bj, pi, pj):
        return (bi*2 + pi) * p + (bj*2 + pj)
    
    def c_idx(bi, bj, pi, pj, half):
        # C is 4x8. For a given half (0 for left, 1 for right):
        # The output C[i, k] where i is row, k is col
        # Block (ci, cj) of C corresponds to rows 2*ci+pi, cols 2*cj+pj within the 4x4 half
        # For half=0 (left), the k index is 0..3, for half=1 (right), k index is 4..7
        i = bi * 2 + pi
        k = half * 4 + bj * 2 + pj
        return k * n + i
    
    # Build the decomposition
    col = 0
    for half in range(2):  # 0: left 4x4, 1: right 4x4
        for outer_k in range(7):
            # Coefficients for A and B blocks
            A_coeff = outer_A_coeffs[outer_k]  # 2x2 matrix of coefficients
            B_coeff = outer_B_coeffs[outer_k]
            
            for inner_r in range(7):
                u_inner = inner_u[inner_r]
                v_inner = inner_v[inner_r]
                w_inner = inner_w[inner_r]
                
                # For each element of the inner u vector (position in 2x2 block of A):
                # u_inner[p] corresponds to position p in the 2x2 block
                # p: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                # We need to distribute this across the 4x4 A matrix blocks
                
                for pi in range(2):
                    for pj in range(2):
                        p = pi * 2 + pj  # position in 2x2
                        coeff = u_inner[p]
                        if coeff == 0:
                            continue
                        # This contributes to all A blocks (bi, bj) with coefficient A_coeff[bi][bj]
                        for bi in range(2):
                            for bj in range(2):
                                a_c = A_coeff[bi][bj]
                                if a_c == 0:
                                    continue
                                idx_a = a_idx(bi, bj, pi, pj)
                                U[idx_a][col] += Fraction(coeff) * Fraction(a_c)
                
                # Similarly for V
                for pi in range(2):
                    for pj in range(2):
                        p = pi * 2 + pj
                        coeff = v_inner[p]
                        if coeff == 0:
                            continue
                        for bi in range(2):
                            for bj in range(2):
                                b_c = B_coeff[bi][bj]
                                if b_c == 0:
                                    continue
                                idx_b = b_idx(bi, bj, pi, pj)
                                V[idx_b][col] += Fraction(coeff) * Fraction(b_c)
                
                # For W: the inner w vector gives contributions to the 2x2 C block
                # w_inner[p] for position p in 2x2
                # We need to distribute across C blocks with the outer C coefficients
                
                # Outer C coefficients for each outer term:
                # C11 = M1 - M4 + M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M3 + M6 + M7
                outer_C_coeffs = [
                    [1, 0, 0, -1, 1, 0, 1],   # C11
                    [0, 0, 1, 0, 1, 0, 0],    # C12
                    [0, 1, 0, 1, 0, 0, 0],    # C21
                    [1, 0, -1, 0, 0, 1, 1],   # C22
                ]
                # outer_C_coeffs[ci][cj][outer_k] is the coefficient
                
                for ci in range(2):
                    for cj in range(2):
                        c_coeff = outer_C_coeffs[ci][cj][outer_k]
                        if c_coeff == 0:
                            continue
                        for pi in range(2):
                            for pj in range(2):
                                p = pi * 2 + pj
                                w_coeff = w_inner[p]
                                if w_coeff == 0:
                                    continue
                                idx_c = c_idx(ci, cj, pi, pj, half)
                                W[idx_c][col] += Fraction(c_coeff) * Fraction(w_coeff)
                
                col += 1
    
    # Convert to lists of floats/ints for the interface
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

