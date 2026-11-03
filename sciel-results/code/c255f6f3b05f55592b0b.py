"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _build_rank_44(n, m, p):
    # n=3, m=4, p=4
    # Decomposition:
    # 1. A_1 (1x4) * B (4x4) -> C_1 (1x4). Rank 16.
    # 2. A_2 (2x4) * B (4x4) -> C_2 (2x4). Rank 28.
    # Total Rank 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (row 0) * B -> C_1 (row 0)
    # A_1 has indices i=0, j=0..3.
    # B has indices j=0..3, k=0..3.
    # C_1 has indices k=0..3, i=0.
    # Term: sum_{j} A_{0,j} B_{j,k} C_{0,k}
    # This is 4 dot products of length 4.
    # For each k (0..3), we need sum_{j=0..3} A_{0,j} B_{j,k}.
    # This requires 4 multiplications per k.
    # Total 16 multiplications.
    for k in range(4):
        for j in range(4):
            # U index: i=0, j=j -> 0*4 + j = j
            U[j][col] = 1
            # V index: j=j, k=k -> j*4 + k
            V[j * 4 + k][col] = 1
            # W index: k=k, i=0 -> k*3 + 0 = 3*k
            W[3 * k][col] = 1
            col += 1
            
    # Part 2: A_2 (rows 1,2) * B -> C_2 (rows 1,2)
    # A_2 is 2x4. Indices i=1,2; j=0..3.
    # B is 4x4. Indices j=0..3; k=0..3.
    # C_2 is 2x4. Indices k=0..3; i=1,2.
    # We split A_2 into A_21 (cols 0,1) and A_22 (cols 2,3).
    # And B into B_1 (rows 0,1) and B_2 (rows 2,3).
    # C_2 = A_21 B_1 + A_22 B_2.
    # A_21 is 2x2. B_1 is 2x4. Product is 2x4. Rank mu(2,2,4) = 14.
    # A_22 is 2x2. B_2 is 2x4. Product is 2x4. Rank mu(2,2,4) = 14.
    
    # Helper to add mu(2,2,4) block
    # A_block is 2x2 at (i_offset, j_offset)
    # B_block is 2x4 at (j_offset, k_offset)
    # C_block is 2x4 at (k_offset, i_offset)
    
    def add_mu_2_2_4(i_off, j_off, k_off):
        nonlocal col
        # A is 2x2. B is 2x4.
        # We split B into two 2x2 blocks (cols 0,1 and 2,3).
        # B_left (2x2), B_right (2x2).
        # C_left (2x2), C_right (2x2).
        # C_left = A * B_left. Rank 7 (Strassen).
        # C_right = A * B_right. Rank 7 (Strassen).
        
        # Strassen for 2x2 * 2x2
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b00)
        # m4 = a11(b11-b01)
        # m5 = (a00+a01)b01
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # Indices:
        # A: i in {i_off, i_off+1}, j in {j_off, j_off+1}
        # B: j in {j_off, j_off+1}, k in {k_off, k_off+1, k_off+2, k_off+3}
        # C: k in {k_off, k_off+1, k_off+2, k_off+3}, i in {i_off, i_off+1}
        
        # Left block (k_off, k_off+1)
        # A indices: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B Left indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C Left indices: (k_off, i_off), (k_off, i_off+1), (k_off+1, i_off), (k_off+1, i_off+1)
        
        def add_strassen(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_k0, c_k1, c_i0, c_i1):
            nonlocal col
            # m1
            U[a_i0 * 4 + a_j0][col] += 1
            U[a_i1 * 4 + a_j1][col] += 1
            V[b_j0 * 4 + b_k0][col] += 1
            V[b_j1 * 4 + b_k1][col] += 1
            W[c_k0 * 3 + c_i0][col] += 1
            W[c_k1 * 3 + c_i1][col] += 1
            col += 1
            # m2
            U[a_i1 * 4 + a_j0][col] += 1
            U[a_i1 * 4 + a_j1][col] += 1
            V[b_j0 * 4 + b_k0][col] += 1
            W[c_k0 * 3 + c_i1][col] += 1
            col += 1
            # m3
            U[a_i0 * 4 + a_j0][col] += 1
            V[b_j1 * 4 + b_k0][col] += 1
            V[b_j0 * 4 + b_k0][col] -= 1
            W[c_k0 * 3 + c_i0][col] += 1
            col += 1
            # m4
            U[a_i1 * 4 + a_j1][col] += 1
            V[b_j1 * 4 + b_k1][col] += 1
            V[b_j0 * 4 + b_k1][col] -= 1
            W[c_k1 * 3 + c_i1][col] += 1
            col += 1
            # m5
            U[a_i0 * 4 + a_j0][col] += 1
            U[a_i0 * 4 + a_j1][col] += 1
            V[b_j0 * 4 + b_k1][col] += 1
            W[c_k1 * 3 + c_i0][col] += 1
            col += 1
            # m6
            U[a_i1 * 4 + a_j0][col] += 1
            U[a_i0 * 4 + a_j0][col] -= 1
            V[b_j0 * 4 + b_k0][col] += 1
            V[b_j0 * 4 + b_k1][col] += 1
            W[c_k0 * 3 + c_i1][col] += 1
            col += 1
            # m7
            U[a_i0 * 4 + a_j1][col] += 1
            U[a_i1 * 4 + a_j1][col] -= 1
            V[b_j1 * 4 + b_k0][col] += 1
            V[b_j1 * 4 + b_k1][col] += 1
            W[c_k1 * 3 + c_i0][col] += 1
            col += 1

        # Left 2x2 block of B (cols 0,1)
        add_strassen(
            i_off, i_off+1, j_off, j_off+1,
            j_off, j_off+1, k_off, k_off+1,
            k_off, k_off+1, i_off, i_off+1
        )
        # Right 2x2 block of B (cols 2,3)
        add_strassen(
            i_off, i_off+1, j_off, j_off+1,
            j_off, j_off+1, k_off+2, k_off+3,
            k_off+2, k_off+3, i_off, i_off+1
        )

    # Call for A_21 (i=1,2; j=0,1) and B_1 (j=0,1; k=0..3)
    # Note: B_1 is rows 0,1 of B.
    add_mu_2_2_4(1, 0, 0)
    
    # Call for A_22 (i=1,2; j=2,3) and B_2 (j=2,3; k=0..3)
    # Note: B_2 is rows 2,3 of B.
    add_mu_2_2_4(1, 2, 0)
    
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Use the rank 44 decomposition
    result = _build_rank_44(n, m, p)
    
    # Verify if time permits
    if time.time() < deadline - 10:
        U = result['U']
        V = result['V']
        W = result['W']
        rank = len(U[0])
        
        # Quick verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - expected) > 1e-9:
                        return _schoolbook(n, m, p)
                        
    return result
# EVOLVE-BLOCK-END
