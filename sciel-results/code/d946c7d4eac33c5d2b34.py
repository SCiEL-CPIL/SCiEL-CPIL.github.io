"""
Rank-minimal tensor decomposition for 2x2 * 2x15 matrix multiplication.
Uses Strassen's algorithm (rank 7) for the 2x2x2 core, extended to 15 columns.
"""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [2, 2, 15]


def construct():
    n, m, p = DIMENSIONS  # 2, 2, 15

    # Strassen's 7-term decomposition for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    #
    # p1 = (a+d)(e+h)
    # p2 = (c+d)f
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)g
    # p6 = (c-d)(e+h)
    # p7 = (b-d)(f+g)
    #
    # C00 = p1 + p4 - p5 - p7
    # C01 = p3 + p5
    # C10 = p2 + p6
    # C11 = p1 - p2 + p3 + p7
    #
    # For the full 2x2 * 2x15 problem:
    # A is 2x2: [[a, b], [c, d]] (indices 0,1,2,3)
    # B is 2x15: B[j,k] for j in {0,1}, k in {0..14}
    # C is 2x15: C[i,k] for i in {0,1}, k in {0..14}
    #
    # We decompose B into two 2x2 blocks and a 2x11 remainder... 
    # Actually, better: decompose into pairs.
    #
    # For each pair of columns (2t, 2t+1) of B, we can apply Strassen.
    # 15 = 7*2 + 1, so we have 7 pairs and 1 single column.
    #
    # For a single column, we need rank 4 (schoolbook for 2x2*2x1).
    # For a pair of columns, Strassen gives rank 7.
    #
    # Total rank: 7*7 + 4 = 49 + 4 = 53.
    #
    # Wait, but we can also group the single column with one pair to make a 3-column block.
    # For 2x2*2x3, the rank is at least max(4, 6, 6) = 6.
    # Actually, let's just do 7 pairs + 1 single.
    #
    # Let me define the structure more carefully.
    #
    # A has entries indexed as: A[0]=a, A[1]=b, A[2]=c, A[3]=d (row-major, 2x2)
    # B has entries indexed as: B[j*15+k] for j in {0,1}, k in {0..14}
    # C has entries indexed as: C[k*2+i] for k in {0..14}, i in {0,1} (per W[k*n+i, r])
    #
    # For Strassen on a 2-column block (columns 2t and 2t+1):
    # B block = [[B[0,2t], B[0,2t+1]], [B[1,2t], B[1,2t+1]]]
    # Let e=B[0,2t], f=B[0,2t+1], g=B[1,2t], h=B[1,2t+1]
    #
    # The 7 Strassen terms:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)f
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)g
    # p6 = (c-d)(e+h)
    # p7 = (b-d)(f+g)
    #
    # Results:
    # C[0,2t] = p1 + p4 - p5 - p7
    # C[0,2t+1] = p3 + p5
    # C[1,2t] = p2 + p6
    # C[1,2t+1] = p1 - p2 + p3 + p7
    #
    # For a single column (column 14):
    # B block = [[B[0,14]], [B[1,14]]]
    # Let e=B[0,14], g=B[1,14]
    # C[0,14] = a*e + b*g
    # C[1,14] = c*e + d*g
    # This needs 4 terms: (a,e), (b,g), (c,e), (d,g)
    #
    # Total rank = 7*7 + 4 = 53.

    rank_total = 53
    U = np.zeros((n * m, rank_total), dtype=object)  # 4 x 53
    V = np.zeros((m * p, rank_total), dtype=object)  # 30 x 53
    W = np.zeros((p * n, rank_total), dtype=object)  # 30 x 53

    # A indices: 0=a, 1=b, 2=c, 3=d
    # U[i*m+j, r] where i in {0,1}, j in {0,1}
    # U[0, r] = coefficient of a
    # U[1, r] = coefficient of b
    # U[2, r] = coefficient of c
    # U[3, r] = coefficient of d

    # B indices: j*p+k, j in {0,1}, k in {0..14}
    # V[j*15+k, r] = coefficient of B[j,k]

    # C indices: k*n+i, k in {0..14}, i in {0,1}
    # W[k*2+i, r] = coefficient for C[i,k]

    col = 0
    # 7 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)
    for t in range(7):
        k0 = 2 * t
        k1 = 2 * t + 1
        # Strassen terms for this pair
        # p1 = (a+d)(e+h): U=[1,0,0,1], V: e at (0,k0), h at (1,k1)
        U[0, col] = 1
        U[3, col] = 1
        V[0 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        # W: C[0,k0] += p1, C[1,k1] += p1
        W[k0 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # p2 = (c+d)f: U=[0,0,1,1], V: f at (0,k1)
        U[2, col] = 1
        U[3, col] = 1
        V[0 * p + k1, col] = 1
        W[k0 * n + 1, col] = 1  # C[1,k0] += p2
        col += 1

        # p3 = a(f-h): U=[1,0,0,0], V: f at (0,k1), -h at (1,k1)
        U[0, col] = 1
        V[0 * p + k1, col] = 1
        V[1 * p + k1, col] = -1
        W[k1 * n + 0, col] = 1  # C[0,k1] += p3
        W[k1 * n + 1, col] = 1  # C[1,k1] += p3
        col += 1

        # p4 = d(g-e): U=[0,0,0,1], V: -e at (0,k0), g at (1,k0)
        U[3, col] = 1
        V[0 * p + k0, col] = -1
        V[1 * p + k0, col] = 1
        W[k0 * n + 0, col] = 1  # C[0,k0] += p4
        col += 1

        # p5 = (a+b)g: U=[1,1,0,0], V: g at (1,k0)
        U[0, col] = 1
        U[1, col] = 1
        V[1 * p + k0, col] = 1
        W[k0 * n + 0, col] = -1  # C[0,k0] -= p5
        W[k1 * n + 0, col] = 1   # C[0,k1] += p5
        col += 1

        # p6 = (c-d)(e+h): U=[0,0,1,-1], V: e at (0,k0), h at (1,k1)
        U[2, col] = 1
        U[3, col] = -1
        V[0 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 1, col] = 1  # C[1,k0] += p6
        col += 1

        # p7 = (b-d)(f+g): U=[0,1,0,-1], V: f at (0,k1), g at (1,k0)
        U[1, col] = 1
        U[3, col] = -1
        V[0 * p + k1, col] = 1
        V[1 * p + k0, col] = 1
        W[k0 * n + 0, col] = -1  # C[0,k0] -= p7
        W[k1 * n + 1, col] = 1   # C[1,k1] += p7
        col += 1

    # Single column 14: schoolbook 4 terms
    k = 14
    # C[0,14] = a*B[0,14] + b*B[1,14]
    # C[1,14] = c*B[0,14] + d*B[1,14]
    # Term: a * B[0,14]
    U[0, col] = 1
    V[0 * p + k, col] = 1
    W[k * n + 0, col] = 1
    col += 1

    # Term: b * B[1,14]
    U[1, col] = 1
    V[1 * p + k, col] = 1
    W[k * n + 0, col] = 1
    col += 1

    # Term: c * B[0,14]
    U[2, col] = 1
    V[0 * p + k, col] = 1
    W[k * n + 1, col] = 1
    col += 1

    # Term: d * B[1,14]
    U[3, col] = 1
    V[1 * p + k, col] = 1
    W[k * n + 1, col] = 1
    col += 1

    assert col == rank_total, f"Expected {rank_total} columns, got {col}"

    # Convert to list of ints
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
