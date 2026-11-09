import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 matrix multiplication
    # P1 = (x00 + x11) * (y00 + y11)
    # P2 = (x10 + x11) * y00
    # P3 = x00 * (y01 - y11)
    # P4 = x11 * (y10 - y00)
    # P5 = (x00 + x01) * y11
    # P6 = (x10 - x00) * (y00 + y01)
    # P7 = (x01 - x11) * (y10 + y11)
    
    # Z reconstruction:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # Define Strassen steps as (x_mask, y_mask, z_mask)
    # x_mask: dict mapping x_index to coefficient
    # y_mask: dict mapping y_index to coefficient
    # z_mask: dict mapping z_index to coefficient
    
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Dimensions:
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We decompose the multiplication into 8 sub-problems of size 2x2x2
    # by iterating over 4 output column-pairs and 2 input row-pairs?
    # Actually, the previous logic was:
    # t in 0..3 (output block index)
    # s in 0..1 (input block index)
    # This covers 8 blocks.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    # t corresponds to output column pair (0-1, 2-3, 4-5, 6-7)
    # s corresponds to input row pair (0-1, 2-3)
    for t in range(4):
        for s in range(2):
            for step in steps:
                x_c, y_c, z_c = step
                
                # U indices for x
                # x00: A[0, 2s] -> U[2s]
                # x01: A[0, 2s+1] -> U[2s+1]
                # x10: A[1, 2s] -> U[4+2s]
                # x11: A[1, 2s+1] -> U[5+2s]
                u_map = {
                    0: 2*s,
                    1: 2*s+1,
                    2: 4+2*s,
                    3: 5+2*s
                }
                
                # V indices for y
                # y00: B[2s, 2t] -> V[16s + 2t]
                # y01: B[2s, 2t+1] -> V[16s + 2t+1]
                # y10: B[2s+1, 2t] -> V[16s + 8 + 2t]
                # y11: B[2s+1, 2t+1] -> V[16s + 8 + 2t+1]
                v_map = {
                    0: 16*s + 2*t,
                    1: 16*s + 2*t+1,
                    2: 16*s + 8 + 2*t,
                    3: 16*s + 8 + 2*t+1
                }
                
                # W indices for z
                # z00: C[0, 2t] -> W[4t]
                # z01: C[0, 2t+1] -> W[4t+2]
                # z10: C[1, 2t] -> W[4t+1]
                # z11: C[1, 2t+1] -> W[4t+3]
                w_map = {
                    0: 4*t,
                    1: 4*t+2,
                    2: 4*t+1,
                    3: 4*t+3
                }
                
                for x_i, val in x_c.items():
                    U[u_map[x_i]][col_idx] += val
                
                for y_i, val in y_c.items():
                    V[v_map[y_i]][col_idx] += val
                
                for z_i, val in z_c.items():
                    W[w_map[z_i]][col_idx] += val
                
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
