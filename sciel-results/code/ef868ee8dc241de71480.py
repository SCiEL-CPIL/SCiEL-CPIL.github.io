import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Laderman's 23-term decomposition for 3x3 * 3x3
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # U: 9x23 (A indices), V: 9x23 (B indices), W: 9x23 (C indices)
    
    # Define the 23 terms from Laderman's algorithm
    # Each term is (u, v, w) where u, v, w are 9-element vectors
    laderman_23 = [
        # Term 1: A00*B00 -> C00
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 2: (A01+A11)*(B10+B20) -> C00
        ([0,1,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0]),
        # Term 3: (A02+A12)*(B20+B10) -> C01
        ([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,1,0, 0,0,0, 0,0,0]),
        # Term 4: (A10+A20)*(B01+B11) -> C10
        ([0,0,0, 1,0,0, 1,0,0], [1,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 5: (A11+A21)*(B11+B21) -> C11
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0]),
        # Term 6: (A12+A22)*(B21+B11) -> C12
        ([0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),
        # Term 7: (A20+A10)*(B02+B22) -> C20
        ([0,0,0, 1,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        # Term 8: (A21+A11)*(B12+B22) -> C21
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        # Term 9: (A22+A12)*(B22+B12) -> C22
        ([0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Term 10: A00*B11 -> C01
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 11: A00*B22 -> C02
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        # Term 12: A11*B00 -> C10
        ([0,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 13: A11*B22 -> C12
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),
        # Term 14: A22*B00 -> C20
        ([0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 15: A22*B11 -> C21
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 16: (A01-A11)*(B10-B20) -> C00
        ([0,1,0, -1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,0, 0,0,0]),
        # Term 17: (A02-A12)*(B20-B10) -> C01
        ([0,0,1, 0,0,-1, 0,0,0], [0,0,0, 0,1,0, 0,0,-1], [0,1,0, 0,0,0, 0,0,0]),
        # Term 18: (A10-A20)*(B01-B11) -> C10
        ([0,0,0, 1,0,0, -1,0,0], [1,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 19: (A11-A21)*(B11-B21) -> C11
        ([0,0,0, 0,1,0, 0,-1,0], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,1,0, 0,0,0]),
        # Term 20: (A12-A22)*(B21-B11) -> C12
        ([0,0,0, 0,0,1, 0,0,-1], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,1, 0,0,0]),
        # Term 21: (A20-A10)*(B02-B22) -> C20
        ([0,0,0, 1,0,0, -1,0,0], [1,0,0, 0,0,0, 0,0,-1], [0,0,0, 0,0,0, 1,0,0]),
        # Term 22: (A21-A11)*(B12-B22) -> C21
        ([0,0,0, 0,1,0, 0,-1,0], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,0, 0,1,0]),
        # Term 23: (A22-A12)*(B22-B12) -> C22
        ([0,0,0, 0,0,1, 0,0,-1], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,0, 0,0,1]),
    ]
    
    # Build 3x3x3 matrices
    U3 = [[0]*23 for _ in range(9)]
    V3 = [[0]*23 for _ in range(9)]
    W3 = [[0]*23 for _ in range(9)]
    
    for c in range(23):
        u, v, w = laderman_23[c]
        for idx in range(9):
            U3[idx][c] = u[idx]
            V3[idx][c] = v[idx]
            W3[idx][c] = w[idx]
    
    # Verify 3x3x3 decomposition
    def verify_3x3(U, V, W, rank):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for c in range(rank):
                        s += U[i*3+j][c] * V[j*3+k][c] * W[k*3+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify_3x3(U3, V3, W3, 23):
        # Fallback to schoolbook if Laderman doesn't verify
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Now extend to 3x3 * 3x6
    # The tensor is T[i,j,k] where i in [0,3), j in [0,3), k in [0,6)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can split B into two 3x3 blocks: B_left (k=0,1,2) and B_right (k=3,4,5)
    # C[i,0..2] = sum_j A[i,j] B[j,0..2]
    # C[i,3..5] = sum_j A[i,j] B[j,3..5]
    
    # Use two copies of the 23-term decomposition
    # Total rank = 46
    
    rank = 46
    U46 = [[0]*rank for _ in range(9)]  # n*m = 9
    V46 = [[0]*rank for _ in range(18)]  # m*p = 3*6 = 18
    W46 = [[0]*rank for _ in range(18)]  # p*n = 6*3 = 18
    
    # First 23 terms: handle B[j,0..2] -> C[i,0..2]
    for c in range(23):
        for idx in range(9):
            U46[idx][c] = U3[idx][c]
            # V index: j*p + k = j*6 + k, where k in [0,3)
            v_idx = idx % 3 * 6 + (idx // 3)  # This is wrong, let me fix
            # Actually V[j,k] where j is row of B, k is column of B
            # In V3, index is j*3+k for 3x3 B
            # In V46, index is j*6+k for 3x6 B
            # So for k in [0,3): V46[j*6+k][c] = V3[j*3+k][c]
        
        # Let me redo this properly
        for j in range(3):
            for k in range(3):
                # V3 index: j*3+k
                # V46 index: j*6+k
                V46[j*6+k][c] = V3[j*3+k][c]
                # W3 index: k*3+i
                # W46 index: k*3+i (since k in [0,3), same)
                for i in range(3):
                    W46[k*3+i][c] = W3[k*3+i][c]
    
    # Second 23 terms (c=23..45): handle B[j,3..5] -> C[i,3..5]
    for c in range(23):
        c2 = c + 23
        for idx in range(9):
            U46[idx][c2] = U3[idx][c]
        
        for j in range(3):
            for k in range(3):
                # V3 index: j*3+k
                # V46 index: j*6+(k+3)
                V46[j*6+(k+3)][c2] = V3[j*3+k][c]
                # W3 index: k*3+i
                # W46 index: (k+3)*3+i
                for i in range(3):
                    W46[(k+3)*3+i][c2] = W3[k*3+i][c]
    
    # Verify the 3x3x6 decomposition
    def verify_3x3x6(U, V, W, rank):
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    s = 0
                    for c in range(rank):
                        s += U[i*3+j][c] * V[j*6+k][c] * W[k*3+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify_3x3x6(U46, V46, W46, 46):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U46, "V": V46, "W": W46}
# EVOLVE-BLOCK-END
