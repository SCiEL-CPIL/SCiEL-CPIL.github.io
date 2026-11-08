import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # Lx[t][i, j] is coeff of X[i, j] in Lx[t]
    # Ly[t][i, j] is coeff of Y[i, j] in Ly[t]
    # W2x2[p, q, t] is coeff of M[t] in Z[p, q]
    
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    
    Lx = [
        [[1, 0], [0, 1]],   # 0: x00 + x11
        [[0, 0], [1, 1]],   # 1: x10 + x11
        [[1, 0], [0, 0]],   # 2: x00
        [[0, 0], [0, 1]],   # 3: x11
        [[1, 1], [0, 0]],   # 4: x00 + x01
        [[-1, 0], [1, 0]],  # 5: x10 - x00
        [[0, 1], [0, -1]]   # 6: x01 - x11
    ]
    
    Ly = [
        [[1, 0], [0, 1]],   # 0: y00 + y11
        [[1, 0], [0, 0]],   # 1: y00
        [[0, 1], [0, -1]],  # 2: y01 - y11
        [[0, -1], [1, 0]],  # 3: y10 - y00
        [[0, 0], [0, 1]],   # 4: y11
        [[1, 1], [0, 0]],   # 5: y00 + y01
        [[0, 0], [1, 1]]    # 6: y10 + y11
    ]
    
    # Z = [[z00, z01], [z10, z11]]
    # z00 = M0 + M3 - M4 + M6
    # z01 = M2 + M4
    # z10 = M1 + M3
    # z11 = M0 - M1 + M2 + M5
    
    W2x2 = [
        [ [1, 0, 0, 1, -1, 0, 1],   # z00
          [0, 0, 1, 0, 1, 0, 0] ],  # z01
        [ [0, 1, 0, 1, 0, 0, 0],    # z10
          [1, -1, 1, 0, 0, 1, 0] ]  # z11
    ]
    
    # Outer Strassen steps s=0..6
    # Xs is 2x2 matrix of blocks of A
    # Ys is 2x2 matrix of blocks of B
    # Block indices: 00, 01, 10, 11
    
    # s=0: X = [[A00, 0], [0, A11]], Y = [[B00, 0], [0, B11]]
    # s=1: X = [[0, 0], [A10, A11]], Y = [[B00, 0], [0, 0]]
    # s=2: X = [[A00, 0], [0, 0]], Y = [[0, B01], [0, -B11]]
    # s=3: X = [[0, 0], [0, A11]], Y = [[-B00, 0], [B10, 0]]
    # s=4: X = [[A00, A01], [0, 0]], Y = [[0, 0], [0, B11]]
    # s=5: X = [[-A00, 0], [A10, 0]], Y = [[B00, B01], [0, 0]]
    # s=6: X = [[0, A01], [0, -A11]], Y = [[0, 0], [B10, B11]]
    
    # Represent Xs and Ys as 2x2 arrays of block coefficients
    # Block coeff is a 2x2 matrix of coefficients for the 2x2 block
    # e.g., A00 is [[1,0],[0,1]] for the block itself, 0 for others.
    
    def block_id_to_idx(bi, bj):
        return bi * 2 + bj

    # Xs[s][p][q] is a 2x2 matrix of coefficients for A_block[p][q]
    # Actually, Xs[s] is a 2x2 matrix of blocks.
    # Let's define Xs[s][p][q] as a 2x2 matrix C such that Xs[s][p][q] = sum C[i,j] A[i,j]
    
    Xs = []
    Ys = []
    
    for s in range(7):
        Xs_s = [[None for _ in range(2)] for _ in range(2)]
        Ys_s = [[None for _ in range(2)] for _ in range(2)]
        
        # Initialize to zero
        for p in range(2):
            for q in range(2):
                Xs_s[p][q] = [[0,0],[0,0]]
                Ys_s[p][q] = [[0,0],[0,0]]
                
        if s == 0:
            Xs_s[0][0] = [[1,0],[0,1]]
            Xs_s[1][1] = [[1,0],[0,1]]
            Ys_s[0][0] = [[1,0],[0,1]]
            Ys_s[1][1] = [[1,0],[0,1]]
        elif s == 1:
            Xs_s[1][0] = [[1,0],[0,1]]
            Xs_s[1][1] = [[1,0],[0,1]]
            Ys_s[0][0] = [[1,0],[0,1]]
        elif s == 2:
            Xs_s[0][0] = [[1,0],[0,1]]
            Ys_s[0][1] = [[1,0],[0,1]]
            Ys_s[1][1] = [[-1,0],[0,-1]]
        elif s == 3:
            Xs_s[1][1] = [[1,0],[0,1]]
            Ys_s[0][0] = [[-1,0],[0,-1]]
            Ys_s[1][0] = [[1,0],[0,1]]
        elif s == 4:
            Xs_s[0][0] = [[1,0],[0,1]]
            Xs_s[0][1] = [[1,0],[0,1]]
            Ys_s[1][1] = [[1,0],[0,1]]
        elif s == 5:
            Xs_s[0][0] = [[-1,0],[0,-1]]
            Xs_s[1][0] = [[1,0],[0,1]]
            Ys_s[0][0] = [[1,0],[0,1]]
            Ys_s[0][1] = [[1,0],[0,1]]
        elif s == 6:
            Xs_s[0][1] = [[1,0],[0,1]]
            Xs_s[1][1] = [[-1,0],[0,-1]]
            Ys_s[1][0] = [[1,0],[0,1]]
            Ys_s[1][1] = [[1,0],[0,1]]
            
        Xs.append(Xs_s)
        Ys.append(Ys_s)
        
    # Coefficients c_s_pq for C_block[p][q] = sum_s c_s_pq M_s
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    C_coeffs = [
        [ [1, 0, 0, 1, -1, 0, 1],   # C00
          [0, 0, 1, 0, 1, 0, 0] ],  # C01
        [ [0, 1, 0, 1, 0, 0, 0],    # C10
          [1, -1, 1, 0, 0, 1, 0] ]  # C11
    ]
    
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            
            # Calculate U
            # U[i, j, r] is coeff of A[i, j] in Lx[t](Xs[s])
            # Lx[t] is a linear form on Xs[s]'s entries (which are 2x2 blocks)
            # Xs[s] is a 2x2 matrix of blocks.
            # Lx[t][p][q] is the coeff of the block Xs[s][p][q] in the linear form.
            # Xs[s][p][q] is a 2x2 matrix of A.
            # So U[i, j, r] = sum_{p,q} Lx[t][p][q] * Xs[s][p][q][i%2, j%2]
            
            for i in range(4):
                for j in range(4):
                    val = 0
                    for p in range(2):
                        for q in range(2):
                            coeff_block = Lx[t][p][q]
                            if coeff_block != 0:
                                # Xs[s][p][q] is a 2x2 matrix
                                # i%2, j%2 are the local indices
                                val += coeff_block * Xs[s][p][q][i % 2][j % 2]
                    U[i * 4 + j][r] = val
                    
            # Calculate V
            # V[j, k, r] is coeff of B[j, k] in Ly[t](Ys[s])
            for j in range(4):
                for k in range(4):
                    val = 0
                    for p in range(2):
                        for q in range(2):
                            coeff_block = Ly[t][p][q]
                            if coeff_block != 0:
                                val += coeff_block * Ys[s][p][q][j % 2][k % 2]
                    V[j * 4 + k][r] = val
                    
            # Calculate W
            # W[k, i, r] is coeff of C[i, k] in the sum
            # C[i, k] is in block C_block[p][q] where p=i//2, q=k//2
            # C_block[p][q] = sum_s c_s_pq M_s
            # M_s = sum_t W2x2[p][q][t] M_t (scalar product)
            # So W[k, i, r] = C_coeffs[p][q][s] * W2x2[p][q][t]
            
            p = i // 2
            q = k // 2
            
            # Note: W index is k, i. C index is i, k.
            # So for W[k, i, r], we look at C[i, k].
            # Block is p, q.
            
            for i in range(4):
                for k in range(4):
                    p = i // 2
                    q = k // 2
                    val = C_coeffs[p][q][s] * W2x2[p][q][t]
                    W[k * 4 + i][r] = val
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
