import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 24: 2 blocks of (7 Strassen + 4 Standard + 1 Outer)
    rank = 24
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Strassen factors for 2x2x2
    S_U = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    S_V = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    S_W = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    # Standard factors for 2x2x1 (A 2x2, B 2x1, C 2x1)
    # C_i0 = A_i0 B_00 + A_i1 B_10
    Std2_U = [
        np.array([[1, 0], [0, 0]]), # A00
        np.array([[0, 0], [1, 0]]), # A10
        np.array([[0, 1], [0, 0]]), # A01
        np.array([[0, 0], [0, 1]])  # A11
    ]
    Std2_V = [
        np.array([[1], [0]]), # B00
        np.array([[1], [0]]), # B00
        np.array([[0], [1]]), # B10
        np.array([[0], [1]])  # B10
    ]
    Std2_W = [
        np.array([[1, 0]]), # C00
        np.array([[0, 1]]), # C10
        np.array([[1, 0]]), # C00
        np.array([[0, 1]])  # C10
    ]
    
    # Outer product factors for 2x1x3 (A 2x1, B 1x3, C 2x3)
    # C_ik = A_i0 B_0k
    Outer_U = [
        np.array([[1], [0]]), # A00
        np.array([[0], [1]])  # A10
    ]
    Outer_V = [
        np.array([[1, 1, 1]]), # B00, B01, B02 all 1? No.
        # We need C_ik = A_i B_k.
        # If we use rank 1, U=[1,0], V=[1,1,1], W=1?
        # C_0k = 1 * B_k * 1 = B_k.
        # C_1k = 0 * B_k * 1 = 0.
        # This only gives row 0.
        # We need rank 2?
        # C_0k = A_0 B_k.
        # C_1k = A_1 B_k.
        # This is rank 2?
        # r=1: U=[1,0], V=[1,1,1], W=[1,0]. C_0k = B_k.
        # r=2: U=[0,1], V=[1,1,1], W=[0,1]. C_1k = B_k.
        # This gives C = [B; B]. We want [A_0 B; A_1 B].
        # So U must scale by A.
        # But U is constant!
        # Ah, U, V, W are constant matrices. A and B are inputs.
        # So we cannot scale by A in U.
        # The decomposition is C = sum U_r A V_r W_r? No.
        # C_ik = sum U_ij,r V_jk,r W_ki,r.
        # For 2x1x3:
        # C_ik = A_i0 B_0k.
        # We need C_ik = sum_r U_i0,r V_0k,r W_ki,r.
        # If we set W_ki,r = delta_{ki}? No, W is p x n.
        # If we use rank 2:
        # r=1: U_00,1=1, U_10,1=0. V_0k,1=1. W_k0,1=1, W_k1,1=0.
        # C_0k = 1 * B_k * 1 = B_k.
        # C_1k = 0 * B_k * 0 = 0.
        # r=2: U_00,2=0, U_10,2=1. V_0k,2=1. W_k0,2=0, W_k1,2=1.
        # C_0k = 0.
        # C_1k = 1 * B_k * 1 = B_k.
        # Result C = [B; B].
        # But we want C = [A_0 B; A_1 B].
        # This implies we need to multiply by A.
        # But A is in U? No, U is constant.
        # The formula is C_ik = sum_j A_ij B_jk.
        # In tensor decomposition, U, V, W are the *structure* of the multiplication.
        # The values A, B are plugged into the "slots" of U, V.
        # Wait, the interface is U, V, W matrices.
        # The evaluator computes C = sum_r (A * U_r) (V_r * B) W_r?
        # No, the standard definition is:
        # C_ik = sum_r U_{ir} (A * V_{r})_{?} ...
        # Actually, the task says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)".
        # And "exactly decompose the matrix multiplication tensor".
        # This means T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
        # Where T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ki}.
        # For 2x1x3:
        # C_{k i} = A_{i 0} B_{0 k}.
        # So T_{i 0 k} = 1 if i=0,k=0? No.
        # T_{i j k} is 1 if j=0.
        # So we need sum_r U_{i 0, r} V_{0 k, r} W_{k i, r} = 1 for all i,k.
        # This is a system of equations.
        # We found that rank 2 gives C=[B;B] if U,V are identity-like.
        # But we need C = A B.
        # The "1" in T_{i0k} means the coefficient is 1.
        # So we just need to represent the tensor T where T_{i0k}=1.
        # T is a 2x1x3 tensor with all 1s.
        # Rank of all-1s tensor 2x1x3 is 1?
        # T_{i0k} = 1.
        # U_{i0,r}=1, V_{0k,r}=1, W_{ki,r}=1.
        # sum_r 1*1*1 = 1 (if r=1).
        # Yes, rank 1.
        # So Outer_U = [[1], [1]], Outer_V = [[1,1,1]], Outer_W = [[1,1],[1,1]]?
        # W is p x n = 3 x 2.
        # W_{k i} = 1.
        # So Outer_W = [[1,1],[1,1],[1,1]].
        
    ]
    
    col = 0
    for b in range(2):
        # Block 1: A[0:2, 0:2] * B[0:2, 3b:3b+3]
        # Strassen for A[0:2, 0:2] * B[0:2, 3b:3b+2]
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i, j]
            for j in range(2):
                for kp in range(2):
                    k_global = 3 * b + kp
                    V[j * p + k_global, col] = S_V[r][j, kp]
            for kp in range(2):
                for i in range(2):
                    k_global = 3 * b + kp
                    W[k_global * n + i, col] = S_W[r][kp, i]
            col += 1
            
        # Standard for A[0:2, 0:2] * B[0:2, 3b+2]
        for r in range(4):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = Std2_U[r][i, j]
            for j in range(2):
                k_global = 3 * b + 2
                V[j * p + k_global, col] = Std2_V[r][j, 0]
            for i in range(2):
                k_global = 3 * b + 2
                W[k_global * n + i, col] = Std2_W[r][0, i]
            col += 1
            
        # Outer for A[0:2, 2] * B[2, 3b:3b+3]
        # A is 2x1, B is 1x3.
        # T_{i 2 k} = 1.
        # U_{i 2, r} = 1.
        # V_{2 k, r} = 1.
        # W_{k i, r} = 1.
        U[0 * m + 2, col] = 1
        U[1 * m + 2, col] = 1
        for k in range(3):
            V[2 * p + 3 * b + k, col] = 1
        for k in range(3):
            for i in range(2):
                W[(3 * b + k) * n + i, col] = 1
        col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
