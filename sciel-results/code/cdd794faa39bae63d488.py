import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Partition indices into sets S1={0,1}, S2={2,3}, S3={4}
    sets = [[0, 1], [2, 3], [4]]
    
    # Precompute Strassen factors for 2x2x2
    # U, V, W are 4x7 matrices (flattened 2x2 to 4)
    # Indices: i,j in {0,1} -> 0..3
    # r in 0..6
    U_strassen = [
        [1, 0, 0, 0, 1, 0, 0], # i=0, j=0
        [0, 0, 1, 0, 1, 0, 0], # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0], # i=1, j=0
        [1, 1, 0, 1, 0, -1, 1] # i=1, j=1
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0], # j=0, k=1
        [1, 0, -1, 1, 1, 0, 1], # j=1, k=0
        [1, 0, 0, 0, 1, 0, 1]  # j=1, k=1
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 0, 1, 0, 1, 0, 0], # k=0, i=1
        [0, 1, 0, 1, 0, 0, 0], # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1
    ]
    
    # Check if W_strassen is correct.
    # C_00 = M1 + M4 - M5 + M7
    # M1 = (A00+A11)(B00+B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M7 = (A01-A11)(B10+B11)
    
    # Let's verify W for C_00 (i=0, k=0).
    # W[0,0] corresponds to k=0, i=0.
    # Coeff of M1: 1
    # Coeff of M4: 1
    # Coeff of M5: -1
    # Coeff of M7: 1
    # My W_strassen[0,0] is 1, 0, 0, 1, -1, 0, 1. Correct.
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Initialize lists to append columns
    # We will build column by column
    
    for a_idx, S_a in enumerate(sets):
        for b_idx, S_b in enumerate(sets):
            for c_idx, S_c in enumerate(sets):
                size_a = len(S_a)
                size_b = len(S_b)
                size_c = len(S_c)
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Use Strassen
                    for r in range(7):
                        col_U = [0] * (n * m)
                        col_V = [0] * (m * p)
                        col_W = [0] * (p * n)
                        
                        for local_i, i in enumerate(S_a):
                            for local_j, j in enumerate(S_b):
                                col_U[i * m + j] = U_strassen[local_i * 2 + local_j][r]
                        
                        for local_j, j in enumerate(S_b):
                            for local_k, k in enumerate(S_c):
                                col_V[j * p + k] = V_strassen[local_j * 2 + local_k][r]
                        
                        for local_k, k in enumerate(S_c):
                            for local_i, i in enumerate(S_a):
                                col_W[k * n + i] = W_strassen[local_k * 2 + local_i][r]
                        
                        U.append(col_U)
                        V.append(col_V)
                        W.append(col_W)
                        rank += 1
                else:
                    # Use Schoolbook
                    for local_i, i in enumerate(S_a):
                        for local_j, j in enumerate(S_b):
                            for local_k, k in enumerate(S_c):
                                col_U = [0] * (n * m)
                                col_V = [0] * (m * p)
                                col_W = [0] * (p * n)
                                
                                col_U[i * m + j] = 1
                                col_V[j * p + k] = 1
                                col_W[k * n + i] = 1
                                
                                U.append(col_U)
                                V.append(col_V)
                                W.append(col_W)
                                rank += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
