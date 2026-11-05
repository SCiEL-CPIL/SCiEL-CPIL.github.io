import numpy as np

def construct():
    n, m, p = 3, 3, 6
    
    # Rank 23 Winograd matrices for 3x3x3
    # U: 9x23, V: 9x23, W: 9x23
    # Indices: A[i*m+j], B[j*p+k], C[k*n+i]
    
    # I will hardcode the Winograd 3x3 matrices.
    # Source: "Fast Matrix Multiplication" or equivalent tensor decomposition databases.
    
    # The following matrices U, V, W define the 23-term scheme.
    # U[i,j,r] contributes to A'_r from A[i,j]
    # V[j,k,r] contributes to B'_r from B[j,k]
    # W[k,i,r] contributes to C[k,i] from A'_r * B'_r
    
    # Note: The specific values depend on the ordering of the 23 terms.
    # I will use a known valid set.
    
    U_33 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    # This is just a guess for the first few columns. I need the full 23 columns.
    
    # Actually, I can generate the Winograd matrices programmatically or use a known complete set.
    # A complete set for 3x3x3 rank 23 is:
    
    U = np.zeros((9, 23), dtype=int)
    V = np.zeros((9, 23), dtype=int)
    W = np.zeros((9, 23), dtype=int)
    
    # I will use the matrices from the "Strassen-Winograd" algorithm.
    # The specific matrices are:
    
    # Columns 0-8: Basic terms? No, Winograd is specific.
    
    # Let's use a verified set.
    # Reference: https://github.com/... or similar tensor decomposition code.
    
    # Since I cannot browse, I will rely on the fact that I can construct a valid 46-rank decomposition
    # if I have a valid 23-rank decomposition.
    
    # I will use the following known Winograd 3x3 matrices (U, V, W are 9x23):
    
    U[0, 0] = 1
    U[1, 0] = 1
    U[2, 0] = 1
    
    U[0, 1] = 1
    U[1, 1] = 0
    U[2, 1] = 1
    
    # This is getting risky to hardcode from memory.
    # Alternative: Use the rank 54 schoolbook if I can't guarantee correctness of Winograd.
    # But 46 is a significant improvement.
    
    # Let's try to implement a simple search for a rank 46 decomposition?
    # No, search is too slow for 120s.
    
    # I will use the known Winograd matrices.
    # Here is a standard set:
    
    U_data = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is only 9 columns. I need 23.
    
    # I will return the schoolbook decomposition to be safe, as I cannot guarantee the Winograd matrices from memory.
    # Score 1.0 is acceptable if I can't improve it reliably.
    
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}
