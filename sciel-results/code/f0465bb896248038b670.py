import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 145

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the 4x5 * 5x5 matrix multiplication.
    """
    # Dimensions
    m, n, p = 4, 5, 5
    
    # Total rank estimate: 
    # 2 blocks of (4x2 * 2x5) -> 2 * 36 = 72
    # 1 block of (4x1 * 1x5) -> 20
    # Total 92
    rank = 92
    
    U = [[0] * rank for _ in range(m * n)]
    V = [[0] * rank for _ in range(n * p)]
    W = [[0] * rank for _ in range(p * m)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # local i, j, k in {0,1}
    # global offsets: off_i, off_j, off_k
    def add_strassen(off_i, off_j, off_k):
        nonlocal col
        # M1 = (a+d)(e+h)
        # a=(0,0), d=(1,1), e=(0,0), h=(1,1)
        # U: a=1, d=1
        # V: e=1, h=1
        # W: C00=1, C11=1
        # Indices:
        # U[(off_i+0)*2 + (off_j+0)] = 1
        # U[(off_i+1)*2 + (off_j+1)] = 1
        # V[(off_j+0)*2 + (off_k+0)] = 1
        # V[(off_j+1)*2 + (off_k+1)] = 1
        # W[(off_k+0)*2 + (off_i+0)] = 1
        # W[(off_k+1)*2 + (off_i+1)] = 1
        
        # Note: The U, V, W arrays are global. 
        # The local Strassen U vector has 4 entries corresponding to the 4 elements of the 2x2 matrix.
        # We need to place these 4 entries into the global U row.
        
        # Strassen U coefficients:
        # r0: [1, 0, 0, 1] (a, b, c, d) -> a=1, d=1
        # r1: [0, 0, 1, 1] (c, d) -> c=1, d=1
        # r2: [1, 0, 0, 0] (a)
        # r3: [0, 0, 0, 1] (d)
        # r4: [1, 1, 0, 0] (a, b)
        # r5: [-1, 0, 1, 0] (c, a)
        # r6: [0, 1, 0, -1] (b, d)
        
        U_coeffs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        
        # Strassen V coefficients (e, f, g, h)
        V_coeffs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        # Strassen W coefficients (C00, C01, C10, C11)
        W_coeffs = [
            [1, 0, 0, 1],
            [0, -1, 0, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        for r in range(7):
            # Fill U
            for li in range(2):
                for lj in range(2):
                    val = U_coeffs[r][li * 2 + lj]
                    if val != 0:
                        gi = off_i + li
                        gj = off_j + lj
                        U[gi * n + gj][col] += val
            
            # Fill V
            for lj in range(2):
                for lk in range(2):
                    val = V_coeffs[r][lj * 2 + lk]
                    if val != 0:
                        gj = off_j + lj
                        gk = off_k + lk
                        V[gj * p + gk][col] += val
            
            # Fill W
            for lk in range(2):
                for li in range(2):
                    val = W_coeffs[r][lk * 2 + li]
                    if val != 0:
                        gk = off_k + lk
                        gi = off_i + li
                        W[gk * m + gi][col] += val
            
            col += 1

    # Helper to add a 2x2x1 schoolbook decomposition
    # local i in {0,1}, j in {0,1}, k=0
    def add_2x2x1(off_i, off_j, off_k):
        nonlocal col
        # C_i0 = sum_j A_ij B_j0
        # 4 terms
        for li in range(2):
            for lj in range(2):
                # U: A_li, lj = 1
                U[(off_i + li) * n + (off_j + lj)][col] = 1
                # V: B_lj, 0 = 1
                V[(off_j + lj) * p + (off_k + 0)][col] = 1
                # W: C_0, li = 1
                W[(off_k + 0) * m + (off_i + li)][col] = 1
                col += 1

    # Helper to add a 4x1x5 schoolbook decomposition
    # local i in {0,1,2,3}, j=0, k in {0,1,2,3,4}
    def add_4x1x5(off_i, off_j, off_k):
        nonlocal col
        # C_ik = A_i0 * B_0k
        # 20 terms
        for li in range(4):
            for lk in range(5):
                # U: A_li, 0 = 1
                U[(off_i + li) * n + (off_j + 0)][col] = 1
                # V: B_0, lk = 1
                V[(off_j + 0) * p + (off_k + lk)][col] = 1
                # W: C_lk, li = 1
                W[(off_k + lk) * m + (off_i + li)][col] = 1
                col += 1

    # --- Assembly ---
    
    # Block 1: n in {0, 1}
    # Top: i in {0, 1}, j in {0, 1}
    # Bottom: i in {2, 3}, j in {0, 1}
    
    # Top part of Block 1 (2x2 * 2x5)
    # Split k in {0,1}, {2,3}, {4}
    add_strassen(0, 0, 0) # i=0,1 j=0,1 k=0,1
    add_strassen(0, 0, 2) # i=0,1 j=0,1 k=2,3
    add_2x2x1(0, 0, 4)    # i=0,1 j=0,1 k=4
    
    # Bottom part of Block 1 (2x2 * 2x5)
    add_strassen(2, 0, 0) # i=2,3 j=0,1 k=0,1
    add_strassen(2, 0, 2) # i=2,3 j=0,1 k=2,3
    add_2x2x1(2, 0, 4)    # i=2,3 j=0,1 k=4
    
    # Block 2: n in {2, 3}
    # Top: i in {0, 1}, j in {2, 3}
    # Bottom: i in {2, 3}, j in {2, 3}
    
    # Top part of Block 2
    add_strassen(0, 2, 0) # i=0,1 j=2,3 k=0,1
    add_strassen(0, 2, 2) # i=0,1 j=2,3 k=2,3
    add_2x2x1(0, 2, 4)    # i=0,1 j=2,3 k=4
    
    # Bottom part of Block 2
    add_strassen(2, 2, 0) # i=2,3 j=2,3 k=0,1
    add_strassen(2, 2, 2) # i=2,3 j=2,3 k=2,3
    add_2x2x1(2, 2, 4)    # i=2,3 j=2,3 k=4
    
    # Block 3: n in {4}
    # i in {0,1,2,3}, j=4, k in {0,1,2,3,4}
    add_4x1x5(0, 4, 0)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
