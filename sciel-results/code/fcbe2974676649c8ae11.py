import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2x2 block
    # i_off, j_off, k_off are the global offsets for the local 2x2x2 block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11
        U[i_off * m + j_off][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        # V: B00, B11
        V[j_off * p + k_off][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        # W: C00, C11
        W[k_off * n + i_off][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        # U: A10, A11
        U[(i_off + 1) * m + j_off][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        # V: B00
        V[j_off * p + k_off][col] = 1
        # W: C10, C11
        W[k_off * n + (i_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = -1
        col += 1
        
        # M3 = A00(B01 - B11)
        # U: A00
        U[i_off * m + j_off][col] = 1
        # V: B01, B11
        V[j_off * p + (k_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = -1
        # W: C01, C11
        W[(k_off + 1) * n + i_off][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M4 = A11(B10 - B00)
        # U: A11
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        # V: B10, B00
        V[(j_off + 1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        # W: C00, C10
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off + 1)][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        # U: A00, A01
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off + 1)][col] = 1
        # V: B11
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        # W: C00, C01
        W[k_off * n + i_off][col] = -1
        W[(k_off + 1) * n + i_off][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # U: A10, A00
        U[(i_off + 1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        # V: B00, B01
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off + 1)][col] = 1
        # W: C11
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # U: A01, A11
        U[i_off * m + (j_off + 1)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = -1
        # V: B10, B11
        V[(j_off + 1) * p + k_off][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        # W: C00
        W[k_off * n + i_off][col] = 1
        col += 1

    # Block 1: j=0 (2x1x5)
    # i in {0,1}, j=0, k in {0..4}
    for i in range(n):
        for k in range(p):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2 (2x2x5)
    # Split k into {0,1} and {2,3,4}
    
    # Block 2a: k=0,1 (2x2x2) -> Strassen
    # i_off=0, j_off=1, k_off=0
    add_strassen(0, 1, 0)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Block 2b: k=2,3,4 (2x2x3)
    # Split into k=2,3 (2x2x2) and k=4 (2x2x1)
    
    # Block 2b1: k=2,3 (2x2x2) -> Strassen
    # i_off=0, j_off=1, k_off=2
    add_strassen(0, 1, 2)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Block 2b2: k=4 (2x2x1) -> Schoolbook Rank 4
    # i in {0,1}, j in {1,2}, k=4
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
