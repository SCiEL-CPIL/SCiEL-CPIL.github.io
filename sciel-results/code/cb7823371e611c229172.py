import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 6 x 16
    V = [[0] * rank for _ in range(m * p)]  # 15 x 16
    W = [[0] * rank for _ in range(p * n)]  # 10 x 16
    
    col = 0
    
    def add_strassen_2x2x2(k_off):
        """Add a rank-7 Strassen decomposition for i∈{0,1}, j∈{0,1,2}, k∈{k_off, k_off+1}."""
        nonlocal col
        # M1: (A_i0_j0 + A_i1_j1) * (B_j0_k0 + B_j1_k1)
        U[0*m+0][col] = 1
        U[1*m+1][col] = 1
        V[0*p+k_off][col] = 1
        V[1*p+k_off+1][col] = 1
        W[k_off*n+0][col] = 1
        W[(k_off+1)*n+1][col] = 1
        col += 1
        
        # M2: (A_i1_j0 + A_i1_j1) * B_j0_k0
        U[1*m+0][col] = 1
        U[1*m+1][col] = 1
        V[0*p+k_off][col] = 1
        W[k_off*n+1][col] = 1
        W[(k_off+1)*n+1][col] = -1
        col += 1
        
        # M3: A_i0_j0 * (B_j0_k1 - B_j1_k1)
        U[0*m+0][col] = 1
        V[0*p+k_off+1][col] = 1
        V[1*p+k_off+1][col] = -1
        W[(k_off+1)*n+0][col] = 1
        W[(k_off+1)*n+1][col] = 1
        col += 1
        
        # M4: A_i1_j1 * (B_j1_k0 - B_j0_k0)
        U[1*m+1][col] = 1
        V[1*p+k_off][col] = 1
        V[0*p+k_off][col] = -1
        W[k_off*n+0][col] = 1
        W[k_off*n+1][col] = 1
        col += 1
        
        # M5: (A_i0_j0 + A_i0_j1) * B_j1_k1
        U[0*m+0][col] = 1
        U[0*m+1][col] = 1
        V[1*p+k_off+1][col] = 1
        W[k_off*n+0][col] = -1
        W[(k_off+1)*n+0][col] = 1
        col += 1
        
        # M6: (A_i1_j0 - A_i0_j0) * (B_j0_k0 + B_j0_k1)
        U[1*m+0][col] = 1
        U[0*m+0][col] = -1
        V[0*p+k_off][col] = 1
        V[0*p+k_off+1][col] = 1
        W[(k_off+1)*n+1][col] = 1
        col += 1
        
        # M7: (A_i0_j1 - A_i1_j1) * (B_j1_k0 + B_j1_k1)
        U[0*m+1][col] = 1
        U[1*m+1][col] = -1
        V[1*p+k_off][col] = 1
        V[1*p+k_off+1][col] = 1
        W[k_off*n+0][col] = 1
        col += 1
    
    # Block 1: k = (0, 1), Strassen rank 7
    add_strassen_2x2x2(0)
    
    # Block 2: k = (2, 3), Strassen rank 7
    add_strassen_2x2x2(2)
    
    # Block 3: k = 4, rank 2
    # Term 1: row i=0
    for j in range(m):
        U[0*m+j][col] = 1
    for j in range(m):
        V[j*p+4][col] = 1
    W[4*n+0][col] = 1
    col += 1
    
    # Term 2: row i=1
    for j in range(m):
        U[1*m+j][col] = 1
    for j in range(m):
        V[j*p+4][col] = 1
    W[4*n+1][col] = 1
    col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
