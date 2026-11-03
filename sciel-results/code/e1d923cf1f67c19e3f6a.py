"""Matrix multiplication 2,3,5 — rank-28 decomposition via two Strassen 2x2x2 blocks."""
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-28 ansatz.
    
    Strategy: Use Strassen's 7-term decomposition for two disjoint 2x2x2 blocks.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    The remaining 14 terms are handled by schoolbook multiplication.
    Total rank: 7 + 7 + 14 = 28.
    
    Index conventions:
      U[i*m + j, r]  — coefficient for A[i,j] in rank-r term
      V[j*p + k, r]  — coefficient for B[j,k] in rank-r term
      W[k*n + i, r]  — coefficient for C[i,k] in rank-r term (trace convention W[k*n+i,r])
    """
    deadline = time.time() + 145  # safety margin before 150s kill
    
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    def add_strassen_block(j_off, k_off):
        """Add a 7-term Strassen decomposition for a 2x2x2 sub-block.
        
        Sub-block covers: i in {0,1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        """
        nonlocal c
        
        # m1 = (a[0][j0] + a[1][j1]) * (b[j0][k0] + b[j1][k1])
        # C[0][k0] += m1, C[1][k1] += m1
        U[0*m + j_off][c] = 1
        U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m2 = (a[1][j0] + a[1][j1]) * b[j0][k0]
        # C[1][k0] += m2, C[1][k1] -= m2
        U[1*m + j_off][c] = 1
        U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1
        W[(k_off+1)*n + 1][c] = -1
        c += 1
        
        # m3 = a[0][j0] * (b[j0][k1] - b[j1][k1])
        # C[0][k1] += m3, C[1][k1] += m3
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m4 = a[1][j1] * (b[j1][k0] - b[j0][k0])
        # C[0][k0] += m4, C[1][k0] += m4
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1
        V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1
        W[k_off*n + 1][c] = 1
        c += 1
        
        # m5 = (a[0][j0] + a[0][j1]) * b[j1][k1]
        # C[0][k0] -= m5, C[0][k1] += m5
        U[0*m + j_off][c] = 1
        U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1
        W[(k_off+1)*n + 0][c] = 1
        c += 1
        
        # m6 = (a[1][j0] - a[0][j0]) * (b[j0][k0] + b[j0][k1])
        # C[1][k1] += m6
        U[1*m + j_off][c] = 1
        U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m7 = (a[0][j1] - a[1][j1]) * (b[j1][k0] + b[j1][k1])
        # C[0][k0] += m7
        U[0*m + (j_off+1)][c] = 1
        U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1
    
    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(0, 2)
    
    # Remaining 14 schoolbook terms
    # Covered by Strassen: j in {0,1} and k in {0,1,2,3} → 2*2*4 = 16 terms
    # Remaining: j in {0,1}, k in {4} → 2*1 = 2 terms
    #            j in {2},   k in {0,1,2,3,4} → 2*5 = 10 terms
    # Total remaining: 12... wait, let me recount.
    # Total terms: 2*3*5 = 30
    # Block 1 covers: i∈{0,1}, j∈{0,1}, k∈{0,1} → 2*2*2 = 8 terms
    # Block 2 covers: i∈{0,1}, j∈{0,1}, k∈{2,3} → 2*2*2 = 8 terms
    # Total covered: 16 terms
    # Remaining: 30 - 16 = 14 terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1):
                    continue  # Block 1
                if j in (0, 1) and k in (2, 3):
                    continue  # Block 2
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
