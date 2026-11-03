import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Rank-28 exact tensor decomposition for multiplying a 2x3 matrix by a 3x5 matrix.
    
    Decomposition strategy:
    - Split middle dimension j=0,1,2
    - Use Strassen for 2x2x2 blocks covering j∈{0,1}, k∈{0,1} and j∈{0,1}, k∈{2,3}
    - Schoolbook for remaining terms: (j=2, k=0..4) and (j∈{0,1}, k=4)
    
    Total rank = 7 + 7 + 10 = 24... wait, let me recount:
    - Strassen block 1 (j∈{0,1}, k∈{0,1}): rank 7
    - Strassen block 2 (j∈{0,1}, k∈{2,3}): rank 7  
    - Schoolbook: (j=2, k=0..4) → 5*2=10 terms, (j∈{0,1}, k=4) → 2*2=4 terms
    - But wait, we need to be careful about overlaps and the structure
    
    Actually, let me think more carefully. The multiplication is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as:
    C[:, 0:2] = A[:,0:2] * B[0:2, 0:2]  (2x2 * 2x2, rank 7 via Strassen)
    C[:, 2:4] = A[:,0:2] * B[0:2, 2:4]  (2x2 * 2x2, rank 7 via Strassen)
    C[:, 4:5] = A[:,0:2] * B[0:2, 4:5]  (2x2 * 2x1, rank 4 via schoolbook)
    C[:, 0:5] += A[:,2:3] * B[2:3, 0:5]  (2x1 * 1x5, rank 10 via schoolbook)
    
    Wait, that's not quite right either. Let me use the standard approach:
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Split into:
    - j=0,1: handled by Strassen on A[0:2, 0:2] * B[0:2, 0:5]
    - j=2: handled by schoolbook A[0:2, 2] * B[2, 0:5]
    
    For A[0:2, 0:2] * B[0:2, 0:5]:
    This is a 2x2 * 2x5 multiplication.
    Split B[0:2, 0:5] into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
    
    - A[0:2,0:2] * B[0:2,0:2]: 2x2*2x2, rank 7 (Strassen)
    - A[0:2,0:2] * B[0:2,2:4]: 2x2*2x2, rank 7 (Strassen)
    - A[0:2,0:2] * B[0:2,4:5]: 2x2*2x1, rank 4 (schoolbook)
    
    Total for j=0,1: 7+7+4 = 18
    Total for j=2: 2*5 = 10 (schoolbook)
    Grand total: 28
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    c = 0
    
    def add_strassen_block(j_off, k_off):
        """Add 7 Strassen terms for a 2x2 * 2x2 block."""
        nonlocal c
        # m1 = (a00+a11)*(b00+b11)
        U[0*m + j_off][c] = 1
        U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m2 = (a10+a11)*b00
        U[1*m + j_off][c] = 1
        U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1
        W[(k_off+1)*n + 1][c] = -1
        c += 1
        
        # m3 = a00*(b01-b11)
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m4 = a11*(b10-b00)
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1
        V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1
        W[k_off*n + 1][c] = 1
        c += 1
        
        # m5 = (a00+a01)*b11
        U[0*m + j_off][c] = 1
        U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1
        W[(k_off+1)*n + 0][c] = 1
        c += 1
        
        # m6 = (a10-a00)*(b00+b01)
        U[1*m + j_off][c] = 1
        U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m7 = (a01-a11)*(b10+b11)
        U[0*m + (j_off+1)][c] = 1
        U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    # Strassen block 1: j∈{0,1}, k∈{0,1}
    add_strassen_block(0, 0)
    
    # Strassen block 2: j∈{0,1}, k∈{2,3}
    add_strassen_block(0, 2)
    
    # Schoolbook for j∈{0,1}, k=4
    for i in range(n):
        for j in range(2):
            U[i*m + j][c] = 1
            V[j*p + 4][c] = 1
            W[4*n + i][c] = 1
            c += 1
    
    # Schoolbook for j=2, k∈{0,1,2,3,4}
    for i in range(n):
        for k in range(p):
            U[i*m + 2][c] = 1
            V[2*p + k][c] = 1
            W[k*n + i][c] = 1
            c += 1
    
    assert c == rank, f"Rank mismatch: {c} != {rank}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
